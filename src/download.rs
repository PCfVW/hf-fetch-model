// SPDX-License-Identifier: MIT OR Apache-2.0

//! Download orchestration for `HuggingFace` model repositories.
//!
//! This module coordinates the download of all files in a model
//! repository using `hf-hub`'s high-throughput mode, with concurrent
//! file downloads, filtering, progress reporting, retry, checksum
//! verification, and timeouts.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use hf_hub::api::tokio::ApiRepo;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::checksum;
use crate::chunked;
use crate::config::{file_matches, FetchConfig, ProgressCallback};
use crate::error::{FetchError, FileFailure};
use crate::progress;
use crate::repo::{self, RepoFile};
use crate::retry::{self, RetryPolicy};

/// Default timeout per file when no config is provided (5 minutes).
const DEFAULT_TIMEOUT_PER_FILE: Duration = Duration::from_secs(300);

// ---------------------------------------------------------------------------
// DownloadOutcome — cache vs network result indicator
// ---------------------------------------------------------------------------

/// Indicates whether files were resolved from local cache or freshly downloaded.
///
/// Wraps the result value (a path or file map) so callers can distinguish
/// between a cache hit (zero network requests) and a network download.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum DownloadOutcome<T> {
    /// All requested files were found in the local cache (no network requests).
    Cached(T),
    /// Files were downloaded from the network (or a mix of cache and network).
    Downloaded(T),
}

impl<T> DownloadOutcome<T> {
    /// Returns the inner value regardless of cache/download origin.
    #[must_use]
    pub fn into_inner(self) -> T {
        match self {
            Self::Cached(v) | Self::Downloaded(v) => v,
        }
    }

    /// Returns `true` if the result came entirely from local cache.
    #[must_use]
    pub fn is_cached(&self) -> bool {
        matches!(self, Self::Cached(_))
    }

    /// Returns a reference to the inner value.
    #[must_use]
    pub fn inner(&self) -> &T {
        match self {
            Self::Cached(v) | Self::Downloaded(v) => v,
        }
    }
}

// ---------------------------------------------------------------------------
// DownloadSettings — resolved config parameters
// ---------------------------------------------------------------------------

/// Resolved download parameters extracted from [`FetchConfig`].
///
/// Groups all config-derived values controlling download behavior,
/// avoiding repetitive option unpacking in the download pipeline.
#[derive(Clone)]
struct DownloadSettings {
    /// Retry policy for transient failures.
    retry_policy: RetryPolicy,
    /// Per-file timeout.
    timeout_per_file: Duration,
    /// Overall timeout for the entire batch.
    timeout_total: Option<Duration>,
    /// Maximum concurrent downloads.
    concurrency: usize,
    /// Connections per chunked download.
    connections_per_file: usize,
    /// File size threshold for multi-connection chunked downloads.
    chunk_threshold: u64,
    /// Whether to verify SHA256 checksums after download.
    verify_checksums: bool,
}

impl DownloadSettings {
    /// Builds settings from optional config, using sensible defaults.
    fn from_config(config: Option<&FetchConfig>) -> Self {
        Self {
            retry_policy: RetryPolicy {
                max_retries: config.map_or(3, |c| c.max_retries),
                ..RetryPolicy::default()
            },
            timeout_per_file: config
                .and_then(|c| c.timeout_per_file)
                .unwrap_or(DEFAULT_TIMEOUT_PER_FILE),
            timeout_total: config.and_then(|c| c.timeout_total),
            concurrency: config.map_or(4, |c| c.concurrency).max(1),
            connections_per_file: config.map_or(8, |c| c.connections_per_file),
            chunk_threshold: config.map_or(u64::MAX, |c| c.chunk_threshold),
            verify_checksums: config.is_some_and(|c| c.verify_checksums),
        }
    }
}

// ---------------------------------------------------------------------------
// Public download entry points
// ---------------------------------------------------------------------------

/// Downloads all files from a repository and returns the cache directory.
///
/// Each file is downloaded via `hf-hub`'s `.get()` method, which respects
/// the `HuggingFace` cache layout (`~/.cache/huggingface/hub/`).
///
/// - **Concurrency**: downloads up to `concurrency` files in parallel (auto-tuned by the plan optimizer, or 4 fallback).
/// - **Resume**: `hf-hub` skips already-cached files automatically.
/// - **Retry**: transient failures are retried with exponential backoff + jitter.
/// - **Checksum**: SHA256 verification against `HuggingFace` LFS metadata.
/// - **Timeout**: per-file and overall time limits.
/// - **Structured errors**: partial failures reported via [`FetchError::PartialDownload`].
///
/// # Errors
///
/// Returns [`FetchError::PartialDownload`] if some files fail and others succeed.
/// Returns [`FetchError::Api`] if the file listing fails.
/// Returns [`FetchError::RepoNotFound`] if the repository does not exist.
/// Returns [`FetchError::NoFilesMatched`] if the repository is empty or all files were filtered out.
/// Returns [`FetchError::Timeout`] if the overall timeout is exceeded.
pub async fn download_all_files(
    repo: ApiRepo,
    repo_id: String,
    config: Option<&FetchConfig>,
) -> Result<DownloadOutcome<PathBuf>, FetchError> {
    // BORROW: clone before move into download_all_files_map for error context
    let repo_id_for_error = repo_id.clone();
    let outcome = download_all_files_map(repo, repo_id, config).await?;
    let was_cached = outcome.is_cached();

    // Extract the snapshot directory from any downloaded file path.
    // All files in a repo share the same snapshot directory.
    // hf-hub cache layout: .cache/huggingface/hub/models--org--name/snapshots/<sha>/<relative_path>
    let file_map = outcome.into_inner();
    let (filename, path) =
        file_map
            .into_iter()
            .next()
            .ok_or_else(|| FetchError::NoFilesMatched {
                repo_id: repo_id_for_error,
            })?;

    let root = snapshot_root(&filename, &path);
    if was_cached {
        Ok(DownloadOutcome::Cached(root))
    } else {
        Ok(DownloadOutcome::Downloaded(root))
    }
}

/// Downloads all files from a repository and returns a filename → path map.
///
/// Each key is the relative filename within the repository (e.g.,
/// `"config.json"`, `"model.safetensors"`), and each value is the
/// absolute local path to the downloaded file.
///
/// # Errors
///
/// Returns [`FetchError::PartialDownload`] if some files fail and others succeed.
/// Returns [`FetchError::Api`] if the file listing fails.
/// Returns [`FetchError::RepoNotFound`] if the repository does not exist.
/// Returns [`FetchError::NoFilesMatched`] if the repository is empty or all files were filtered out.
/// Returns [`FetchError::Timeout`] if the overall timeout is exceeded.
pub async fn download_all_files_map(
    repo: ApiRepo,
    repo_id: String,
    config: Option<&FetchConfig>,
) -> Result<DownloadOutcome<HashMap<String, PathBuf>>, FetchError> {
    let overall_start = tokio::time::Instant::now();

    // Check local cache first — return immediately if all files are present (no network).
    if let Some(file_map) = try_resolve_repo_from_cache(config, repo_id.as_str())? {
        return Ok(DownloadOutcome::Cached(file_map));
    }

    // Cache miss — list files from the network.
    tracing::debug!(repo_id = %repo_id, "listing repository files");
    let include = config.and_then(|c| c.include.as_ref());
    let exclude = config.and_then(|c| c.exclude.as_ref());
    let all_files = repo::list_repo_files(&repo, repo_id.clone()).await?;
    let files: Vec<_> = all_files
        .into_iter()
        // BORROW: explicit .as_str() instead of Deref coercion
        .filter(|f| file_matches(f.filename.as_str(), include, exclude))
        .collect();

    // Build download settings and shared HTTP client.
    let mut settings = DownloadSettings::from_config(config);
    let on_progress = config.and_then(|c| c.on_progress.clone());
    let token_ref = config.and_then(|c| c.token.as_deref());
    let http_client = Arc::new(chunked::build_client(token_ref)?);

    // Fetch metadata using the shared client.
    let metadata_map = fetch_metadata_if_needed(
        config,
        repo_id.as_str(),
        settings.verify_checksums,
        settings.chunk_threshold,
        &http_client,
    )
    .await;

    // Build remaining shared state, reusing the HTTP client.
    let (chunked_client, cache_dir, repo_folder, revision, token) =
        build_shared_state(config, repo_id.as_str(), &settings, &http_client)?;

    // Implicit plan optimization: compute a lightweight plan from the
    // already-fetched metadata and merge recommended settings into any
    // fields the user did not explicitly set.
    merge_plan_recommended(
        &mut settings,
        config,
        &files,
        &metadata_map,
        &cache_dir,
        &repo_folder,
        &revision,
    );

    let total = files.len();
    tracing::debug!(
        total_files = total,
        concurrency = settings.concurrency,
        "download settings (after plan optimization)"
    );

    // Check available disk space before starting downloads.
    check_disk_space(
        &cache_dir,
        &files,
        &metadata_map,
        repo_folder.as_str(),
        revision.as_str(),
    );

    // Spawn concurrent download tasks.
    let repo = Arc::new(repo);
    let metadata_map = Arc::new(metadata_map);
    let semaphore = Arc::new(Semaphore::new(settings.concurrency));
    let completed = Arc::new(AtomicUsize::new(0));
    let mut join_set = JoinSet::new();

    for file in files {
        if let Some(total_limit) = settings.timeout_total {
            if overall_start.elapsed() >= total_limit {
                join_set.abort_all();
                return Err(FetchError::Timeout {
                    filename: file.filename,
                    seconds: total_limit.as_secs(),
                });
            }
        }

        let permit = Arc::clone(&semaphore)
            .acquire_owned()
            .await
            .map_err(|e| FetchError::Http(e.to_string()))?;

        let task_repo = Arc::clone(&repo);
        let task_meta = Arc::clone(&metadata_map);
        let task_chunked_client = chunked_client.clone();
        let task_http_client = Arc::clone(&http_client);
        let task_cache_dir = cache_dir.clone();
        let task_repo_folder = Arc::clone(&repo_folder);
        let task_revision = Arc::clone(&revision);
        // BORROW: explicit .clone() for repo_id
        let task_repo_id = repo_id.clone();
        let task_token = Arc::clone(&token);
        let task_settings = settings.clone();
        let task_on_progress = on_progress.clone();
        let task_completed = Arc::clone(&completed);

        join_set.spawn(async move {
            let result = dispatch_download(
                &task_repo,
                &file,
                &task_meta,
                task_chunked_client.as_deref(),
                &task_http_client,
                &task_cache_dir,
                &task_repo_folder,
                &task_revision,
                task_repo_id.as_str(),
                (*task_token).clone(),
                &task_settings,
                task_on_progress,
                total.saturating_sub(task_completed.load(Ordering::Relaxed) + 1),
            )
            .await;
            drop(permit);
            (file, result)
        });
    }

    // Collect results and check for failures.
    let (file_map, failures) = collect_results(
        &mut join_set,
        settings.timeout_total,
        overall_start,
        on_progress.as_ref(),
        total,
        &completed,
    )
    .await?;

    let file_map = validate_download_results(file_map, failures, repo_id.as_str())?;
    tracing::debug!(files_downloaded = file_map.len(), "download complete");
    Ok(DownloadOutcome::Downloaded(file_map))
}

// ---------------------------------------------------------------------------
// Single-file download methods
// ---------------------------------------------------------------------------

/// Downloads a single file with retry and timeout, then optionally verifies its checksum.
async fn download_single_file(
    repo: &ApiRepo,
    file: &RepoFile,
    metadata_map: &HashMap<String, RepoFile>,
    verify_checksums: bool,
    retry_policy: &RetryPolicy,
    timeout: Duration,
) -> Result<PathBuf, FetchError> {
    // BORROW: explicit .clone() for owned String in closure
    let filename = file.filename.clone();

    // Download with retry.
    let path = retry::retry_async(retry_policy, retry::is_retryable, || {
        let fname = filename.clone();
        let timeout_dur = timeout;
        async move {
            // BORROW: explicit .as_str() instead of Deref coercion
            let download_fut = repo.get(fname.as_str());
            tokio::time::timeout(timeout_dur, download_fut)
                .await
                .map_err(|_elapsed| FetchError::Timeout {
                    filename: fname.clone(),
                    seconds: timeout_dur.as_secs(),
                })?
                .map_err(FetchError::Api)
        }
    })
    .await?;

    // Verify SHA256 if enabled and metadata is available.
    // BORROW: explicit .as_str() instead of Deref coercion
    if verify_checksums {
        if let Some(meta) = metadata_map.get(file.filename.as_str()) {
            if let Some(ref expected_sha) = meta.sha256 {
                checksum::verify_sha256(&path, file.filename.as_str(), expected_sha.as_str())
                    .await?;
            }
        }
    }

    Ok(path)
}

/// Downloads a large file using multi-connection chunked download with retry and checksum.
///
/// Before starting the download, checks the CDN signed URL's `X-Amz-Expires`
/// parameter. If the estimated download time exceeds the remaining URL validity,
/// logs a warning and re-probes for a fresh URL.
#[allow(clippy::too_many_arguments)]
async fn download_single_file_chunked(
    client: &reqwest::Client,
    file: &RepoFile,
    cache_dir: &std::path::Path,
    repo_folder: &str,
    revision: &str,
    repo_id: &str,
    token: Option<String>,
    metadata_map: &HashMap<String, RepoFile>,
    verify_checksums: bool,
    retry_policy: &RetryPolicy,
    connections: usize,
    // TRAIT_OBJECT: heterogeneous progress handlers from different callers
    on_progress: Option<ProgressCallback>,
    files_remaining: usize,
) -> Result<PathBuf, FetchError> {
    // Probe for Range support.
    // BORROW: explicit .as_str() for URL construction
    let url = chunked::build_download_url(repo_id, revision, file.filename.as_str());
    // BORROW: explicit .clone() for potential re-probe on CDN URL expiry
    let token_for_reprobe = token.clone();
    let range_info = chunked::probe_range_support(client.clone(), url, token).await?;

    let Some(mut range_info) = range_info else {
        // Range not supported — this shouldn't happen for LFS files, but fall back
        // gracefully. Return an error that will be caught and retried via the standard path.
        return Err(FetchError::ChunkedDownload {
            // BORROW: explicit .clone() for owned String
            filename: file.filename.clone(),
            reason: String::from("server does not support Range requests"),
        });
    };

    // Check if the CDN signed URL has enough remaining validity for the download.
    // Conservative estimate: 5 MB/s per connection.
    if let Some(expires_at) = range_info.cdn_expires_at {
        let remaining = expires_at.saturating_duration_since(std::time::Instant::now());
        let throughput = u64::try_from(connections)
            .unwrap_or(1)
            .saturating_mul(5_000_000);
        let estimated = Duration::from_secs(range_info.content_length / throughput.max(1));
        tracing::debug!(
            filename = %file.filename,
            cdn_expires_in_secs = remaining.as_secs(),
            estimated_download_secs = estimated.as_secs(),
            "CDN URL expiry parsed"
        );
        if remaining < estimated + Duration::from_secs(60) {
            tracing::warn!(
                filename = %file.filename,
                remaining_secs = remaining.as_secs(),
                estimated_secs = estimated.as_secs(),
                "CDN URL may expire before download completes, re-probing for fresh URL"
            );
            let fresh_url = chunked::build_download_url(repo_id, revision, file.filename.as_str());
            if let Some(fresh_info) =
                chunked::probe_range_support(client.clone(), fresh_url, token_for_reprobe).await?
            {
                range_info = fresh_info;
            }
        }
    }

    let path = chunked::download_chunked(
        client.clone(),
        range_info,
        // BORROW: explicit .to_path_buf() for owned PathBuf
        cache_dir.to_path_buf(),
        // BORROW: explicit .to_owned() for owned String
        repo_folder.to_owned(),
        // BORROW: explicit .to_owned() for owned String
        revision.to_owned(),
        // BORROW: explicit .clone() for owned String
        file.filename.clone(),
        connections,
        retry_policy.clone(),
        on_progress,
        files_remaining,
    )
    .await?;

    // Verify SHA256 if enabled and metadata is available.
    // BORROW: explicit .as_str() instead of Deref coercion
    if verify_checksums {
        if let Some(meta) = metadata_map.get(file.filename.as_str()) {
            if let Some(ref expected_sha) = meta.sha256 {
                checksum::verify_sha256(&path, file.filename.as_str(), expected_sha.as_str())
                    .await?;
            }
        }
    }

    Ok(path)
}

/// Downloads a single named file from a repository and returns its cache path.
///
/// This is the single-file counterpart to [`download_all_files_map()`]. It reuses
/// the same download pipeline (chunked or standard, retry, checksum, 416 fallback)
/// via [`dispatch_download()`].
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the file does not exist in the repository.
/// Returns [`FetchError::Api`] on download failure (after retries).
/// Returns [`FetchError::Checksum`] if verification is enabled and fails.
pub(crate) async fn download_file_by_name(
    repo: ApiRepo,
    repo_id: String,
    filename: &str,
    config: &FetchConfig,
) -> Result<DownloadOutcome<PathBuf>, FetchError> {
    // Check local cache first — return immediately if the file is present.
    let cache_dir = config
        .output_dir
        .clone()
        .map_or_else(crate::cache::hf_cache_dir, Ok)?;
    // BORROW: explicit .as_str() instead of Deref coercion
    let repo_folder = crate::cache_layout::repo_folder_name(repo_id.as_str());
    let revision_str = config.revision.as_deref().unwrap_or("main");
    if let Some(cached) =
        resolve_cached_file(&cache_dir, repo_folder.as_str(), revision_str, filename)
    {
        return Ok(DownloadOutcome::Cached(cached));
    }

    let settings = DownloadSettings::from_config(Some(config));
    // BORROW: explicit .clone() for Arc-wrapped callback
    let on_progress = config.on_progress.clone();
    let http_client = chunked::build_client(config.token.as_deref())?;

    let metadata_map = fetch_metadata_if_needed(
        Some(config),
        repo_id.as_str(),
        settings.verify_checksums,
        settings.chunk_threshold,
        &http_client,
    )
    .await;

    // Check disk space for the single file.
    if let Some(size) = metadata_map.get(filename).and_then(|m| m.size) {
        let single_file = RepoFile {
            filename: filename.to_owned(),
            size: Some(size),
            sha256: None,
        };
        check_disk_space(
            &cache_dir,
            &[single_file],
            &metadata_map,
            repo_folder.as_str(),
            revision_str,
        );
    }

    // Build a RepoFile for this filename from metadata (or with no metadata).
    let file_meta = metadata_map.get(filename);
    // BORROW: explicit .to_owned()/.clone() for owned String fields
    let file = RepoFile {
        filename: filename.to_owned(),
        size: file_meta.and_then(|m| m.size),
        sha256: file_meta.and_then(|m| m.sha256.clone()),
    };

    // Build reqwest client (used by chunked downloads and 416 fallback).
    // BORROW: explicit .as_deref() for Option<String> → Option<&str>
    let http_client = chunked::build_client(config.token.as_deref())?;

    // Reuse cache_dir and repo_folder resolved above for the cache check.
    // BORROW: explicit .to_owned() for &str → owned String
    let revision = revision_str.to_owned();

    let chunked_client = if settings.chunk_threshold < u64::MAX {
        Some(&http_client)
    } else {
        None
    };

    let result = dispatch_download(
        &repo,
        &file,
        &metadata_map,
        chunked_client,
        &http_client,
        &cache_dir,
        // BORROW: explicit .as_str() for String → &str conversions
        repo_folder.as_str(),
        revision.as_str(),
        repo_id.as_str(),
        // BORROW: explicit .clone() for owned Option<String>
        config.token.clone(),
        &settings,
        on_progress.clone(),
        0, // files_remaining: only one file
    )
    .await;

    let path = result?;

    // Report progress for the completed file.
    if let Some(ref cb) = on_progress {
        let file_size = tokio::fs::metadata(&path)
            .await
            .map(|m| m.len())
            .unwrap_or(0);
        let event = progress::completed_event(filename, file_size, 0);
        cb(&event);
    }

    Ok(DownloadOutcome::Downloaded(path))
}

// ---------------------------------------------------------------------------
// Shared download helpers (factored from download_all_files_map and
// download_file_by_name to eliminate duplication)
// ---------------------------------------------------------------------------

/// Builds the shared `Arc`-wrapped state needed for concurrent downloads.
///
/// Accepts a pre-built HTTP client to reuse TCP/TLS connections from earlier
/// metadata requests. Returns `(chunked_client, cache_dir, repo_folder, revision, token)`.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the cache directory cannot be resolved.
#[allow(clippy::type_complexity)]
fn build_shared_state(
    config: Option<&FetchConfig>,
    repo_id: &str,
    settings: &DownloadSettings,
    http_client: &Arc<reqwest::Client>,
) -> Result<
    (
        Option<Arc<reqwest::Client>>,
        Arc<PathBuf>,
        Arc<String>,
        Arc<String>,
        Arc<Option<String>>,
    ),
    FetchError,
> {
    let chunked_client = if settings.chunk_threshold < u64::MAX {
        Some(Arc::clone(http_client))
    } else {
        None
    };

    let cache_dir = Arc::new(
        config
            .and_then(|c| c.output_dir.clone())
            .map_or_else(crate::cache::hf_cache_dir, Ok)?,
    );
    // BORROW: explicit .as_str() instead of Deref coercion
    let repo_folder = Arc::new(crate::cache_layout::repo_folder_name(repo_id));
    let revision = Arc::new(
        config
            .and_then(|c| c.revision.clone())
            .unwrap_or_else(|| String::from("main")),
    );
    let token = Arc::new(config.and_then(|c| c.token.clone()));

    Ok((chunked_client, cache_dir, repo_folder, revision, token))
}

/// Downloads a single file, choosing the best method and applying fallbacks.
///
/// This is the core download logic shared by [`download_all_files_map()`]
/// (batch) and [`download_file_by_name()`] (single-file). It:
///
/// 1. Returns immediately if the file exists in the local cache
/// 2. Chooses chunked (multi-connection) or single-connection download
/// 3. Falls back to direct HTTP GET on HTTP 416 Range Not Satisfiable
/// 4. Logs the result with timing and throughput
#[allow(clippy::too_many_arguments)]
async fn dispatch_download(
    repo: &ApiRepo,
    file: &RepoFile,
    metadata_map: &HashMap<String, RepoFile>,
    chunked_client: Option<&reqwest::Client>,
    http_client: &reqwest::Client,
    cache_dir: &std::path::Path,
    repo_folder: &str,
    revision: &str,
    repo_id: &str,
    token: Option<String>,
    settings: &DownloadSettings,
    on_progress: Option<ProgressCallback>,
    files_remaining: usize,
) -> Result<PathBuf, FetchError> {
    // Check local cache first — skip the network entirely if the file exists.
    if let Some(cached) =
        resolve_cached_file(cache_dir, repo_folder, revision, file.filename.as_str())
    {
        return Ok(cached);
    }

    let file_size = metadata_map
        .get(file.filename.as_str())
        .and_then(|m| m.size);
    let start = std::time::Instant::now();

    // Choose download method based on file size and chunked client availability.
    let result = if let (Some(size), Some(client)) = (file_size, chunked_client) {
        if size >= settings.chunk_threshold {
            tracing::debug!(
                filename = %file.filename,
                size_mib = size / 1_048_576,
                connections = settings.connections_per_file,
                "chunked download (multi-connection)"
            );
            let chunked_fut = download_single_file_chunked(
                client,
                file,
                cache_dir,
                repo_folder,
                revision,
                repo_id,
                token,
                metadata_map,
                settings.verify_checksums,
                &settings.retry_policy,
                settings.connections_per_file,
                on_progress,
                files_remaining,
            );
            tokio::time::timeout(settings.timeout_per_file, chunked_fut)
                .await
                .map_err(|_elapsed| FetchError::Timeout {
                    // BORROW: explicit .clone() for owned String
                    filename: file.filename.clone(),
                    seconds: settings.timeout_per_file.as_secs(),
                })?
        } else {
            tracing::debug!(
                filename = %file.filename,
                size_mib = size / 1_048_576,
                "single-connection download (below chunk threshold)"
            );
            download_single_file(
                repo,
                file,
                metadata_map,
                settings.verify_checksums,
                &settings.retry_policy,
                settings.timeout_per_file,
            )
            .await
        }
    } else {
        let reason = if file_size.is_none() {
            "file size unknown (metadata missing)"
        } else {
            "chunked downloads disabled"
        };
        tracing::debug!(
            filename = %file.filename,
            file_size = ?file_size,
            reason = reason,
            "single-connection download"
        );
        download_single_file(
            repo,
            file,
            metadata_map,
            settings.verify_checksums,
            &settings.retry_policy,
            settings.timeout_per_file,
        )
        .await
    };

    // Fall back to direct HTTP GET if hf-hub fails with 416 Range Not Satisfiable.
    // This happens for small git-stored files that don't support Range requests.
    let result = if is_range_not_satisfiable(&result) {
        chunked::download_direct(
            http_client,
            repo_id,
            revision,
            file.filename.as_str(),
            cache_dir,
        )
        .await
    } else {
        result
    };

    log_download_result(file.filename.as_str(), &result, file_size, start.elapsed());
    result
}

/// Collects download task results into a file map and failure list.
///
/// Drains the [`JoinSet`], checking the overall timeout between results.
/// Reports per-file completion progress via the callback.
async fn collect_results(
    join_set: &mut JoinSet<(RepoFile, Result<PathBuf, FetchError>)>,
    timeout_total: Option<Duration>,
    overall_start: tokio::time::Instant,
    on_progress: Option<&ProgressCallback>,
    total: usize,
    completed: &Arc<AtomicUsize>,
) -> Result<(HashMap<String, PathBuf>, Vec<FileFailure>), FetchError> {
    let mut file_map: HashMap<String, PathBuf> = HashMap::with_capacity(total);
    let mut failures: Vec<FileFailure> = Vec::new();

    while let Some(join_result) = join_set.join_next().await {
        // Check overall timeout between result collections.
        if let Some(total_limit) = timeout_total {
            if overall_start.elapsed() >= total_limit {
                join_set.abort_all();
                return Err(FetchError::Timeout {
                    filename: String::from("(overall timeout exceeded)"),
                    seconds: total_limit.as_secs(),
                });
            }
        }

        let (file, download_result) =
            join_result.map_err(|e| FetchError::Http(format!("download task failed: {e}")))?;

        // Increment shared counter so in-flight tasks see updated remaining count.
        let completed_count = completed.fetch_add(1, Ordering::Relaxed) + 1;

        match download_result {
            Ok(path) => {
                // Report progress for completed file.
                let remaining = total.saturating_sub(completed_count);
                let file_size = tokio::fs::metadata(&path)
                    .await
                    .map(|m| m.len())
                    .unwrap_or(0);
                // BORROW: explicit .as_str() instead of Deref coercion
                let event = progress::completed_event(file.filename.as_str(), file_size, remaining);

                if let Some(cb) = on_progress {
                    cb(&event);
                }

                file_map.insert(file.filename, path);
            }
            Err(e) => {
                failures.push(FileFailure {
                    filename: file.filename,
                    reason: e.to_string(),
                    retryable: retry::is_retryable(&e),
                });
            }
        }
    }

    Ok((file_map, failures))
}

/// Checks download results for failures or empty file maps.
///
/// Returns the file map on success, or an appropriate error.
fn validate_download_results(
    file_map: HashMap<String, PathBuf>,
    failures: Vec<FileFailure>,
    repo_id: &str,
) -> Result<HashMap<String, PathBuf>, FetchError> {
    if !failures.is_empty() {
        let path = file_map
            .iter()
            .next()
            .map(|(filename, path)| snapshot_root(filename, path));
        return Err(FetchError::PartialDownload { path, failures });
    }
    if file_map.is_empty() {
        // BORROW: explicit .to_owned() for &str → owned String
        return Err(FetchError::NoFilesMatched {
            repo_id: repo_id.to_owned(),
        });
    }
    Ok(file_map)
}

/// Fetches extended file metadata if needed for checksums or chunked downloads.
///
/// Returns an empty map if neither checksums nor chunked downloads are enabled,
/// or if the metadata fetch fails (with a warning log).
async fn fetch_metadata_if_needed(
    config: Option<&FetchConfig>,
    repo_id: &str,
    verify_checksums: bool,
    chunk_threshold: u64,
    http_client: &reqwest::Client,
) -> HashMap<String, RepoFile> {
    let needs_metadata = verify_checksums || chunk_threshold < u64::MAX;
    if !needs_metadata {
        tracing::debug!("skipping metadata fetch (checksums disabled, chunk_threshold=MAX)");
        return HashMap::new();
    }

    tracing::debug!(
        "fetching extended metadata (checksums={verify_checksums}, chunk_threshold={chunk_threshold} bytes)"
    );
    match fetch_metadata_map(
        repo_id,
        config.and_then(|c| c.token.as_deref()),
        config.and_then(|c| c.revision.as_deref()),
        http_client,
    )
    .await
    {
        Ok(map) => {
            let with_size = map.values().filter(|f| f.size.is_some()).count();
            tracing::debug!(
                files_with_size = with_size,
                total_files = map.len(),
                "metadata fetch succeeded"
            );
            map
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "metadata fetch failed; chunked downloads disabled for this run"
            );
            HashMap::new()
        }
    }
}

/// Logs the result of a file download with timing and throughput.
fn log_download_result(
    filename: &str,
    result: &Result<PathBuf, FetchError>,
    file_size: Option<u64>,
    elapsed: std::time::Duration,
) {
    match result {
        Ok(_) => {
            if let Some(size) = file_size {
                // CAST: u64 → f64, precision loss acceptable; value is a display-only throughput scalar
                #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
                let mbps = (size as f64 * 8.0) / elapsed.as_secs_f64() / 1_000_000.0;
                tracing::debug!(
                    filename = %filename,
                    elapsed_secs = format_args!("{:.1}", elapsed.as_secs_f64()),
                    throughput_mbps = format_args!("{mbps:.1}"),
                    "download complete"
                );
            } else {
                tracing::debug!(
                    filename = %filename,
                    elapsed_secs = format_args!("{:.1}", elapsed.as_secs_f64()),
                    "download complete (size unknown)"
                );
            }
        }
        Err(e) => {
            tracing::debug!(
                filename = %filename,
                error = %e,
                "download failed"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/// Checks available disk space before download.
///
/// Shows download size, available space, and projected remaining space.
/// Warns if space is tight (less than 10% margin) or insufficient.
fn check_disk_space(
    cache_dir: &std::path::Path,
    files: &[RepoFile],
    metadata_map: &HashMap<String, RepoFile>,
    repo_folder: &str,
    revision: &str,
) {
    use fs2::available_space;

    // Sum sizes of files that are NOT already cached.
    let mut download_bytes: u64 = 0;
    for file in files {
        // Skip files already in cache.
        if resolve_cached_file(cache_dir, repo_folder, revision, file.filename.as_str()).is_some() {
            continue;
        }
        // Use metadata size if available, otherwise the file's own size.
        let size = metadata_map
            .get(file.filename.as_str())
            .and_then(|m| m.size)
            .or(file.size)
            .unwrap_or(0);
        download_bytes = download_bytes.saturating_add(size);
    }

    if download_bytes == 0 {
        return;
    }

    let available = match available_space(cache_dir) {
        Ok(a) => a,
        Err(e) => {
            tracing::debug!(error = %e, "could not check available disk space");
            return;
        }
    };

    let after_available = available.saturating_sub(download_bytes);

    // CAST: u64 → f64, precision loss acceptable; display-only size scalars
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let fmt_gib = |v: u64| -> String {
        let gib = v as f64 / (1024.0 * 1024.0 * 1024.0);
        format!("{gib:.2} GiB")
    };

    if available < download_bytes {
        eprintln!(
            "warning: insufficient disk space \u{2014} download needs {}, only {} available",
            fmt_gib(download_bytes),
            fmt_gib(available),
        );
        tracing::warn!(download_bytes, available, "insufficient disk space");
    } else {
        eprintln!(
            "  Disk: {} to fetch, {} available ({} after download)",
            fmt_gib(download_bytes),
            fmt_gib(available),
            fmt_gib(after_available),
        );

        // Warn if less than 10% margin.
        // CAST: u64 → f64, precision loss acceptable; ratio comparison
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let ratio = available as f64 / download_bytes as f64;
        if ratio < 1.1 {
            eprintln!(
                "warning: disk space is tight \u{2014} only {} will remain after download",
                fmt_gib(after_available),
            );
            tracing::warn!(after_available, "disk space is tight after download");
        }
    }
}

/// Attempts to resolve a single file from the local `HuggingFace` cache.
///
/// Looks up: `<cache_dir>/<repo_folder>/snapshots/<commit_hash>/<filename>`.
///
/// Returns `Some(path)` if the file exists locally, `None` otherwise.
fn resolve_cached_file(
    cache_dir: &std::path::Path,
    repo_folder: &str,
    revision: &str,
    filename: &str,
) -> Option<PathBuf> {
    let repo_dir = cache_dir.join(repo_folder);
    let commit_hash = crate::cache::read_ref(&repo_dir, revision)?;
    let cached_path = crate::cache_layout::pointer_path(&repo_dir, &commit_hash, filename);
    if cached_path.exists() {
        tracing::debug!(
            filename = %filename,
            path = %cached_path.display(),
            "file resolved from local cache"
        );
        Some(cached_path)
    } else {
        None
    }
}

/// Attempts to resolve all repository files from the local cache (no network).
///
/// Resolves the cache directory and repo folder from config, then delegates
/// to [`try_resolve_all_from_cache()`].
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the cache directory cannot be resolved.
fn try_resolve_repo_from_cache(
    config: Option<&FetchConfig>,
    repo_id: &str,
) -> Result<Option<HashMap<String, PathBuf>>, FetchError> {
    let cache_dir = config
        .and_then(|c| c.output_dir.clone())
        .map_or_else(crate::cache::hf_cache_dir, Ok)?;
    // BORROW: explicit .as_str() instead of Deref coercion
    let repo_folder = crate::cache_layout::repo_folder_name(repo_id);
    let revision = config.and_then(|c| c.revision.as_deref()).unwrap_or("main");
    let include = config.and_then(|c| c.include.as_ref());
    let exclude = config.and_then(|c| c.exclude.as_ref());

    Ok(try_resolve_all_from_cache(
        &cache_dir,
        repo_folder.as_str(),
        revision,
        include,
        exclude,
    ))
}

/// Attempts to resolve all repository files from the local cache (no network).
///
/// Scans `<cache_dir>/<repo_folder>/snapshots/<commit_hash>/` for files,
/// applies include/exclude filters, and returns a complete `filename → path`
/// map if any files are found. Returns `None` if the snapshot directory
/// does not exist or contains no matching files.
fn try_resolve_all_from_cache(
    cache_dir: &std::path::Path,
    repo_folder: &str,
    revision: &str,
    include: Option<&globset::GlobSet>,
    exclude: Option<&globset::GlobSet>,
) -> Option<HashMap<String, PathBuf>> {
    let repo_dir = cache_dir.join(repo_folder);
    let commit_hash = crate::cache::read_ref(&repo_dir, revision)?;
    let snapshot_dir = crate::cache_layout::snapshot_dir(&repo_dir, &commit_hash);

    if !snapshot_dir.is_dir() {
        return None;
    }

    let mut file_map = HashMap::new();
    collect_cached_files_recursive(
        &snapshot_dir,
        &snapshot_dir,
        include,
        exclude,
        &mut file_map,
    );

    if file_map.is_empty() {
        return None;
    }

    tracing::debug!(
        cached_files = file_map.len(),
        "all files resolved from local cache (no network)"
    );
    Some(file_map)
}

/// Recursively collects files from a snapshot directory into a filename → path map.
fn collect_cached_files_recursive(
    base: &std::path::Path,
    dir: &std::path::Path,
    include: Option<&globset::GlobSet>,
    exclude: Option<&globset::GlobSet>,
    file_map: &mut HashMap<String, PathBuf>,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if path.is_dir() {
            collect_cached_files_recursive(base, &path, include, exclude, file_map);
        } else {
            // Compute relative filename from snapshot root.
            let Ok(relative) = path.strip_prefix(base) else {
                continue;
            };
            // BORROW: explicit .to_string_lossy() for Path → String conversion
            let filename = relative.to_string_lossy().replace('\\', "/");
            // BORROW: explicit .as_str() instead of Deref coercion
            if file_matches(filename.as_str(), include, exclude) {
                file_map.insert(filename, path);
            }
        }
    }
}

/// Derives the snapshot root directory from a `(filename, downloaded_path)` pair.
///
/// `hf-hub` cache layout: `.../snapshots/<sha>/<relative_filename>`
/// For a nested file like `subdir/file.bin`, the downloaded path is
/// `.../snapshots/<sha>/subdir/file.bin`. Stripping the filename's
/// path components from the tail recovers `.../snapshots/<sha>/`.
fn snapshot_root(filename: &str, path: &std::path::Path) -> PathBuf {
    let depth = std::path::Path::new(filename).components().count();
    let mut root = path.to_path_buf();
    for _ in 0..depth {
        if !root.pop() {
            break;
        }
    }
    root
}

/// Returns whether a download result contains an HTTP 416 Range Not Satisfiable error.
///
/// `hf-hub`'s `.get()` internally sends `Range: bytes=0-0` for all files. Small git-stored
/// files (not LFS) may not support Range requests and return 416.
fn is_range_not_satisfiable(result: &Result<PathBuf, FetchError>) -> bool {
    match result {
        Err(e) => {
            let msg = e.to_string();
            msg.contains("416") || msg.contains("Range Not Satisfiable")
        }
        Ok(_) => false,
    }
}

/// Merges plan-recommended settings into `DownloadSettings` for fields the
/// user did not explicitly set.
///
/// This enables implicit plan optimization: every download benefits from
/// plan-based tuning automatically, without requiring `--dry-run`.
#[allow(clippy::too_many_arguments)]
fn merge_plan_recommended(
    settings: &mut DownloadSettings,
    config: Option<&FetchConfig>,
    files: &[RepoFile],
    metadata_map: &HashMap<String, RepoFile>,
    cache_dir: &std::path::Path,
    repo_folder: &str,
    revision: &str,
) {
    let Some(cfg) = config else {
        return;
    };

    // Build a lightweight plan from the already-fetched file list.
    let plan_files: Vec<crate::plan::FilePlan> = files
        .iter()
        .map(|f| {
            let size = metadata_map
                // BORROW: explicit .as_str() instead of Deref coercion
                .get(f.filename.as_str())
                .and_then(|m| m.size)
                .unwrap_or(0);
            let cached = resolve_cached_file(
                cache_dir,
                repo_folder,
                revision,
                // BORROW: explicit .as_str() instead of Deref coercion
                f.filename.as_str(),
            )
            .is_some();
            crate::plan::FilePlan {
                // BORROW: explicit .clone() for owned String field
                filename: f.filename.clone(),
                size,
                cached,
            }
        })
        .collect();

    let total_bytes: u64 = plan_files.iter().map(|f| f.size).sum();
    let cached_bytes: u64 = plan_files.iter().filter(|f| f.cached).map(|f| f.size).sum();

    let plan = crate::plan::DownloadPlan {
        repo_id: String::new(), // Not used by recommended_config_builder.
        revision: String::new(),
        files: plan_files,
        total_bytes,
        cached_bytes,
        download_bytes: total_bytes.saturating_sub(cached_bytes),
    };

    // Only override fields the user did not explicitly set.
    if let Ok(rec) = plan.recommended_config() {
        if !cfg.explicit.concurrency {
            settings.concurrency = rec.concurrency();
        }
        if !cfg.explicit.connections_per_file {
            settings.connections_per_file = rec.connections_per_file();
        }
        if !cfg.explicit.chunk_threshold {
            settings.chunk_threshold = rec.chunk_threshold();
        }

        tracing::debug!(
            concurrency = settings.concurrency,
            connections_per_file = settings.connections_per_file,
            chunk_threshold = settings.chunk_threshold,
            "merged plan-recommended settings"
        );
    }
}

/// Fetches extended metadata and builds a filename → `RepoFile` lookup map.
async fn fetch_metadata_map(
    repo_id: &str,
    token: Option<&str>,
    revision: Option<&str>,
    http_client: &reqwest::Client,
) -> Result<HashMap<String, RepoFile>, FetchError> {
    let files = repo::list_repo_files_with_metadata(repo_id, token, revision, http_client).await?;

    // BORROW: explicit .clone() for owned String HashMap key
    let map = files.into_iter().map(|f| (f.filename.clone(), f)).collect();

    Ok(map)
}
