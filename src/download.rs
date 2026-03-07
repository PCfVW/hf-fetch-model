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

/// Downloads all files from a repository and returns the cache directory.
///
/// Each file is downloaded via `hf-hub`'s `.get()` method, which respects
/// the `HuggingFace` cache layout (`~/.cache/huggingface/hub/`).
///
/// - **Concurrency**: downloads up to `concurrency` files in parallel (default 4).
/// - **Resume**: hf-hub skips already-cached files automatically.
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
) -> Result<PathBuf, FetchError> {
    // BORROW: clone before move into download_all_files_map for error context
    let repo_id_for_error = repo_id.clone();
    let file_map = download_all_files_map(repo, repo_id, config).await?;

    // Extract the snapshot directory from any downloaded file path.
    // All files in a repo share the same snapshot directory.
    // hf-hub cache layout: .cache/huggingface/hub/models--org--name/snapshots/<sha>/<relative_path>
    let (filename, path) = file_map
        .into_iter()
        .next()
        .ok_or_else(|| FetchError::NoFilesMatched {
            repo_id: repo_id_for_error,
        })?;

    Ok(snapshot_root(&filename, &path))
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
) -> Result<HashMap<String, PathBuf>, FetchError> {
    let overall_start = tokio::time::Instant::now();

    // Fetch file list with basic metadata from hf-hub.
    tracing::debug!(repo_id = %repo_id, "listing repository files");
    let all_files = repo::list_repo_files(&repo, repo_id.clone()).await?;

    // Apply include/exclude filters.
    let include = config.and_then(|c| c.include.as_ref());
    let exclude = config.and_then(|c| c.exclude.as_ref());

    let files: Vec<_> = all_files
        .into_iter()
        // BORROW: explicit .as_str() instead of Deref coercion
        .filter(|f| file_matches(f.filename.as_str(), include, exclude))
        .collect();

    // Fetch extended metadata (SHA256, sizes) if checksum verification is enabled
    // or chunked downloads need file sizes to determine which files exceed the threshold.
    let verify_checksums = config.is_some_and(|c| c.verify_checksums);
    let chunk_threshold = config.map_or(u64::MAX, |c| c.chunk_threshold);
    let needs_metadata = verify_checksums || chunk_threshold < u64::MAX;
    let metadata_map = if needs_metadata {
        tracing::debug!("fetching extended metadata (checksums={verify_checksums}, chunk_threshold={chunk_threshold} bytes)");
        match fetch_metadata_map(
            // BORROW: explicit .as_str() instead of Deref coercion
            repo_id.as_str(),
            config.and_then(|c| c.token.as_deref()),
            config.and_then(|c| c.revision.as_deref()),
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
    } else {
        tracing::debug!("skipping metadata fetch (checksums disabled, chunk_threshold=MAX)");
        HashMap::new()
    };

    // Build retry policy from config.
    let retry_policy = RetryPolicy {
        max_retries: config.map_or(3, |c| c.max_retries),
        ..RetryPolicy::default()
    };

    let timeout_per_file = config
        .and_then(|c| c.timeout_per_file)
        .unwrap_or(DEFAULT_TIMEOUT_PER_FILE);
    let timeout_total = config.and_then(|c| c.timeout_total);
    let concurrency = config.map_or(4, |c| c.concurrency).max(1);
    let on_progress = config.and_then(|c| c.on_progress.clone());
    let connections_per_file = config.map_or(8, |c| c.connections_per_file);

    // Build reqwest client up front (used by chunked downloads and 416 fallback).
    let token_ref = config.and_then(|c| c.token.as_deref());
    let http_client = Arc::new(chunked::build_client(token_ref)?);
    let chunked_client = if chunk_threshold < u64::MAX {
        Some(Arc::clone(&http_client))
    } else {
        None
    };

    // Resolve cache directory (used by chunked downloads and 416 fallback).
    let cache_dir = Arc::new(
        config
            .and_then(|c| c.output_dir.clone())
            .map_or_else(crate::cache::hf_cache_dir, Ok)?,
    );
    // BORROW: explicit .as_str() instead of Deref coercion
    let chunked_repo_folder = Arc::new(chunked::repo_folder_name(repo_id.as_str()));
    let chunked_revision = Arc::new(
        config
            .and_then(|c| c.revision.clone())
            .unwrap_or_else(|| String::from("main")),
    );
    let chunked_token = Arc::new(config.and_then(|c| c.token.clone()));

    let total = files.len();
    let chunked_enabled = chunked_client.is_some();
    tracing::debug!(
        total_files = total,
        concurrency = concurrency,
        connections_per_file = connections_per_file,
        chunk_threshold_mib = chunk_threshold / 1_048_576,
        chunked_enabled = chunked_enabled,
        verify_checksums = verify_checksums,
        "download plan"
    );

    // Wrap shared state in Arc for concurrent task access.
    let repo = Arc::new(repo);
    let metadata_map = Arc::new(metadata_map);
    let semaphore = Arc::new(Semaphore::new(concurrency));
    // Shared counter for completed files, used by streaming progress events.
    let completed = Arc::new(AtomicUsize::new(0));
    let mut join_set = JoinSet::new();

    // Spawn download tasks, limited to `concurrency` in-flight at a time.
    // The semaphore blocks the spawn loop until a permit is available,
    // ensuring at most `concurrency` downloads run simultaneously.
    for file in files {
        // Check overall timeout before queuing next download.
        if let Some(total_limit) = timeout_total {
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
        let task_policy = retry_policy.clone();
        let task_chunked_client = chunked_client.clone();
        let task_cache_dir = cache_dir.clone();
        let task_repo_folder = Arc::clone(&chunked_repo_folder);
        let task_revision = Arc::clone(&chunked_revision);
        let task_on_progress = on_progress.clone();
        // BORROW: explicit .clone() for repo_id
        let task_repo_id = repo_id.clone();
        let task_token = Arc::clone(&chunked_token);
        let task_http_client = Arc::clone(&http_client);
        let task_completed = Arc::clone(&completed);

        join_set.spawn(async move {
            // Determine file size from metadata for chunked download decision.
            let file_size = task_meta.get(file.filename.as_str()).and_then(|m| m.size);
            let file_start = tokio::time::Instant::now();

            let result = if let (Some(size), Some(ref client)) = (file_size, &task_chunked_client) {
                if size >= chunk_threshold {
                    tracing::debug!(
                        filename = %file.filename,
                        size_mib = size / 1_048_576,
                        connections = connections_per_file,
                        "chunked download (multi-connection)"
                    );
                    download_single_file_chunked(
                        client,
                        &file,
                        &task_cache_dir,
                        &task_repo_folder,
                        &task_revision,
                        task_repo_id.as_str(),
                        (*task_token).clone(),
                        &task_meta,
                        verify_checksums,
                        &task_policy,
                        connections_per_file,
                        task_on_progress,
                        total.saturating_sub(task_completed.load(Ordering::Relaxed) + 1),
                    )
                    .await
                } else {
                    tracing::debug!(
                        filename = %file.filename,
                        size_mib = size / 1_048_576,
                        "single-connection download (below chunk threshold)"
                    );
                    download_single_file(
                        &task_repo,
                        &file,
                        &task_meta,
                        verify_checksums,
                        &task_policy,
                        timeout_per_file,
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
                    &task_repo,
                    &file,
                    &task_meta,
                    verify_checksums,
                    &task_policy,
                    timeout_per_file,
                )
                .await
            };

            // Fall back to direct HTTP GET if hf-hub fails with 416 Range Not Satisfiable.
            // This happens for small git-stored files that don't support Range requests.
            let result = if is_range_not_satisfiable(&result) {
                chunked::download_direct(
                    &task_http_client,
                    task_repo_id.as_str(),
                    &task_revision,
                    file.filename.as_str(),
                    &task_cache_dir,
                )
                .await
            } else {
                result
            };

            let elapsed = file_start.elapsed();
            match &result {
                Ok(_) => {
                    if let Some(size) = file_size {
                        // CAST: u64 → f64, precision loss acceptable; value is a display-only throughput scalar
                        #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
                        let mbps = (size as f64 * 8.0) / elapsed.as_secs_f64() / 1_000_000.0;
                        tracing::debug!(
                            filename = %file.filename,
                            elapsed_secs = format_args!("{:.1}", elapsed.as_secs_f64()),
                            throughput_mbps = format_args!("{mbps:.1}"),
                            "download complete"
                        );
                    } else {
                        tracing::debug!(
                            filename = %file.filename,
                            elapsed_secs = format_args!("{:.1}", elapsed.as_secs_f64()),
                            "download complete (size unknown)"
                        );
                    }
                }
                Err(e) => {
                    tracing::debug!(
                        filename = %file.filename,
                        error = %e,
                        "download failed"
                    );
                }
            }

            drop(permit);
            (file, result)
        });
    }

    // Collect results as tasks complete.
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

                if let Some(ref cb) = on_progress {
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

    // If some files failed, report structured errors.
    if !failures.is_empty() {
        let path = file_map
            .iter()
            .next()
            .map(|(filename, path)| snapshot_root(filename, path));
        return Err(FetchError::PartialDownload { path, failures });
    }

    if file_map.is_empty() {
        return Err(FetchError::NoFilesMatched {
            repo_id: repo_id.clone(),
        });
    }

    let total_elapsed = overall_start.elapsed();
    tracing::debug!(
        files_downloaded = file_map.len(),
        files_failed = failures.len(),
        total_elapsed_secs = format_args!("{:.1}", total_elapsed.as_secs_f64()),
        "download complete"
    );

    Ok(file_map)
}

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
    let range_info = chunked::probe_range_support(client.clone(), url, token).await?;

    let Some(range_info) = range_info else {
        // Range not supported — this shouldn't happen for LFS files, but fall back
        // gracefully. Return an error that will be caught and retried via the standard path.
        return Err(FetchError::ChunkedDownload {
            // BORROW: explicit .clone() for owned String
            filename: file.filename.clone(),
            reason: String::from("server does not support Range requests"),
        });
    };

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
/// but applied to exactly one file.
///
/// # Errors
///
/// * [`FetchError::Http`] — if the file does not exist in the repository.
/// * [`FetchError::Api`] — on download failure (after retries).
/// * [`FetchError::Checksum`] — if verification is enabled and fails.
pub(crate) async fn download_file_by_name(
    repo: ApiRepo,
    repo_id: String,
    filename: &str,
    config: &FetchConfig,
) -> Result<PathBuf, FetchError> {
    // Fetch extended metadata (SHA256, sizes) if checksum verification is enabled
    // or chunked downloads need file sizes to determine whether to chunk.
    let verify_checksums = config.verify_checksums;
    let chunk_threshold = config.chunk_threshold;
    let needs_metadata = verify_checksums || chunk_threshold < u64::MAX;
    let metadata_map = if needs_metadata {
        fetch_metadata_map(
            // BORROW: explicit .as_str()/.as_deref() for owned → borrowed conversions
            repo_id.as_str(),
            config.token.as_deref(),
            config.revision.as_deref(),
        )
        .await
        .unwrap_or_else(|e| {
            tracing::warn!(
                filename = %filename,
                error = %e,
                "metadata fetch failed; file size unknown, chunked download disabled"
            );
            HashMap::new()
        })
    } else {
        HashMap::new()
    };

    // Build a RepoFile for this filename from metadata (or with no metadata).
    let file_meta = metadata_map.get(filename);
    // BORROW: explicit .to_owned()/.clone() for owned String fields
    let file = RepoFile {
        filename: filename.to_owned(),
        size: file_meta.and_then(|m| m.size),
        sha256: file_meta.and_then(|m| m.sha256.clone()),
    };

    // Build retry policy from config.
    let retry_policy = RetryPolicy {
        max_retries: config.max_retries,
        ..RetryPolicy::default()
    };

    let timeout_per_file = config.timeout_per_file.unwrap_or(DEFAULT_TIMEOUT_PER_FILE);
    // BORROW: explicit .clone() for Arc-wrapped callback
    let on_progress = config.on_progress.clone();
    let connections_per_file = config.connections_per_file;

    // Build reqwest client (used by chunked downloads and 416 fallback).
    // BORROW: explicit .as_deref() for Option<String> → Option<&str>
    let http_client = chunked::build_client(config.token.as_deref())?;

    // Resolve cache directory.
    // BORROW: explicit .clone() for owned PathBuf
    let cache_dir = config
        .output_dir
        .clone()
        .map_or_else(crate::cache::hf_cache_dir, Ok)?;
    // BORROW: explicit .as_str() instead of Deref coercion
    let repo_folder = chunked::repo_folder_name(repo_id.as_str());
    // BORROW: explicit .clone() for owned String
    let revision = config
        .revision
        .clone()
        .unwrap_or_else(|| String::from("main"));

    // Determine file size from metadata for chunked download decision.
    let file_size = file.size;
    let start = std::time::Instant::now();

    let result = if let Some(size) = file_size {
        if size >= chunk_threshold {
            tracing::debug!(
                filename = %filename,
                size_mib = size / 1_048_576,
                connections = connections_per_file,
                "chunked download (multi-connection)"
            );
            download_single_file_chunked(
                &http_client,
                &file,
                &cache_dir,
                // BORROW: explicit .as_str() for String → &str conversions
                repo_folder.as_str(),
                revision.as_str(),
                repo_id.as_str(),
                // BORROW: explicit .clone() for owned Option<String> and Arc
                config.token.clone(),
                &metadata_map,
                verify_checksums,
                &retry_policy,
                connections_per_file,
                on_progress.clone(),
                0, // files_remaining: only one file
            )
            .await
        } else {
            tracing::debug!(
                filename = %filename,
                size_mib = size / 1_048_576,
                "single-connection download (below chunk threshold)"
            );
            download_single_file(
                &repo,
                &file,
                &metadata_map,
                verify_checksums,
                &retry_policy,
                timeout_per_file,
            )
            .await
        }
    } else {
        tracing::debug!(
            filename = %filename,
            "single-connection download (file size unknown)"
        );
        download_single_file(
            &repo,
            &file,
            &metadata_map,
            verify_checksums,
            &retry_policy,
            timeout_per_file,
        )
        .await
    };

    // Fall back to direct HTTP GET if hf-hub fails with 416 Range Not Satisfiable.
    let result = if is_range_not_satisfiable(&result) {
        chunked::download_direct(
            &http_client,
            // BORROW: explicit .as_str() for String → &str conversions
            repo_id.as_str(),
            revision.as_str(),
            filename,
            &cache_dir,
        )
        .await
    } else {
        result
    };

    let path = result?;

    // Log completion with elapsed time and throughput.
    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    if let Some(size) = file_size {
        // CAST: u64 → f64, precision loss acceptable; value is a display-only throughput scalar
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let mbps = if elapsed_secs > 0.0 {
            (size as f64 * 8.0) / elapsed_secs / 1_000_000.0
        } else {
            0.0
        };
        tracing::debug!(
            filename = %filename,
            elapsed_secs = format_args!("{elapsed_secs:.1}"),
            throughput_mbps = format_args!("{mbps:.1}"),
            "download complete"
        );
    } else {
        tracing::debug!(
            filename = %filename,
            elapsed_secs = format_args!("{elapsed_secs:.1}"),
            "download complete"
        );
    }

    // Report progress for the completed file.
    if let Some(ref cb) = on_progress {
        let file_size = tokio::fs::metadata(&path)
            .await
            .map(|m| m.len())
            .unwrap_or(0);
        let event = progress::completed_event(filename, file_size, 0);
        cb(&event);
    }

    Ok(path)
}

/// Derives the snapshot root directory from a `(filename, downloaded_path)` pair.
///
/// hf-hub cache layout: `.../snapshots/<sha>/<relative_filename>`
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
/// hf-hub's `.get()` internally sends `Range: bytes=0-0` for all files. Small git-stored
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

/// Fetches extended metadata and builds a filename → `RepoFile` lookup map.
async fn fetch_metadata_map(
    repo_id: &str,
    token: Option<&str>,
    revision: Option<&str>,
) -> Result<HashMap<String, RepoFile>, FetchError> {
    let files = repo::list_repo_files_with_metadata(repo_id, token, revision).await?;

    let map = files.into_iter().map(|f| (f.filename.clone(), f)).collect();

    Ok(map)
}
