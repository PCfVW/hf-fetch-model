// SPDX-License-Identifier: MIT OR Apache-2.0

//! Download orchestration for `HuggingFace` model repositories.
//!
//! This module coordinates the download of all files in a model
//! repository using `hf-hub`'s high-throughput mode, with concurrent
//! file downloads, filtering, progress reporting, retry, checksum
//! verification, and timeouts.

use std::collections::HashMap;
use std::path::PathBuf;
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
/// Returns [`FetchError::Timeout`] if the overall timeout is exceeded.
pub async fn download_all_files(
    repo: ApiRepo,
    repo_id: String,
    config: Option<&FetchConfig>,
) -> Result<PathBuf, FetchError> {
    let file_map = download_all_files_map(repo, repo_id, config).await?;

    // Extract the snapshot directory from any downloaded file path.
    // All files in a repo share the same snapshot directory.
    // hf-hub cache layout: .cache/huggingface/hub/models--org--name/snapshots/<sha>/file
    let any_path = file_map
        .into_values()
        .next()
        .ok_or_else(|| FetchError::RepoNotFound {
            repo_id: String::from("(empty repository or all files filtered out)"),
        })?;

    let cache_dir = any_path
        .parent()
        .map_or_else(|| any_path.clone(), std::path::Path::to_path_buf);

    Ok(cache_dir)
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
/// Returns [`FetchError::Timeout`] if the overall timeout is exceeded.
pub async fn download_all_files_map(
    repo: ApiRepo,
    repo_id: String,
    config: Option<&FetchConfig>,
) -> Result<HashMap<String, PathBuf>, FetchError> {
    let overall_start = tokio::time::Instant::now();

    // Fetch file list with basic metadata from hf-hub.
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
        fetch_metadata_map(
            // BORROW: explicit .as_str() instead of Deref coercion
            repo_id.as_str(),
            config.and_then(|c| c.token.as_deref()),
            config.and_then(|c| c.revision.as_deref()),
        )
        .await
        .unwrap_or_default()
    } else {
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

    // Wrap shared state in Arc for concurrent task access.
    let repo = Arc::new(repo);
    let metadata_map = Arc::new(metadata_map);
    let semaphore = Arc::new(Semaphore::new(concurrency));
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

        join_set.spawn(async move {
            // Determine file size from metadata for chunked download decision.
            let file_size = task_meta.get(file.filename.as_str()).and_then(|m| m.size);

            let result = if let (Some(size), Some(ref client)) = (file_size, &task_chunked_client) {
                if size >= chunk_threshold {
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
                        total.saturating_sub(1),
                    )
                    .await
                } else {
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

            drop(permit);
            (file, result)
        });
    }

    // Collect results as tasks complete.
    let mut file_map: HashMap<String, PathBuf> = HashMap::with_capacity(total);
    let mut failures: Vec<FileFailure> = Vec::new();
    let mut completed_count: usize = 0;

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

        completed_count += 1;

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
        let path = file_map.values().next().cloned();
        return Err(FetchError::PartialDownload { path, failures });
    }

    if file_map.is_empty() {
        return Err(FetchError::RepoNotFound {
            repo_id: String::from("(empty repository or all files filtered out)"),
        });
    }

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
/// Returns [`FetchError::Http`] if the file does not exist in the repository.
/// Returns [`FetchError::Api`] on download failure (after retries).
/// Returns [`FetchError::Checksum`] if verification is enabled and fails.
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
            repo_id.as_str(),
            config.token.as_deref(),
            config.revision.as_deref(),
        )
        .await
        .unwrap_or_default()
    } else {
        HashMap::new()
    };

    // Build a RepoFile for this filename from metadata (or with no metadata).
    let file_meta = metadata_map.get(filename);
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
    let on_progress = config.on_progress.clone();
    let connections_per_file = config.connections_per_file;

    // Build reqwest client (used by chunked downloads and 416 fallback).
    let http_client = chunked::build_client(config.token.as_deref())?;

    // Resolve cache directory.
    let cache_dir = config
        .output_dir
        .clone()
        .map_or_else(crate::cache::hf_cache_dir, Ok)?;
    // BORROW: explicit .as_str() instead of Deref coercion
    let repo_folder = chunked::repo_folder_name(repo_id.as_str());
    let revision = config
        .revision
        .clone()
        .unwrap_or_else(|| String::from("main"));

    // Determine file size from metadata for chunked download decision.
    let file_size = file.size;

    let result = if let Some(size) = file_size {
        if size >= chunk_threshold {
            download_single_file_chunked(
                &http_client,
                &file,
                &cache_dir,
                repo_folder.as_str(),
                revision.as_str(),
                repo_id.as_str(),
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
