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
use crate::config::{file_matches, FetchConfig};
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

    // Fetch extended metadata (SHA256, sizes) if checksum verification is enabled.
    let verify_checksums = config.is_some_and(|c| c.verify_checksums);
    let metadata_map = if verify_checksums {
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

        join_set.spawn(async move {
            let result = download_single_file(
                &task_repo,
                &file,
                &task_meta,
                verify_checksums,
                &task_policy,
                timeout_per_file,
            )
            .await;
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
