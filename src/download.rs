// SPDX-License-Identifier: MIT OR Apache-2.0

//! Download orchestration for `HuggingFace` model repositories.
//!
//! This module coordinates the download of all files in a model
//! repository using `hf-hub`'s high-throughput mode, with optional
//! filtering and progress reporting.

use std::path::PathBuf;

use hf_hub::api::tokio::ApiRepo;

use crate::config::{file_matches, FetchConfig};
use crate::error::FetchError;
use crate::progress;
use crate::repo;

/// Downloads all files from a repository and returns the cache directory.
///
/// Each file is downloaded via `hf-hub`'s `.get()` method, which respects
/// the `HuggingFace` cache layout (`~/.cache/huggingface/hub/`).
///
/// # Errors
///
/// Returns [`FetchError::Api`] if any file download fails.
/// Returns [`FetchError::RepoNotFound`] if the repository does not exist.
pub async fn download_all_files(
    repo: &ApiRepo,
    repo_id: String,
    config: Option<&FetchConfig>,
) -> Result<PathBuf, FetchError> {
    let all_files = repo::list_repo_files(repo, repo_id).await?;

    // Apply include/exclude filters.
    let include = config.and_then(|c| c.include.as_ref());
    let exclude = config.and_then(|c| c.exclude.as_ref());

    let files: Vec<_> = all_files
        .into_iter()
        .filter(|f| file_matches(f.filename.as_str(), include, exclude))
        .collect();

    let total = files.len();
    let mut last_path: Option<PathBuf> = None;

    for (i, file) in files.iter().enumerate() {
        // BORROW: explicit .as_str() instead of Deref coercion
        let path = repo.get(file.filename.as_str()).await?;

        // Report progress for completed file.
        let remaining = total.saturating_sub(i + 1);
        let file_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let event = progress::completed_event(file.filename.as_str(), file_size, remaining);

        if let Some(cfg) = config {
            if let Some(ref cb) = cfg.on_progress {
                cb(&event);
            }
        }

        last_path = Some(path);
    }

    // The cache directory is the parent of any downloaded file.
    // All files in a repo share the same snapshot directory.
    // We navigate up from any file to find the snapshot root.
    let file_path = last_path.ok_or_else(|| FetchError::RepoNotFound {
        repo_id: String::from("(empty repository or all files filtered out)"),
    })?;

    // hf-hub cache layout: .cache/huggingface/hub/models--org--name/snapshots/<sha>/file
    // We want to return the snapshot directory (parent of the file).
    let cache_dir = file_path
        .parent()
        .map_or_else(|| file_path.clone(), std::path::Path::to_path_buf);

    Ok(cache_dir)
}
