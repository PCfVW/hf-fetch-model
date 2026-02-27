// SPDX-License-Identifier: MIT OR Apache-2.0

//! Download orchestration for `HuggingFace` model repositories.
//!
//! This module coordinates the parallel download of all files in a model
//! repository using `hf-hub`'s high-throughput mode.

use std::path::PathBuf;

use hf_hub::api::tokio::ApiRepo;

use crate::error::FetchError;
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
pub async fn download_all_files(repo: &ApiRepo, repo_id: String) -> Result<PathBuf, FetchError> {
    let files = repo::list_repo_files(repo, repo_id).await?;

    let mut last_path: Option<PathBuf> = None;

    for file in &files {
        // BORROW: explicit .as_str() instead of Deref coercion
        let path = repo.get(file.filename.as_str()).await?;
        last_path = Some(path);
    }

    // The cache directory is the parent of any downloaded file.
    // All files in a repo share the same snapshot directory.
    // We navigate up from any file to find the snapshot root.
    let file_path = last_path.ok_or_else(|| FetchError::RepoNotFound {
        repo_id: String::from("(empty repository)"),
    })?;

    // hf-hub cache layout: .cache/huggingface/hub/models--org--name/snapshots/<sha>/file
    // We want to return the snapshot directory (parent of the file).
    let cache_dir = file_path
        .parent()
        .map_or_else(|| file_path.clone(), std::path::Path::to_path_buf);

    Ok(cache_dir)
}
