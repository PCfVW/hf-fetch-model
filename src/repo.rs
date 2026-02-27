// SPDX-License-Identifier: MIT OR Apache-2.0

//! Repository file listing via the `HuggingFace` API.
//!
//! This module provides functions to list all files in a `HuggingFace` model
//! repository, using the `hf-hub` crate's `info()` API.

use hf_hub::api::tokio::ApiRepo;

use crate::error::FetchError;

/// A file entry in a `HuggingFace` repository.
#[derive(Debug, Clone)]
pub struct RepoFile {
    /// The relative path of the file within the repository.
    pub filename: String,
}

/// Lists all files in the given repository.
///
/// # Errors
///
/// Returns [`FetchError::Api`] if the `HuggingFace` API request fails.
/// Returns [`FetchError::RepoNotFound`] if the repository does not exist.
pub async fn list_repo_files(repo: &ApiRepo, repo_id: String) -> Result<Vec<RepoFile>, FetchError> {
    let info = repo.info().await.map_err(|e| {
        // BORROW: explicit .to_string() for error message inspection
        let msg = e.to_string();
        if msg.contains("404") {
            FetchError::RepoNotFound { repo_id }
        } else {
            FetchError::Api(e)
        }
    })?;

    let files = info
        .siblings
        .into_iter()
        .map(|s| RepoFile {
            filename: s.rfilename,
        })
        .collect();

    Ok(files)
}
