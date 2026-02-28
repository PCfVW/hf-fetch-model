// SPDX-License-Identifier: MIT OR Apache-2.0

//! Repository file listing via the `HuggingFace` API.
//!
//! This module provides functions to list all files in a `HuggingFace` model
//! repository, using the `hf-hub` crate's `info()` API and optionally
//! fetching extended metadata (sizes and SHA256 hashes) via a direct HTTP call.

use hf_hub::api::tokio::ApiRepo;
use serde::Deserialize;

use crate::error::FetchError;

/// A file entry in a `HuggingFace` repository.
#[derive(Debug, Clone)]
pub struct RepoFile {
    /// The relative path of the file within the repository.
    pub filename: String,
    /// File size in bytes (if known from API metadata).
    pub size: Option<u64>,
    /// SHA256 hex digest (if the file is stored in LFS).
    pub sha256: Option<String>,
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
            size: None,
            sha256: None,
        })
        .collect();

    Ok(files)
}

// --- Direct HF API metadata (for SHA256 and file sizes) ---

/// Raw JSON sibling entry from the `HuggingFace` API.
#[derive(Debug, Deserialize)]
struct ApiSibling {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    lfs: Option<ApiLfs>,
}

/// LFS metadata attached to a sibling entry.
#[derive(Debug, Deserialize)]
struct ApiLfs {
    sha256: String,
    size: u64,
}

/// Raw JSON response from `GET /api/models/{repo_id}`.
#[derive(Debug, Deserialize)]
struct ApiModelInfo {
    siblings: Vec<ApiSibling>,
}

/// Fetches extended file metadata (sizes and SHA256 hashes) via the `HuggingFace` REST API.
///
/// This makes a direct HTTP call to `https://huggingface.co/api/models/{repo_id}`
/// to retrieve LFS metadata that `hf-hub`'s `info()` does not expose.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the HTTP request fails.
/// Returns [`FetchError::RepoNotFound`] if the repository does not exist.
pub async fn list_repo_files_with_metadata(
    repo_id: &str,
    token: Option<&str>,
    revision: Option<&str>,
) -> Result<Vec<RepoFile>, FetchError> {
    let mut url = format!("https://huggingface.co/api/models/{repo_id}");
    if let Some(rev) = revision {
        url = format!("{url}?revision={rev}");
    }

    let client = reqwest::Client::new();
    // BORROW: explicit .as_str() instead of Deref coercion
    let mut request = client.get(url.as_str());
    if let Some(t) = token {
        request = request.bearer_auth(t);
    }

    let response = request
        .send()
        .await
        .map_err(|e| FetchError::Http(e.to_string()))?;

    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(FetchError::RepoNotFound {
            repo_id: repo_id.to_owned(),
        });
    }

    if !response.status().is_success() {
        return Err(FetchError::Http(format!(
            "HF API returned status {}",
            response.status()
        )));
    }

    let info: ApiModelInfo = response
        .json()
        .await
        .map_err(|e| FetchError::Http(e.to_string()))?;

    let files = info
        .siblings
        .into_iter()
        .map(|s| {
            let (size, sha256) = match s.lfs {
                Some(lfs) => (Some(lfs.size), Some(lfs.sha256)),
                None => (s.size, None),
            };
            RepoFile {
                filename: s.rfilename,
                size,
                sha256,
            }
        })
        .collect();

    Ok(files)
}
