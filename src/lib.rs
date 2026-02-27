// SPDX-License-Identifier: MIT OR Apache-2.0

//! # hf-fetch-model
//!
//! Fast `HuggingFace` model downloads for Rust.
//!
//! An embeddable library for downloading `HuggingFace` model repositories
//! with maximum throughput. Wraps [`hf_hub`] and adds repo-level orchestration.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! # async fn example() -> Result<(), hf_fetch_model::FetchError> {
//! let path = hf_fetch_model::download("julien-c/dummy-unknown".to_owned()).await?;
//! println!("Model downloaded to: {}", path.display());
//! # Ok(())
//! # }
//! ```
//!
//! ## `HuggingFace` Cache
//!
//! Downloaded files are stored in the standard `HuggingFace` cache directory
//! (`~/.cache/huggingface/hub/`), ensuring compatibility with Python tooling.
//!
//! ## Authentication
//!
//! Set the `HF_TOKEN` environment variable to access private or gated models.

pub mod download;
pub mod error;
pub mod repo;

pub use error::FetchError;

use std::path::PathBuf;

/// Downloads all files from a `HuggingFace` model repository.
///
/// Uses high-throughput mode for maximum download speed. Files are stored
/// in the standard `HuggingFace` cache layout (`~/.cache/huggingface/hub/`).
///
/// Authentication is handled via the `HF_TOKEN` environment variable when set.
///
/// # Arguments
///
/// * `repo_id` — The repository identifier (e.g., `"google/gemma-2-2b-it"`).
///
/// # Returns
///
/// The path to the snapshot directory containing all downloaded files.
///
/// # Errors
///
/// * [`FetchError::Api`] — if the `HuggingFace` API or download fails.
/// * [`FetchError::RepoNotFound`] — if the repository does not exist.
/// * [`FetchError::Auth`] — if authentication is required but fails.
pub async fn download(repo_id: String) -> Result<PathBuf, FetchError> {
    let api = hf_hub::api::tokio::ApiBuilder::new()
        .high()
        .build()
        .map_err(FetchError::Api)?;

    let repo = api.model(repo_id.clone());
    download::download_all_files(&repo, repo_id).await
}
