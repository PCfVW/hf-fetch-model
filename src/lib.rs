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
//! ## Configured Download
//!
//! ```rust,no_run
//! # async fn example() -> Result<(), hf_fetch_model::FetchError> {
//! use hf_fetch_model::FetchConfig;
//!
//! let config = FetchConfig::builder()
//!     .filter("*.safetensors")
//!     .filter("*.json")
//!     .on_progress(|e| {
//!         println!("{}: {:.1}%", e.filename, e.percent);
//!     })
//!     .build()?;
//!
//! let path = hf_fetch_model::download_with_config(
//!     "google/gemma-2-2b".to_owned(),
//!     &config,
//! ).await?;
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
//! Set the `HF_TOKEN` environment variable to access private or gated models,
//! or use [`FetchConfig::builder().token()`](FetchConfigBuilder::token).

pub mod checksum;
pub mod config;
pub mod download;
pub mod error;
pub mod progress;
pub mod repo;
mod retry;

pub use config::{FetchConfig, FetchConfigBuilder, Filter};
pub use error::{FetchError, FileFailure};
pub use progress::ProgressEvent;

use std::path::PathBuf;

use hf_hub::{Repo, RepoType};

/// Downloads all files from a `HuggingFace` model repository.
///
/// Uses high-throughput mode for maximum download speed. Files are stored
/// in the standard `HuggingFace` cache layout (`~/.cache/huggingface/hub/`).
///
/// Authentication is handled via the `HF_TOKEN` environment variable when set.
///
/// For filtering, progress, and other options, use [`download_with_config()`].
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
    download::download_all_files(&repo, repo_id, None).await
}

/// Downloads files from a `HuggingFace` model repository using the given configuration.
///
/// Supports filtering, progress reporting, custom revision, authentication,
/// and concurrency settings via [`FetchConfig`].
///
/// # Arguments
///
/// * `repo_id` — The repository identifier (e.g., `"google/gemma-2-2b-it"`).
/// * `config` — Download configuration (see [`FetchConfig::builder()`]).
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
pub async fn download_with_config(
    repo_id: String,
    config: &FetchConfig,
) -> Result<PathBuf, FetchError> {
    let mut builder = hf_hub::api::tokio::ApiBuilder::new().high();

    if let Some(ref token) = config.token {
        // BORROW: explicit .clone() to pass owned String
        builder = builder.with_token(Some(token.clone()));
    }

    let api = builder.build().map_err(FetchError::Api)?;

    let hf_repo = match config.revision {
        Some(ref rev) => {
            // BORROW: explicit .clone() for owned String arguments
            Repo::with_revision(repo_id.clone(), RepoType::Model, rev.clone())
        }
        None => Repo::new(repo_id.clone(), RepoType::Model),
    };

    let repo = api.repo(hf_repo);
    download::download_all_files(&repo, repo_id, Some(config)).await
}

/// Blocking version of [`download()`] for non-async callers.
///
/// Creates a Tokio runtime internally. Do not call from within
/// an existing async context (use [`download()`] instead).
///
/// # Errors
///
/// Same as [`download()`].
pub fn download_blocking(repo_id: String) -> Result<PathBuf, FetchError> {
    let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
        path: PathBuf::from("<runtime>"),
        source: e,
    })?;
    rt.block_on(download(repo_id))
}

/// Blocking version of [`download_with_config()`] for non-async callers.
///
/// Creates a Tokio runtime internally. Do not call from within
/// an existing async context (use [`download_with_config()`] instead).
///
/// # Errors
///
/// Same as [`download_with_config()`].
pub fn download_with_config_blocking(
    repo_id: String,
    config: &FetchConfig,
) -> Result<PathBuf, FetchError> {
    let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
        path: PathBuf::from("<runtime>"),
        source: e,
    })?;
    rt.block_on(download_with_config(repo_id, config))
}
