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
//! let outcome = hf_fetch_model::download("julien-c/dummy-unknown".to_owned()).await?;
//! println!("Model at: {}", outcome.inner().display());
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
//! let outcome = hf_fetch_model::download_with_config(
//!     "google/gemma-2-2b".to_owned(),
//!     &config,
//! ).await?;
//! // outcome.is_cached() tells you if it came from local cache
//! let path = outcome.into_inner();
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

pub mod cache;
pub mod checksum;
mod chunked;
pub mod config;
pub mod discover;
pub mod download;
pub mod error;
pub mod inspect;
pub mod plan;
pub mod progress;
pub mod repo;
mod retry;

pub use config::{compile_glob_patterns, file_matches, FetchConfig, FetchConfigBuilder, Filter};
pub use discover::{GateStatus, ModelCardMetadata, SearchResult};
pub use download::DownloadOutcome;
pub use error::{FetchError, FileFailure};
pub use inspect::AdapterConfig;
pub use plan::{download_plan, DownloadPlan, FilePlan};
pub use progress::ProgressEvent;

use std::collections::HashMap;
use std::path::PathBuf;

use hf_hub::{Repo, RepoType};

/// Downloads all files from a `HuggingFace` model repository.
///
/// Uses high-throughput mode for maximum download speed, including
/// auto-tuned concurrency, chunked multi-connection downloads for large
/// files, and plan-optimized settings based on file size distribution.
/// Files are stored in the standard `HuggingFace` cache layout
/// (`~/.cache/huggingface/hub/`).
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
/// * [`FetchError::Api`] — if the `HuggingFace` API or download fails (includes auth failures).
/// * [`FetchError::RepoNotFound`] — if the repository does not exist.
/// * [`FetchError::InvalidPattern`] — if the default config fails to build (should not happen).
pub async fn download(repo_id: String) -> Result<DownloadOutcome<PathBuf>, FetchError> {
    let config = FetchConfig::builder().build()?;
    download_with_config(repo_id, &config).await
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
/// * [`FetchError::Api`] — if the `HuggingFace` API or download fails (includes auth failures).
/// * [`FetchError::RepoNotFound`] — if the repository does not exist.
pub async fn download_with_config(
    repo_id: String,
    config: &FetchConfig,
) -> Result<DownloadOutcome<PathBuf>, FetchError> {
    let mut builder = hf_hub::api::tokio::ApiBuilder::new().high();

    if let Some(ref token) = config.token {
        // BORROW: explicit .clone() to pass owned String
        builder = builder.with_token(Some(token.clone()));
    }

    if let Some(ref dir) = config.output_dir {
        // BORROW: explicit .clone() for owned PathBuf
        builder = builder.with_cache_dir(dir.clone());
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
    download::download_all_files(repo, repo_id, Some(config)).await
}

/// Blocking version of [`download()`] for non-async callers.
///
/// Creates a Tokio runtime internally. Do not call from within
/// an existing async context (use [`download()`] instead).
///
/// # Errors
///
/// Same as [`download()`].
pub fn download_blocking(repo_id: String) -> Result<DownloadOutcome<PathBuf>, FetchError> {
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
) -> Result<DownloadOutcome<PathBuf>, FetchError> {
    let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
        path: PathBuf::from("<runtime>"),
        source: e,
    })?;
    rt.block_on(download_with_config(repo_id, config))
}

/// Downloads all files from a `HuggingFace` model repository and returns
/// a filename → path map.
///
/// Each key is the relative filename within the repository (e.g.,
/// `"config.json"`, `"model.safetensors"`), and each value is the
/// absolute local path to the downloaded file.
///
/// Uses the same high-throughput defaults as [`download()`]: auto-tuned
/// concurrency and chunked multi-connection downloads for large files.
///
/// For filtering, progress, and other options, use
/// [`download_files_with_config()`].
///
/// # Arguments
///
/// * `repo_id` — The repository identifier (e.g., `"google/gemma-2-2b-it"`).
///
/// # Errors
///
/// * [`FetchError::Api`] — if the `HuggingFace` API or download fails (includes auth failures).
/// * [`FetchError::RepoNotFound`] — if the repository does not exist.
/// * [`FetchError::InvalidPattern`] — if the default config fails to build (should not happen).
pub async fn download_files(
    repo_id: String,
) -> Result<DownloadOutcome<HashMap<String, PathBuf>>, FetchError> {
    let config = FetchConfig::builder().build()?;
    download_files_with_config(repo_id, &config).await
}

/// Downloads files from a `HuggingFace` model repository using the given
/// configuration and returns a filename → path map.
///
/// Each key is the relative filename within the repository (e.g.,
/// `"config.json"`, `"model.safetensors"`), and each value is the
/// absolute local path to the downloaded file.
///
/// # Arguments
///
/// * `repo_id` — The repository identifier (e.g., `"google/gemma-2-2b-it"`).
/// * `config` — Download configuration (see [`FetchConfig::builder()`]).
///
/// # Errors
///
/// * [`FetchError::Api`] — if the `HuggingFace` API or download fails (includes auth failures).
/// * [`FetchError::RepoNotFound`] — if the repository does not exist.
pub async fn download_files_with_config(
    repo_id: String,
    config: &FetchConfig,
) -> Result<DownloadOutcome<HashMap<String, PathBuf>>, FetchError> {
    let mut builder = hf_hub::api::tokio::ApiBuilder::new().high();

    if let Some(ref token) = config.token {
        // BORROW: explicit .clone() to pass owned String
        builder = builder.with_token(Some(token.clone()));
    }

    if let Some(ref dir) = config.output_dir {
        // BORROW: explicit .clone() for owned PathBuf
        builder = builder.with_cache_dir(dir.clone());
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
    download::download_all_files_map(repo, repo_id, Some(config)).await
}

/// Blocking version of [`download_files()`] for non-async callers.
///
/// Creates a Tokio runtime internally. Do not call from within
/// an existing async context (use [`download_files()`] instead).
///
/// # Errors
///
/// Same as [`download_files()`].
pub fn download_files_blocking(
    repo_id: String,
) -> Result<DownloadOutcome<HashMap<String, PathBuf>>, FetchError> {
    let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
        path: PathBuf::from("<runtime>"),
        source: e,
    })?;
    rt.block_on(download_files(repo_id))
}

/// Downloads a single file from a `HuggingFace` model repository.
///
/// Returns the local cache path. If the file is already cached (and
/// checksums match when `verify_checksums` is enabled), the download
/// is skipped and the cached path is returned immediately.
///
/// Files at or above [`FetchConfig`]'s `chunk_threshold` (auto-tuned by
/// the download plan optimizer, or 100 MiB fallback) are downloaded using
/// multiple parallel HTTP Range connections (`connections_per_file`,
/// auto-tuned or 8 fallback). Smaller files use a single connection.
///
/// # Arguments
///
/// * `repo_id` — Repository identifier (e.g., `"mntss/clt-gemma-2-2b-426k"`).
/// * `filename` — Exact filename within the repository (e.g., `"W_enc_5.safetensors"`).
/// * `config` — Shared configuration for auth, progress, checksums, retries, and chunking.
///
/// # Errors
///
/// * [`FetchError::Http`] — if the file does not exist in the repository.
/// * [`FetchError::Api`] — on download failure (after retries).
/// * [`FetchError::Checksum`] — if verification is enabled and fails.
pub async fn download_file(
    repo_id: String,
    filename: &str,
    config: &FetchConfig,
) -> Result<DownloadOutcome<PathBuf>, FetchError> {
    let mut builder = hf_hub::api::tokio::ApiBuilder::new().high();

    if let Some(ref token) = config.token {
        // BORROW: explicit .clone() to pass owned String
        builder = builder.with_token(Some(token.clone()));
    }

    if let Some(ref dir) = config.output_dir {
        // BORROW: explicit .clone() for owned PathBuf
        builder = builder.with_cache_dir(dir.clone());
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
    download::download_file_by_name(repo, repo_id, filename, config).await
}

/// Blocking version of [`download_file()`] for non-async callers.
///
/// Creates a Tokio runtime internally. Do not call from within
/// an existing async context (use [`download_file()`] instead).
///
/// # Errors
///
/// Same as [`download_file()`].
pub fn download_file_blocking(
    repo_id: String,
    filename: &str,
    config: &FetchConfig,
) -> Result<DownloadOutcome<PathBuf>, FetchError> {
    let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
        path: PathBuf::from("<runtime>"),
        source: e,
    })?;
    rt.block_on(download_file(repo_id, filename, config))
}

/// Blocking version of [`download_files_with_config()`] for non-async callers.
///
/// Creates a Tokio runtime internally. Do not call from within
/// an existing async context (use [`download_files_with_config()`] instead).
///
/// # Errors
///
/// Same as [`download_files_with_config()`].
pub fn download_files_with_config_blocking(
    repo_id: String,
    config: &FetchConfig,
) -> Result<DownloadOutcome<HashMap<String, PathBuf>>, FetchError> {
    let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
        path: PathBuf::from("<runtime>"),
        source: e,
    })?;
    rt.block_on(download_files_with_config(repo_id, config))
}

/// Downloads files according to an existing [`DownloadPlan`].
///
/// Only uncached files in the plan are downloaded. The `config` controls
/// authentication, progress, timeouts, and performance settings.
/// Use [`DownloadPlan::recommended_config()`] to compute an optimized config,
/// or override specific fields via [`DownloadPlan::recommended_config_builder()`].
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the cache directory cannot be resolved.
/// Same error conditions as [`download_with_config()`] for the download itself.
pub async fn download_with_plan(
    plan: &DownloadPlan,
    config: &FetchConfig,
) -> Result<DownloadOutcome<PathBuf>, FetchError> {
    if plan.fully_cached() {
        // Resolve snapshot path from cache and return immediately.
        let cache_dir = config
            .output_dir
            .clone()
            .map_or_else(cache::hf_cache_dir, Ok)?;
        let repo_folder = format!("models--{}", plan.repo_id.replace('/', "--"));
        let snapshot_dir = cache_dir
            .join(&repo_folder)
            .join("snapshots")
            .join(&plan.revision);
        return Ok(DownloadOutcome::Cached(snapshot_dir));
    }

    // Delegate to the standard download path which will re-check cache
    // internally. The plan's value is the dry-run preview and the
    // recommended config computed by the caller.
    // BORROW: explicit .clone() for owned String argument
    download_with_config(plan.repo_id.clone(), config).await
}

/// Blocking version of [`download_with_plan()`] for non-async callers.
///
/// Creates a Tokio runtime internally. Do not call from within
/// an existing async context (use [`download_with_plan()`] instead).
///
/// # Errors
///
/// Same as [`download_with_plan()`].
pub fn download_with_plan_blocking(
    plan: &DownloadPlan,
    config: &FetchConfig,
) -> Result<DownloadOutcome<PathBuf>, FetchError> {
    let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
        path: PathBuf::from("<runtime>"),
        source: e,
    })?;
    rt.block_on(download_with_plan(plan, config))
}
