// SPDX-License-Identifier: MIT OR Apache-2.0

//! Error types for hf-fetch-model.
//!
//! All fallible operations in this crate return [`FetchError`].

use std::path::PathBuf;

/// Errors that can occur during model fetching.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FetchError {
    /// The hf-hub API returned an error.
    #[error("hf-hub API error: {0}")]
    Api(#[from] hf_hub::api::tokio::ApiError),

    /// An I/O error occurred while accessing the local filesystem.
    #[error("I/O error at {path}: {source}")]
    Io {
        /// The path that caused the error.
        path: PathBuf,
        /// The underlying I/O error.
        source: std::io::Error,
    },

    /// The repository was not found or is inaccessible.
    #[error("repository not found: {repo_id}")]
    RepoNotFound {
        /// The repository identifier that was not found.
        repo_id: String,
    },

    /// Authentication failed (missing or invalid token).
    #[error("authentication failed: {reason}")]
    Auth {
        /// Description of the authentication failure.
        reason: String,
    },

    /// An invalid glob pattern was provided for filtering.
    #[error("invalid glob pattern: {pattern}: {reason}")]
    InvalidPattern {
        /// The glob pattern that failed to parse.
        pattern: String,
        /// Description of the parse error.
        reason: String,
    },
}
