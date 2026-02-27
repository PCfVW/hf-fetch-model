// SPDX-License-Identifier: MIT OR Apache-2.0

//! Error types for hf-fetch-model.
//!
//! All fallible operations in this crate return [`FetchError`].
//! [`FileFailure`] provides structured per-file error reporting.

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

    /// SHA256 checksum mismatch after download.
    #[error("checksum mismatch for {filename}: expected {expected}, got {actual}")]
    Checksum {
        /// The filename that failed verification.
        filename: String,
        /// The expected SHA256 hex digest.
        expected: String,
        /// The actual SHA256 hex digest computed from the file.
        actual: String,
    },

    /// A download operation timed out.
    #[error("timeout downloading {filename} after {seconds}s")]
    Timeout {
        /// The filename that timed out.
        filename: String,
        /// The timeout duration in seconds.
        seconds: u64,
    },

    /// One or more files failed to download.
    ///
    /// Contains the successful path and a list of per-file failures.
    #[error("{} file(s) failed to download", failures.len())]
    PartialDownload {
        /// The snapshot directory (if any files succeeded).
        path: Option<PathBuf>,
        /// Per-file failure details.
        failures: Vec<FileFailure>,
    },

    /// An HTTP request to the `HuggingFace` API failed.
    #[error("HTTP error: {0}")]
    Http(String),
}

/// A per-file download failure with structured context.
#[derive(Debug, Clone)]
pub struct FileFailure {
    /// The filename that failed.
    pub filename: String,
    /// Human-readable description of the failure.
    pub reason: String,
    /// Whether this failure is likely to succeed on retry.
    pub retryable: bool,
}

impl std::fmt::Display for FileFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {} (retryable: {})",
            self.filename, self.reason, self.retryable
        )
    }
}
