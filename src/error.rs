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
    /// The `hf-hub` API returned an error.
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

    /// Authentication failed: a gated repository was requested without a
    /// token, or the supplied token was rejected (HTTP 401/403).
    ///
    /// Returned by the gated-model pre-flight in `download` /
    /// `download_with_config` before any transfer starts; non-retryable. The
    /// library `inspect` functions instead surface the raw HTTP status as
    /// [`FetchError::Http`] (the `hf-fm` CLI upgrades those into this same
    /// diagnosis). See the crate-level *Authentication* section.
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
    #[error("{} file(s) failed to download:{}", failures.len(), format_failures(failures))]
    PartialDownload {
        /// The snapshot directory (if any files succeeded).
        path: Option<PathBuf>,
        /// Per-file failure details.
        failures: Vec<FileFailure>,
    },

    /// A chunked (multi-connection) download failed.
    #[error("chunked download failed for {filename}: {reason}")]
    ChunkedDownload {
        /// The filename that failed.
        filename: String,
        /// Description of the failure.
        reason: String,
    },

    /// An HTTP request to the `HuggingFace` API failed.
    #[error("HTTP error: {0}")]
    Http(String),

    /// An invalid argument was provided.
    #[error("{0}")]
    InvalidArgument(String),

    /// The repository exists but no files matched after filtering,
    /// or the repository contains no files at all.
    #[error("no files matched in repository {repo_id}")]
    NoFilesMatched {
        /// The repository identifier.
        repo_id: String,
    },

    /// A `.safetensors` header is malformed or cannot be parsed.
    #[error("safetensors header error for {filename}: {reason}")]
    SafetensorsHeader {
        /// The filename whose header failed to parse.
        filename: String,
        /// Description of the parse failure.
        reason: String,
    },

    /// `inspect` was asked to read a file whose extension is not supported.
    ///
    /// Emitted before any parse attempt so users see a clear format mismatch
    /// rather than a misleading header-parse error.
    #[error("hf-fm inspect supports .safetensors, .gguf, .npz, or .pth (got .{extension} for {filename})")]
    UnsupportedInspectFormat {
        /// The filename whose extension is unsupported.
        filename: String,
        /// The actual extension without the leading dot, or `unknown` if none.
        extension: String,
    },
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

/// Formats a list of file failures for inclusion in the `PartialDownload` error message.
fn format_failures(failures: &[FileFailure]) -> String {
    let mut s = String::new();
    for f in failures {
        s.push_str("\n  - ");
        s.push_str(f.filename.as_str());
        s.push_str(": ");
        s.push_str(f.reason.as_str());
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_format_error_lists_all_four_formats() {
        // v0.10.3 Phase B commit 7: the `UnsupportedInspectFormat` wording
        // now names every format the cached-inspect dispatcher handles —
        // .safetensors (remote or cached), .gguf / .npz / .pth (cached only,
        // until the `HttpRangeReader` adapter lands in v0.11).
        let e = FetchError::UnsupportedInspectFormat {
            filename: "weights.pt".to_owned(),
            extension: "pt".to_owned(),
        };
        let msg = e.to_string();
        for ext in [".safetensors", ".gguf", ".npz", ".pth"] {
            assert!(msg.contains(ext), "Display message missing {ext}: {msg}");
        }
        // Sanity: the unrecognised extension and filename are still surfaced.
        assert!(
            msg.contains(".pt"),
            "should name the offending extension: {msg}"
        );
        assert!(
            msg.contains("weights.pt"),
            "should name the filename: {msg}"
        );
    }
}
