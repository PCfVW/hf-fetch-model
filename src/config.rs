// SPDX-License-Identifier: MIT OR Apache-2.0

//! Configuration for model downloads.
//!
//! [`FetchConfig`] controls revision, authentication, file filtering,
//! concurrency, timeouts, retry behavior, and progress reporting.

use std::sync::Arc;
use std::time::Duration;

use globset::{Glob, GlobSet, GlobSetBuilder};

use crate::error::FetchError;
use crate::progress::ProgressEvent;

// TRAIT_OBJECT: heterogeneous progress handlers from different callers
type ProgressCallback = Arc<dyn Fn(&ProgressEvent) + Send + Sync>;

/// Configuration for downloading a model repository.
///
/// Use [`FetchConfig::builder()`] to construct.
///
/// # Example
///
/// ```rust
/// use hf_fetch_model::FetchConfig;
///
/// let config = FetchConfig::builder()
///     .revision("main")
///     .filter("*.safetensors")
///     .concurrency(4)
///     .build()
///     .unwrap();
/// ```
pub struct FetchConfig {
    pub(crate) revision: Option<String>,
    pub(crate) token: Option<String>,
    pub(crate) include: Option<GlobSet>,
    pub(crate) exclude: Option<GlobSet>,
    pub(crate) concurrency: usize,
    pub(crate) timeout_per_file: Option<Duration>,
    pub(crate) timeout_total: Option<Duration>,
    pub(crate) max_retries: u32,
    pub(crate) verify_checksums: bool,
    // TRAIT_OBJECT: heterogeneous progress handlers from different callers
    pub(crate) on_progress: Option<ProgressCallback>,
}

impl std::fmt::Debug for FetchConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FetchConfig")
            .field("revision", &self.revision)
            .field("token", &self.token.as_ref().map(|_| "***"))
            .field("include", &self.include)
            .field("exclude", &self.exclude)
            .field("concurrency", &self.concurrency)
            .field("timeout_per_file", &self.timeout_per_file)
            .field("timeout_total", &self.timeout_total)
            .field("max_retries", &self.max_retries)
            .field("verify_checksums", &self.verify_checksums)
            .field(
                "on_progress",
                if self.on_progress.is_some() {
                    &"Some(<fn>)"
                } else {
                    &"None"
                },
            )
            .finish()
    }
}

impl FetchConfig {
    /// Creates a new [`FetchConfigBuilder`].
    #[must_use]
    pub fn builder() -> FetchConfigBuilder {
        FetchConfigBuilder::default()
    }
}

/// Builder for [`FetchConfig`].
#[derive(Default)]
pub struct FetchConfigBuilder {
    revision: Option<String>,
    token: Option<String>,
    include_patterns: Vec<String>,
    exclude_patterns: Vec<String>,
    concurrency: Option<usize>,
    timeout_per_file: Option<Duration>,
    timeout_total: Option<Duration>,
    max_retries: Option<u32>,
    verify_checksums: Option<bool>,
    on_progress: Option<ProgressCallback>,
}

impl FetchConfigBuilder {
    /// Sets the git revision (branch, tag, or commit SHA) to download.
    ///
    /// Defaults to `"main"` if not set.
    #[must_use]
    pub fn revision(mut self, revision: &str) -> Self {
        self.revision = Some(revision.to_owned());
        self
    }

    /// Sets the authentication token.
    #[must_use]
    pub fn token(mut self, token: &str) -> Self {
        self.token = Some(token.to_owned());
        self
    }

    /// Reads the authentication token from the `HF_TOKEN` environment variable.
    #[must_use]
    pub fn token_from_env(mut self) -> Self {
        self.token = std::env::var("HF_TOKEN").ok();
        self
    }

    /// Adds an include glob pattern. Only files matching at least one
    /// include pattern will be downloaded.
    ///
    /// Can be called multiple times to add multiple patterns.
    #[must_use]
    pub fn filter(mut self, pattern: &str) -> Self {
        self.include_patterns.push(pattern.to_owned());
        self
    }

    /// Adds an exclude glob pattern. Files matching any exclude pattern
    /// will be skipped, even if they match an include pattern.
    ///
    /// Can be called multiple times to add multiple patterns.
    #[must_use]
    pub fn exclude(mut self, pattern: &str) -> Self {
        self.exclude_patterns.push(pattern.to_owned());
        self
    }

    /// Sets the number of files to download concurrently.
    ///
    /// Defaults to 4.
    #[must_use]
    pub fn concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = Some(concurrency);
        self
    }

    /// Sets the maximum time allowed per file download.
    ///
    /// If a single file download exceeds this duration, it is aborted
    /// and may be retried according to the retry policy.
    #[must_use]
    pub fn timeout_per_file(mut self, duration: Duration) -> Self {
        self.timeout_per_file = Some(duration);
        self
    }

    /// Sets the maximum total time for the entire download operation.
    ///
    /// If the total download time exceeds this duration, remaining files
    /// are skipped and a [`FetchError::Timeout`] is returned.
    #[must_use]
    pub fn timeout_total(mut self, duration: Duration) -> Self {
        self.timeout_total = Some(duration);
        self
    }

    /// Sets the maximum number of retry attempts per file.
    ///
    /// Defaults to 3. Set to 0 to disable retries.
    /// Uses exponential backoff with jitter (base 300ms, cap 10s).
    #[must_use]
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.max_retries = Some(retries);
        self
    }

    /// Enables or disables SHA256 checksum verification after download.
    ///
    /// When enabled, downloaded files are verified against the SHA256 hash
    /// from `HuggingFace` LFS metadata. Files without LFS metadata (small
    /// config files stored directly in git) are skipped.
    ///
    /// Defaults to `true`.
    #[must_use]
    pub fn verify_checksums(mut self, verify: bool) -> Self {
        self.verify_checksums = Some(verify);
        self
    }

    /// Sets a progress callback invoked for each progress event.
    #[must_use]
    pub fn on_progress<F>(mut self, callback: F) -> Self
    where
        F: Fn(&ProgressEvent) + Send + Sync + 'static,
    {
        self.on_progress = Some(Arc::new(callback));
        self
    }

    /// Builds the [`FetchConfig`].
    ///
    /// # Errors
    ///
    /// Returns [`FetchError::InvalidPattern`] if any glob pattern is invalid.
    pub fn build(self) -> Result<FetchConfig, FetchError> {
        let include = build_globset(&self.include_patterns)?;
        let exclude = build_globset(&self.exclude_patterns)?;

        Ok(FetchConfig {
            revision: self.revision,
            token: self.token,
            include,
            exclude,
            concurrency: self.concurrency.unwrap_or(4),
            timeout_per_file: self.timeout_per_file,
            timeout_total: self.timeout_total,
            max_retries: self.max_retries.unwrap_or(3),
            verify_checksums: self.verify_checksums.unwrap_or(true),
            on_progress: self.on_progress,
        })
    }
}

/// Common filter presets for typical download patterns.
#[non_exhaustive]
pub struct Filter;

impl Filter {
    /// Returns a builder pre-configured to download only `*.safetensors` files
    /// plus common config files.
    #[must_use]
    pub fn safetensors() -> FetchConfigBuilder {
        FetchConfigBuilder::default()
            .filter("*.safetensors")
            .filter("*.json")
            .filter("*.txt")
    }

    /// Returns a builder pre-configured to download only GGUF files
    /// plus common config files.
    #[must_use]
    pub fn gguf() -> FetchConfigBuilder {
        FetchConfigBuilder::default()
            .filter("*.gguf")
            .filter("*.json")
            .filter("*.txt")
    }

    /// Returns a builder pre-configured to download only config files
    /// (no model weights).
    #[must_use]
    pub fn config_only() -> FetchConfigBuilder {
        FetchConfigBuilder::default()
            .filter("*.json")
            .filter("*.txt")
            .filter("*.md")
    }
}

/// Returns whether a filename passes the include/exclude filters.
#[must_use]
pub(crate) fn file_matches(
    filename: &str,
    include: Option<&GlobSet>,
    exclude: Option<&GlobSet>,
) -> bool {
    if let Some(exc) = exclude {
        if exc.is_match(filename) {
            return false;
        }
    }
    if let Some(inc) = include {
        return inc.is_match(filename);
    }
    true
}

fn build_globset(patterns: &[String]) -> Result<Option<GlobSet>, FetchError> {
    if patterns.is_empty() {
        return Ok(None);
    }
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        // BORROW: explicit .as_str() instead of Deref coercion
        let glob = Glob::new(pattern.as_str()).map_err(|e| FetchError::InvalidPattern {
            pattern: pattern.clone(),
            reason: e.to_string(),
        })?;
        builder.add(glob);
    }
    let set = builder.build().map_err(|e| FetchError::InvalidPattern {
        pattern: patterns.join(", "),
        reason: e.to_string(),
    })?;
    Ok(Some(set))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn test_file_matches_no_filters() {
        assert!(file_matches("model.safetensors", None, None));
    }

    #[test]
    fn test_file_matches_include() {
        let include = build_globset(&["*.safetensors".to_owned()]).unwrap();
        assert!(file_matches("model.safetensors", include.as_ref(), None));
        assert!(!file_matches("model.bin", include.as_ref(), None));
    }

    #[test]
    fn test_file_matches_exclude() {
        let exclude = build_globset(&["*.bin".to_owned()]).unwrap();
        assert!(file_matches("model.safetensors", None, exclude.as_ref()));
        assert!(!file_matches("model.bin", None, exclude.as_ref()));
    }

    #[test]
    fn test_exclude_overrides_include() {
        let include = build_globset(&["*.safetensors".to_owned(), "*.bin".to_owned()]).unwrap();
        let exclude = build_globset(&["*.bin".to_owned()]).unwrap();
        assert!(file_matches(
            "model.safetensors",
            include.as_ref(),
            exclude.as_ref()
        ));
        assert!(!file_matches(
            "model.bin",
            include.as_ref(),
            exclude.as_ref()
        ));
    }
}
