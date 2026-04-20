// SPDX-License-Identifier: MIT OR Apache-2.0

//! Configuration for model downloads.
//!
//! [`FetchConfig`] controls revision, authentication, file filtering,
//! concurrency, timeouts, retry behavior, and progress reporting.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use globset::{Glob, GlobSet, GlobSetBuilder};

use crate::error::FetchError;
use crate::progress::ProgressEvent;

// TRAIT_OBJECT: heterogeneous progress handlers from different callers
pub(crate) type ProgressCallback = Arc<dyn Fn(&ProgressEvent) + Send + Sync>;

/// Configuration for downloading a model repository.
///
/// Use [`FetchConfig::builder()`] to construct.
///
/// # Example
///
/// ```rust
/// # fn example() -> Result<(), hf_fetch_model::FetchError> {
/// use hf_fetch_model::FetchConfig;
///
/// let config = FetchConfig::builder()
///     .revision("main")
///     .filter("*.safetensors")
///     .concurrency(4)
///     .build()?;
/// # Ok(())
/// # }
/// ```
pub struct FetchConfig {
    /// Git revision (branch, tag, or commit SHA). `None` means `"main"`.
    pub(crate) revision: Option<String>,
    /// Authentication token for gated/private repositories.
    pub(crate) token: Option<String>,
    /// Compiled include glob patterns. Only matching files are downloaded.
    pub(crate) include: Option<GlobSet>,
    /// Compiled exclude glob patterns. Matching files are skipped.
    pub(crate) exclude: Option<GlobSet>,
    /// Number of files to download in parallel.
    pub(crate) concurrency: usize,
    /// Custom cache directory (overrides the default HF cache).
    pub(crate) output_dir: Option<PathBuf>,
    /// Maximum time allowed for a single file download.
    pub(crate) timeout_per_file: Option<Duration>,
    /// Maximum total time for the entire download operation.
    pub(crate) timeout_total: Option<Duration>,
    /// Maximum retry attempts per file (exponential backoff with jitter).
    pub(crate) max_retries: u32,
    /// Whether to verify SHA256 checksums against HF LFS metadata.
    pub(crate) verify_checksums: bool,
    /// Minimum file size (bytes) for multi-connection chunked download.
    pub(crate) chunk_threshold: u64,
    /// Number of parallel HTTP Range connections per large file.
    pub(crate) connections_per_file: usize,
    // TRAIT_OBJECT: heterogeneous progress handlers from different callers
    /// Progress callback invoked for each download event.
    pub(crate) on_progress: Option<ProgressCallback>,
    /// Tracks which performance fields the user explicitly set via the builder.
    pub(crate) explicit: ExplicitSettings,
}

/// Tracks which performance fields the user explicitly set via the builder.
///
/// Used by the implicit plan retrofit: when a field was not explicitly set,
/// the plan-based optimizer may override it with a recommended value.
#[derive(Debug, Clone, Default)]
pub(crate) struct ExplicitSettings {
    /// Whether `concurrency` was explicitly set by the caller.
    pub(crate) concurrency: bool,
    /// Whether `connections_per_file` was explicitly set by the caller.
    pub(crate) connections_per_file: bool,
    /// Whether `chunk_threshold` was explicitly set by the caller.
    pub(crate) chunk_threshold: bool,
}

impl std::fmt::Debug for FetchConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FetchConfig")
            .field("revision", &self.revision)
            .field("token", &self.token.as_ref().map(|_| "***"))
            .field("include", &self.include)
            .field("exclude", &self.exclude)
            .field("concurrency", &self.concurrency)
            .field("output_dir", &self.output_dir)
            .field("timeout_per_file", &self.timeout_per_file)
            .field("timeout_total", &self.timeout_total)
            .field("max_retries", &self.max_retries)
            .field("verify_checksums", &self.verify_checksums)
            .field("chunk_threshold", &self.chunk_threshold)
            .field("connections_per_file", &self.connections_per_file)
            .field(
                "on_progress",
                if self.on_progress.is_some() {
                    &"Some(<fn>)"
                } else {
                    &"None"
                },
            )
            .field("explicit", &self.explicit)
            .finish()
    }
}

impl FetchConfig {
    /// Creates a new [`FetchConfigBuilder`].
    #[must_use]
    pub fn builder() -> FetchConfigBuilder {
        FetchConfigBuilder::default()
    }

    /// Returns the configured concurrency level (parallel file downloads).
    #[must_use]
    pub const fn concurrency(&self) -> usize {
        self.concurrency
    }

    /// Returns the configured number of parallel HTTP connections per file.
    #[must_use]
    pub const fn connections_per_file(&self) -> usize {
        self.connections_per_file
    }

    /// Returns the chunk threshold in bytes (minimum file size for
    /// multi-connection chunked downloads).
    #[must_use]
    pub const fn chunk_threshold(&self) -> u64 {
        self.chunk_threshold
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
    output_dir: Option<PathBuf>,
    timeout_per_file: Option<Duration>,
    timeout_total: Option<Duration>,
    max_retries: Option<u32>,
    verify_checksums: Option<bool>,
    chunk_threshold: Option<u64>,
    connections_per_file: Option<usize>,
    on_progress: Option<ProgressCallback>,
}

impl FetchConfigBuilder {
    /// Sets the git revision (branch, tag, or commit SHA) to download.
    ///
    /// Defaults to `"main"` if not set.
    #[must_use]
    pub fn revision(mut self, revision: &str) -> Self {
        // BORROW: explicit .to_owned() for &str → owned String
        self.revision = Some(revision.to_owned());
        self
    }

    /// Sets the authentication token.
    #[must_use]
    pub fn token(mut self, token: &str) -> Self {
        // BORROW: explicit .to_owned() for &str → owned String
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
        // BORROW: explicit .to_owned() for &str → owned String
        self.include_patterns.push(pattern.to_owned());
        self
    }

    /// Adds an exclude glob pattern. Files matching any exclude pattern
    /// will be skipped, even if they match an include pattern.
    ///
    /// Can be called multiple times to add multiple patterns.
    #[must_use]
    pub fn exclude(mut self, pattern: &str) -> Self {
        // BORROW: explicit .to_owned() for &str → owned String
        self.exclude_patterns.push(pattern.to_owned());
        self
    }

    /// Sets the number of files to download concurrently.
    ///
    /// When omitted, the download plan optimizer auto-tunes this value
    /// based on file count and size distribution. Falls back to 4 if
    /// no plan recommendation is available.
    #[must_use]
    pub fn concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = Some(concurrency);
        self
    }

    /// Sets a custom output directory for downloaded files.
    ///
    /// By default, files are stored in the standard `HuggingFace` cache directory
    /// (`~/.cache/huggingface/hub/`). When set, the `HuggingFace` cache hierarchy
    /// is created inside this directory instead.
    #[must_use]
    pub fn output_dir(mut self, dir: PathBuf) -> Self {
        self.output_dir = Some(dir);
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

    /// Sets the minimum file size (in bytes) for chunked parallel download.
    ///
    /// Files at or above this threshold are downloaded using multiple HTTP
    /// Range connections in parallel. Files below use the standard single
    /// connection. Set to `u64::MAX` to disable chunked downloads entirely.
    ///
    /// When omitted, the download plan optimizer auto-tunes this value
    /// based on file size distribution. Falls back to 100 MiB
    /// (104\_857\_600 bytes) if no plan recommendation is available.
    #[must_use]
    pub fn chunk_threshold(mut self, bytes: u64) -> Self {
        self.chunk_threshold = Some(bytes);
        self
    }

    /// Sets the number of parallel HTTP connections per large file.
    ///
    /// Only applies to files at or above `chunk_threshold`. When omitted,
    /// the download plan optimizer auto-tunes this value based on file size
    /// distribution. Falls back to 8 if no plan recommendation is available.
    #[must_use]
    pub fn connections_per_file(mut self, connections: usize) -> Self {
        self.connections_per_file = Some(connections);
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

    /// Creates a `tokio::sync::watch` channel for async progress consumption.
    ///
    /// Returns `(self, receiver)`. The receiver yields the latest [`ProgressEvent`]
    /// via `.changed().await` + `.borrow()`. Only the most recent event is retained.
    ///
    /// The channel is initialized with [`ProgressEvent::default()`] (all zeros,
    /// empty filename). Use `.changed().await` rather than eager `.borrow()` to
    /// avoid observing this sentinel value before the first real event.
    ///
    /// Composes with [`on_progress()`](Self::on_progress) — if a callback was
    /// already set, both the callback and the watch channel fire for every event.
    #[must_use]
    pub fn progress_channel(mut self) -> (Self, crate::progress::ProgressReceiver) {
        let (tx, rx) = tokio::sync::watch::channel(ProgressEvent::default());
        let existing = self.on_progress.take();
        // TRAIT_OBJECT: heterogeneous progress handlers composed with watch sender
        self.on_progress = Some(Arc::new(move |event: &ProgressEvent| {
            if let Some(ref cb) = existing {
                cb(event);
            }
            // Skip clone + send if no receiver is listening.
            if tx.receiver_count() > 0 {
                // BORROW: explicit .clone() for owned ProgressEvent sent through watch channel
                let _ = tx.send(event.clone());
            }
        }));
        (self, rx)
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
            output_dir: self.output_dir,
            timeout_per_file: self.timeout_per_file,
            timeout_total: self.timeout_total,
            max_retries: self.max_retries.unwrap_or(3),
            verify_checksums: self.verify_checksums.unwrap_or(true),
            chunk_threshold: self.chunk_threshold.unwrap_or(104_857_600),
            connections_per_file: self.connections_per_file.unwrap_or(8).max(1),
            on_progress: self.on_progress,
            explicit: ExplicitSettings {
                concurrency: self.concurrency.is_some(),
                connections_per_file: self.connections_per_file.is_some(),
                chunk_threshold: self.chunk_threshold.is_some(),
            },
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

    /// Returns a builder pre-configured to download only `.npz` and `.npy`
    /// files plus common config files. Matches NumPy-based weight repos
    /// such as Google's `GemmaScope` transcoders (`config.yaml` + many `.npz`).
    #[must_use]
    pub fn npz() -> FetchConfigBuilder {
        FetchConfigBuilder::default()
            .filter("*.npz")
            .filter("*.npy")
            .filter("config.yaml")
            .filter("*.json")
            .filter("*.txt")
    }

    /// Returns a builder pre-configured to download only `pytorch_model*.bin`
    /// files plus common config files.
    #[must_use]
    pub fn pth() -> FetchConfigBuilder {
        FetchConfigBuilder::default()
            .filter("pytorch_model*.bin")
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

/// Returns whether `filename` passes the given include/exclude glob filters.
///
/// A file matches when it is not excluded by any `exclude` pattern **and**
/// either there are no `include` patterns or it matches at least one.
#[must_use]
pub fn file_matches(filename: &str, include: Option<&GlobSet>, exclude: Option<&GlobSet>) -> bool {
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

/// Builds a compiled [`GlobSet`] from a list of pattern strings.
///
/// Returns `None` if the pattern list is empty. This is useful for callers
/// that need glob filtering outside the download pipeline (e.g., the
/// `list-files` subcommand).
///
/// # Errors
///
/// Returns [`FetchError::InvalidPattern`] if any pattern fails to compile.
pub fn compile_glob_patterns(patterns: &[String]) -> Result<Option<GlobSet>, FetchError> {
    build_globset(patterns)
}

/// Returns `true` if `s` contains glob metacharacters (`*`, `?`, `[`, `{`).
///
/// Useful for detecting whether a user-supplied filename should be treated
/// as a glob pattern or an exact match.
#[must_use]
pub fn has_glob_chars(s: &str) -> bool {
    s.bytes().any(|b| matches!(b, b'*' | b'?' | b'[' | b'{'))
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

    #[test]
    fn test_has_glob_chars() {
        assert!(has_glob_chars("*.safetensors"));
        assert!(has_glob_chars("model-[0-9].bin"));
        assert!(has_glob_chars("model?.bin"));
        assert!(has_glob_chars("{a,b}.bin"));
        assert!(!has_glob_chars("model.safetensors"));
        assert!(!has_glob_chars("config.json"));
        assert!(!has_glob_chars("pytorch_model.bin"));
    }
}
