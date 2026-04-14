// SPDX-License-Identifier: MIT OR Apache-2.0

//! Download plan: metadata-only analysis of what needs downloading.
//!
//! A [`DownloadPlan`] describes which files in a remote `HuggingFace`
//! repository need downloading and which are already cached locally.
//! Use [`download_plan()`] to compute a plan, then inspect it or pass
//! it to [`recommended_config()`](DownloadPlan::recommended_config) for
//! optimized download settings.

use crate::cache;
use crate::cache_layout;
use crate::chunked;
use crate::config::{self, FetchConfig, FetchConfigBuilder};
use crate::error::FetchError;
use crate::repo;

/// Size threshold for "large" files (1 GiB).
const LARGE_FILE_THRESHOLD: u64 = 1_073_741_824;

/// Size threshold for "very large" files (5 GiB).
const VERY_LARGE_FILE_THRESHOLD: u64 = 5_368_709_120;

/// Size threshold for "small" files (10 MiB).
const SMALL_FILE_THRESHOLD: u64 = 10_485_760;

/// Default chunk threshold (100 MiB).
const DEFAULT_CHUNK_THRESHOLD: u64 = 104_857_600;

/// A download plan describing which files need downloading and which are cached.
///
/// Created by [`download_plan()`]. Contains per-file metadata and aggregate
/// byte counts. Use [`recommended_config()`](Self::recommended_config) to
/// compute an optimized [`FetchConfig`] based on the file size distribution.
#[derive(Debug, Clone)]
pub struct DownloadPlan {
    /// Repository identifier (e.g., `"google/gemma-2-2b-it"`).
    pub repo_id: String,
    /// Resolved revision (commit hash or branch name).
    pub revision: String,
    /// Per-file plan entries.
    pub files: Vec<FilePlan>,
    /// Total bytes across all files (cached + uncached).
    pub total_bytes: u64,
    /// Bytes already present in local cache.
    pub cached_bytes: u64,
    /// Bytes that need downloading.
    pub download_bytes: u64,
}

/// Per-file entry within a [`DownloadPlan`].
#[derive(Debug, Clone)]
pub struct FilePlan {
    /// Filename within the repository.
    pub filename: String,
    /// File size in bytes (0 if unknown).
    pub size: u64,
    /// Whether the file is already cached locally.
    pub cached: bool,
}

impl DownloadPlan {
    /// Number of files that still need downloading.
    #[must_use]
    pub fn files_to_download(&self) -> usize {
        self.files.iter().filter(|f| !f.cached).count()
    }

    /// Whether all files are already cached (download would be a no-op).
    #[must_use]
    pub const fn fully_cached(&self) -> bool {
        self.download_bytes == 0
    }

    /// Computes an optimized [`FetchConfig`] based on the size distribution
    /// of uncached files.
    ///
    /// The returned config has no `token`, `revision`, `on_progress`, or
    /// glob filters set — only the performance-tuning fields (`concurrency`,
    /// `connections_per_file`, `chunk_threshold`). Merge with user config
    /// before use.
    ///
    /// # Errors
    ///
    /// Returns [`FetchError::InvalidPattern`] if the internal builder fails
    /// (should not happen since no patterns are set).
    pub fn recommended_config(&self) -> Result<FetchConfig, FetchError> {
        self.recommended_config_builder().build()
    }

    /// Like [`recommended_config()`](Self::recommended_config) but returns a
    /// [`FetchConfigBuilder`] so the caller can override specific fields.
    #[must_use]
    pub fn recommended_config_builder(&self) -> FetchConfigBuilder {
        let uncached: Vec<u64> = self
            .files
            .iter()
            .filter(|f| !f.cached)
            .map(|f| f.size)
            .collect();

        let builder = FetchConfig::builder();

        if uncached.is_empty() {
            // All cached — defaults are fine, download will be a no-op.
            return builder.concurrency(1);
        }

        let count = uncached.len();
        let large_count = uncached
            .iter()
            .filter(|&&s| s >= LARGE_FILE_THRESHOLD)
            .count();
        let very_large = uncached.iter().any(|&s| s >= VERY_LARGE_FILE_THRESHOLD);
        let small_count = uncached
            .iter()
            .filter(|&&s| s < SMALL_FILE_THRESHOLD)
            .count();

        // Strategy: few large files — maximize per-file parallelism.
        if count <= 2 && large_count > 0 {
            let connections = if very_large { 16 } else { 8 };
            return builder
                .concurrency(count.max(1))
                .connections_per_file(connections)
                .chunk_threshold(DEFAULT_CHUNK_THRESHOLD);
        }

        // Strategy: many small files — parallelize at file level.
        // Only applies when there are NO large files; otherwise fall through
        // to the mixed strategy which handles both small and large files.
        if small_count > count / 2 && large_count == 0 {
            return builder
                .concurrency(8.min(count))
                .connections_per_file(1)
                .chunk_threshold(u64::MAX);
        }

        // Strategy: mixed — balanced defaults.
        builder
            .concurrency(4)
            .connections_per_file(8)
            .chunk_threshold(DEFAULT_CHUNK_THRESHOLD)
    }
}

/// Computes a download plan for a repository without downloading anything.
///
/// Fetches remote file metadata and compares against the local cache to
/// classify each file as cached or pending download. Glob filters from
/// `config` are applied.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the `HuggingFace` API request fails.
/// Returns [`FetchError::RepoNotFound`] if the repository does not exist.
/// Returns [`FetchError::Io`] if the cache directory cannot be resolved.
pub async fn download_plan(
    repo_id: &str,
    config: &FetchConfig,
) -> Result<DownloadPlan, FetchError> {
    let revision_str = config.revision.as_deref().unwrap_or("main");
    let token = config.token.as_deref();

    // Fetch remote file list with metadata.
    let client = chunked::build_client(token)?;
    let remote_files =
        repo::list_repo_files_with_metadata(repo_id, token, Some(revision_str), &client).await?;

    // Apply glob filters.
    let filtered: Vec<_> = remote_files
        .into_iter()
        .filter(|f| {
            // BORROW: explicit .as_str() instead of Deref coercion
            config::file_matches(
                f.filename.as_str(),
                config.include.as_ref(),
                config.exclude.as_ref(),
            )
        })
        .collect();

    // Resolve cache state.
    let cache_dir = config
        .output_dir
        .clone()
        .map_or_else(cache::hf_cache_dir, Ok)?;
    let repo_dir = cache_layout::repo_dir(&cache_dir, repo_id);
    let commit_hash = cache::read_ref(&repo_dir, revision_str);
    let snapshot_dir = commit_hash
        .as_deref()
        .map(|hash| cache_layout::snapshot_dir(&repo_dir, hash));

    let mut total_bytes: u64 = 0;
    let mut cached_bytes: u64 = 0;
    let mut files = Vec::with_capacity(filtered.len());

    for rf in &filtered {
        let size = rf.size.unwrap_or(0);
        total_bytes = total_bytes.saturating_add(size);

        let cached = snapshot_dir
            .as_ref()
            // BORROW: explicit .as_str() instead of Deref coercion
            .is_some_and(|dir| dir.join(rf.filename.as_str()).exists());

        if cached {
            cached_bytes = cached_bytes.saturating_add(size);
        }

        files.push(FilePlan {
            // BORROW: explicit .clone() for owned String field
            filename: rf.filename.clone(),
            size,
            cached,
        });
    }

    let download_bytes = total_bytes.saturating_sub(cached_bytes);

    Ok(DownloadPlan {
        // BORROW: explicit .to_owned() for &str → owned String
        repo_id: repo_id.to_owned(),
        // BORROW: explicit .to_owned() for &str → owned String
        revision: commit_hash.unwrap_or_else(|| revision_str.to_owned()),
        files,
        total_bytes,
        cached_bytes,
        download_bytes,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// Builds a `DownloadPlan` from a list of `(size, cached)` pairs.
    fn make_plan(file_specs: &[(u64, bool)]) -> DownloadPlan {
        let mut total_bytes: u64 = 0;
        let mut cached_bytes: u64 = 0;
        let files: Vec<FilePlan> = file_specs
            .iter()
            .enumerate()
            .map(|(i, &(size, cached))| {
                total_bytes = total_bytes.saturating_add(size);
                if cached {
                    cached_bytes = cached_bytes.saturating_add(size);
                }
                FilePlan {
                    filename: format!("file_{i}.bin"),
                    size,
                    cached,
                }
            })
            .collect();

        DownloadPlan {
            repo_id: "test/repo".to_owned(),
            revision: "main".to_owned(),
            files,
            total_bytes,
            cached_bytes,
            download_bytes: total_bytes.saturating_sub(cached_bytes),
        }
    }

    #[test]
    fn all_cached_returns_concurrency_one() {
        let plan = make_plan(&[(1_000_000, true), (2_000_000, true)]);
        assert!(plan.fully_cached());
        assert_eq!(plan.files_to_download(), 0);
        let config = plan.recommended_config().unwrap();
        assert_eq!(config.concurrency(), 1);
    }

    #[test]
    fn single_very_large_file_gets_sixteen_connections() {
        // 6 GiB file, uncached.
        let plan = make_plan(&[(6_442_450_944, false)]);
        assert_eq!(plan.files_to_download(), 1);
        let config = plan.recommended_config().unwrap();
        assert_eq!(config.concurrency(), 1);
        assert_eq!(config.connections_per_file(), 16);
    }

    #[test]
    fn two_large_files_get_eight_connections() {
        // Two 2 GiB files, uncached.
        let plan = make_plan(&[(2_147_483_648, false), (2_147_483_648, false)]);
        assert_eq!(plan.files_to_download(), 2);
        let config = plan.recommended_config().unwrap();
        assert_eq!(config.concurrency(), 2);
        assert_eq!(config.connections_per_file(), 8);
    }

    #[test]
    fn many_small_files_get_high_concurrency_single_connection() {
        // 20 small files (1 MiB each), uncached.
        let specs: Vec<(u64, bool)> = (0..20).map(|_| (1_048_576, false)).collect();
        let plan = make_plan(&specs);
        assert_eq!(plan.files_to_download(), 20);
        let config = plan.recommended_config().unwrap();
        assert_eq!(config.concurrency(), 8);
        assert_eq!(config.connections_per_file(), 1);
        assert_eq!(config.chunk_threshold(), u64::MAX);
    }

    #[test]
    fn mixed_sizes_get_balanced_defaults() {
        // Mix of large and medium files — fewer than half are small.
        let plan = make_plan(&[
            (2_147_483_648, false), // 2 GiB
            (104_857_600, false),   // 100 MiB
            (52_428_800, false),    // 50 MiB
            (1_073_741_824, false), // 1 GiB
            (20_971_520, false),    // 20 MiB
        ]);
        assert_eq!(plan.files_to_download(), 5);
        let config = plan.recommended_config().unwrap();
        assert_eq!(config.concurrency(), 4);
        assert_eq!(config.connections_per_file(), 8);
        assert_eq!(config.chunk_threshold(), DEFAULT_CHUNK_THRESHOLD);
    }

    #[test]
    fn mostly_small_with_large_files_uses_mixed_strategy() {
        // Mirrors the Ministral-3-3B case: 2 large files + 8 small files.
        // Should NOT pick the "many small files" strategy because large
        // files are present — falls through to "mixed" instead.
        let plan = make_plan(&[
            (4_672_561_152, false), // 4.35 GiB
            (4_672_561_152, false), // 4.35 GiB
            (2_355, false),         // 2.3 KiB
            (1_946, false),         // 1.9 KiB
            (131, false),           // 131 B
            (1_229, false),         // 1.2 KiB
            (976, false),           // 976 B
            (16_756_736, false),    // 16 MiB
            (17_081_344, false),    // 16.3 MiB
            (21_197, false),        // 20.7 KiB
        ]);
        assert_eq!(plan.files_to_download(), 10);
        let config = plan.recommended_config().unwrap();
        // Mixed strategy: balanced concurrency with chunked downloads enabled.
        assert_eq!(config.concurrency(), 4);
        assert_eq!(config.connections_per_file(), 8);
        assert_eq!(config.chunk_threshold(), DEFAULT_CHUNK_THRESHOLD);
    }
}
