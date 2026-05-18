// SPDX-License-Identifier: MIT OR Apache-2.0

//! `HuggingFace` cache directory resolution, model family scanning, disk usage,
//! and integrity verification.
//!
//! [`hf_cache_dir()`] locates the local HF cache. [`list_cached_families()`]
//! scans downloaded models and groups them by `model_type`.
//! [`cache_summary()`] provides per-repo size totals,
//! [`cache_repo_usage()`] returns per-file disk usage for a single repo, and
//! [`verify_cache()`] re-checks `SHA256` digests of cached files against
//! `HuggingFace` LFS metadata.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::FetchError;

/// Filename of the per-repo `hf-fm` snapshot sidecar.
///
/// Lives at `{cache_root}/models--{org}--{name}/.hf-fm-snapshot.json`.
/// Written by `download` (recording the active `--preset` / `--filter` /
/// `--exclude`) and consumed by `status` (to distinguish files deliberately
/// skipped via the preset's glob list from files that are genuinely missing).
pub const SNAPSHOT_FILENAME: &str = ".hf-fm-snapshot.json";

/// Schema version of the on-disk [`Snapshot`] file. Bumped on incompatible changes.
pub const SNAPSHOT_VERSION: u32 = 1;

/// On-disk record of the arguments that produced a cached repository.
///
/// Persisted by `hf-fm download` as a small JSON file at the repository's
/// cache root, alongside `refs/`, `blobs/`, and `snapshots/`. Read back by
/// `hf-fm status` so files that don't match the recorded preset can be
/// reported as [`FileStatus::Excluded`] instead of [`FileStatus::Missing`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Snapshot {
    /// Schema version. Equals [`SNAPSHOT_VERSION`] for newly-written files.
    pub version: u32,
    /// Git revision at download time (resolved commit SHA or branch name).
    pub revision: String,
    /// The `--preset` value used at download time, if any. One of
    /// `"safetensors"`, `"gguf"`, `"npz"`, `"pth"`, `"config-only"`.
    pub preset: Option<String>,
    /// `--filter` glob patterns used at download time. Reserved for a later
    /// patch that adds `status --filter`; not yet consumed by `status`.
    pub filter: Vec<String>,
    /// `--exclude` glob patterns used at download time. Reserved for a later
    /// patch that adds `status --exclude`; not yet consumed by `status`.
    pub exclude: Vec<String>,
}

/// Returns the absolute path of the [`SNAPSHOT_FILENAME`] sidecar for a given
/// repository cache directory.
#[must_use]
pub fn snapshot_path(repo_dir: &Path) -> PathBuf {
    repo_dir.join(SNAPSHOT_FILENAME)
}

/// Reads the per-repo [`Snapshot`] sidecar if it exists.
///
/// A missing sidecar is not an error — older caches (downloaded before this
/// feature) simply return `Ok(None)`.
///
/// # Errors
///
/// Returns [`FetchError::Io`](crate::FetchError::Io) if the file exists but
/// cannot be read.
/// Returns [`FetchError::InvalidArgument`](crate::FetchError::InvalidArgument)
/// if the file is present but its JSON cannot be parsed (e.g. corruption or
/// a future schema version this binary cannot understand).
pub fn read_snapshot(repo_dir: &Path) -> Result<Option<Snapshot>, FetchError> {
    let path = snapshot_path(repo_dir);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(FetchError::Io { path, source: e }),
    };
    let snapshot: Snapshot = serde_json::from_slice(&bytes).map_err(|e| {
        FetchError::InvalidArgument(format!("failed to parse snapshot {}: {e}", path.display()))
    })?;
    Ok(Some(snapshot))
}

/// Writes the [`Snapshot`] sidecar for a repository, atomically (write to a
/// `.tmp` sibling, then rename).
///
/// Overwrites any previously-written sidecar — the design is intentionally
/// last-download-wins.
///
/// # Errors
///
/// Returns [`FetchError::Io`](crate::FetchError::Io) on any filesystem
/// failure (parent directory missing, no write permission, rename across
/// filesystems, etc.).
pub fn write_snapshot(repo_dir: &Path, snapshot: &Snapshot) -> Result<(), FetchError> {
    let path = snapshot_path(repo_dir);
    let tmp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(snapshot)
        .map_err(|e| FetchError::InvalidArgument(format!("failed to serialize snapshot: {e}")))?;
    std::fs::write(&tmp, bytes).map_err(|e| FetchError::Io {
        path: tmp.clone(),
        source: e,
    })?;
    std::fs::rename(&tmp, &path).map_err(|e| FetchError::Io {
        path: path.clone(),
        source: e,
    })?;
    Ok(())
}

/// Reconstructs a repo ID from a `models--org--name` directory name.
///
/// Returns `None` if the directory name does not start with `models--`.
fn repo_id_from_folder_name(dir_name: &str) -> Option<String> {
    let repo_part = dir_name.strip_prefix("models--")?;

    // Reconstruct repo_id: replace first "--" with "/".
    let repo_id = match repo_part.find("--") {
        Some(pos) => {
            let (org, name_with_sep) = repo_part.split_at(pos);
            let name = name_with_sep.get(2..).unwrap_or_default();
            format!("{org}/{name}")
        }
        None => repo_part.to_string(),
    };

    Some(repo_id)
}

/// Returns the `HuggingFace` Hub cache directory.
///
/// Resolution order:
/// 1. `HF_HOME` environment variable + `/hub`
/// 2. `~/.cache/huggingface/hub/` (via [`dirs::home_dir()`])
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the home directory cannot be determined.
pub fn hf_cache_dir() -> Result<PathBuf, FetchError> {
    if let Ok(home) = std::env::var("HF_HOME") {
        let mut path = PathBuf::from(home);
        path.push("hub");
        return Ok(path);
    }

    let home = dirs::home_dir().ok_or_else(|| FetchError::Io {
        path: PathBuf::from("~"),
        source: std::io::Error::new(std::io::ErrorKind::NotFound, "home directory not found"),
    })?;

    let mut path = home;
    path.push(".cache");
    path.push("huggingface");
    path.push("hub");
    Ok(path)
}

/// One repository inside a cached family, with optional quantization label.
///
/// Returned by [`list_cached_families`] so callers can render a quant column
/// alongside the repo ID without re-reading every snapshot's `config.json`
/// in a second pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FamilyEntry {
    /// The repository identifier (e.g., `"meta-llama/Llama-3.2-1B"`).
    pub repo_id: String,
    /// Quantization method as reported by `quantization_config.quant_method`
    /// in the cached `config.json`. Falls back to `"gguf"` when any cached
    /// file in the newest snapshot directory has a `.gguf` extension.
    /// `None` for full-precision repos.
    pub quant_method: Option<String>,
}

/// Scans the local HF cache for downloaded models and groups them by `model_type`.
///
/// Looks for `config.json` files inside model snapshot directories:
/// `<cache>/models--<org>--<name>/snapshots/*/config.json`
///
/// Returns a map from `model_type` (e.g., `"llama"`) to a sorted list of
/// [`FamilyEntry`] values, each pairing a repo ID with its quantization
/// label (if any).
///
/// Models without a `model_type` field in their `config.json` are skipped.
///
/// # Errors
///
/// Returns [`FetchError::Io`](crate::FetchError::Io) if the cache directory cannot be read.
pub fn list_cached_families() -> Result<BTreeMap<String, Vec<FamilyEntry>>, FetchError> {
    let cache_dir = hf_cache_dir()?;

    if !cache_dir.exists() {
        return Ok(BTreeMap::new());
    }

    let entries = std::fs::read_dir(&cache_dir).map_err(|e| FetchError::Io {
        path: cache_dir.clone(),
        source: e,
    })?;

    let mut families: BTreeMap<String, Vec<FamilyEntry>> = BTreeMap::new();

    for entry in entries {
        let Ok(entry) = entry else { continue };

        let dir_name = entry.file_name();
        // BORROW: explicit .to_string_lossy() for OsString → str conversion
        let dir_str = dir_name.to_string_lossy();

        let Some(repo_id) = repo_id_from_folder_name(&dir_str) else {
            continue;
        };

        // Find the newest snapshot's config.json
        let snapshots_dir = crate::cache_layout::snapshots_dir(&entry.path());
        if !snapshots_dir.exists() {
            continue;
        }

        if let Some((model_type, quant_method)) = find_family_info_in_snapshots(&snapshots_dir) {
            families.entry(model_type).or_default().push(FamilyEntry {
                repo_id,
                quant_method,
            });
        }
    }

    // Sort repo lists within each family for stable output
    for entries in families.values_mut() {
        entries.sort_by(|a, b| a.repo_id.cmp(&b.repo_id));
    }

    Ok(families)
}

/// Searches snapshot directories for a `config.json` containing `model_type`,
/// and reports an accompanying quantization label when one can be inferred.
///
/// Returns the first `(model_type, quant_method)` pair found. The quant
/// label comes from `quantization_config.quant_method` in `config.json`,
/// or falls back to `Some("gguf".to_owned())` when any sibling file in the
/// same snapshot directory has a `.gguf` extension. Returns `None` if no
/// snapshot yields a parseable `model_type`.
fn find_family_info_in_snapshots(
    snapshots_dir: &std::path::Path,
) -> Option<(String, Option<String>)> {
    let snapshots = std::fs::read_dir(snapshots_dir).ok()?;

    for snap_entry in snapshots {
        let Ok(snap_entry) = snap_entry else { continue };
        let snap_path = snap_entry.path();
        let config_path = snap_path.join("config.json");

        if !config_path.exists() {
            continue;
        }

        if let Some(model_type) = extract_model_type(&config_path) {
            let quant_method = extract_quant_method(&config_path)
                .or_else(|| snapshot_has_gguf(&snap_path).then(|| "gguf".to_owned())); // BORROW: explicit .to_owned()
            return Some((model_type, quant_method));
        }
    }

    None
}

/// Reads a `config.json` file and extracts the `model_type` field.
fn extract_model_type(config_path: &std::path::Path) -> Option<String> {
    let contents = std::fs::read_to_string(config_path).ok()?;
    // BORROW: explicit .as_str() instead of Deref coercion
    let value: serde_json::Value = serde_json::from_str(contents.as_str()).ok()?;
    // BORROW: explicit .as_str() on serde_json Value
    value.get("model_type")?.as_str().map(String::from)
}

/// Reads a `config.json` file and extracts the transformers-standard
/// `quantization_config.quant_method` field, if present.
///
/// Returns `None` when the file is unreadable, malformed, or contains no
/// quantization config (the repo is treated as full-precision in that case;
/// the GGUF filename fallback is applied separately by the caller).
fn extract_quant_method(config_path: &std::path::Path) -> Option<String> {
    let contents = std::fs::read_to_string(config_path).ok()?;
    // BORROW: explicit .as_str() instead of Deref coercion
    let value: serde_json::Value = serde_json::from_str(contents.as_str()).ok()?;
    // BORROW: explicit .as_str() on serde_json Value
    value
        .get("quantization_config")?
        .get("quant_method")?
        .as_str()
        .map(String::from)
}

/// Returns `true` if any file in the snapshot directory has a `.gguf` extension.
///
/// Used as a fallback quant label when `config.json` carries no
/// `quantization_config` (GGUF repos typically lack a transformers-style
/// `config.json` `quantization_config` block). Extension comparison is
/// case-insensitive via `OsStr::eq_ignore_ascii_case`.
fn snapshot_has_gguf(snapshot_dir: &std::path::Path) -> bool {
    let Ok(entries) = std::fs::read_dir(snapshot_dir) else {
        return false;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            return true;
        }
    }
    false
}

/// Status of a single file in the cache.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum FileStatus {
    /// File is fully downloaded (local size matches expected size, or no expected size known).
    Complete {
        /// Local file size in bytes.
        local_size: u64,
    },
    /// File exists but is smaller than expected (interrupted download),
    /// or a `.chunked.part` temp file was found in the blobs directory
    /// (repo-level heuristic — may not correspond to this specific file).
    Partial {
        /// Local file size in bytes.
        local_size: u64,
        /// Expected file size in bytes.
        expected_size: u64,
    },
    /// File is not present in the cache.
    Missing {
        /// Expected file size in bytes (0 if unknown).
        expected_size: u64,
    },
    /// File is on the Hub but was deliberately not requested at download time
    /// (or is now filtered out by `status --preset <P>`). Distinguished from
    /// [`FileStatus::Missing`] so the user does not chase a "fix" for an
    /// intentional skip.
    Excluded {
        /// Expected file size in bytes (0 if unknown).
        expected_size: u64,
    },
}

/// Cache status report for a repository.
#[derive(Debug, Clone)]
pub struct RepoStatus {
    /// The repository identifier.
    pub repo_id: String,
    /// The resolved commit hash (if available).
    pub commit_hash: Option<String>,
    /// The cache directory for this repo.
    pub cache_path: PathBuf,
    /// Per-file status, sorted by filename.
    pub files: Vec<(String, FileStatus)>,
}

impl RepoStatus {
    /// Number of fully downloaded files.
    #[must_use]
    pub fn complete_count(&self) -> usize {
        self.files
            .iter()
            .filter(|(_, s)| matches!(s, FileStatus::Complete { .. }))
            .count()
    }

    /// Number of partially downloaded files.
    #[must_use]
    pub fn partial_count(&self) -> usize {
        self.files
            .iter()
            .filter(|(_, s)| matches!(s, FileStatus::Partial { .. }))
            .count()
    }

    /// Number of missing files.
    #[must_use]
    pub fn missing_count(&self) -> usize {
        self.files
            .iter()
            .filter(|(_, s)| matches!(s, FileStatus::Missing { .. }))
            .count()
    }

    /// Number of files deliberately excluded by the active preset / filter.
    ///
    /// Always `0` when `status` is invoked without a preset (whether from
    /// CLI or sidecar) — the `Excluded` variant requires an active filter.
    #[must_use]
    pub fn excluded_count(&self) -> usize {
        self.files
            .iter()
            .filter(|(_, s)| matches!(s, FileStatus::Excluded { .. }))
            .count()
    }
}

/// Inspects the local cache for a repository and compares against the remote file list.
///
/// When `preset_globs` is `Some`, files that do **not** match any of the
/// supplied glob patterns and are absent locally are classified as
/// [`FileStatus::Excluded`] instead of [`FileStatus::Missing`]. Files that
/// **do** match the preset and are absent are still [`FileStatus::Missing`].
/// Files that are present locally classify as `Complete` / `Partial`
/// regardless of whether they match (the user has them — by what route is
/// not status's concern).
///
/// # Arguments
///
/// * `repo_id` — The repository identifier (e.g., `"RWKV/RWKV7-Goose-World3-1.5B-HF"`).
/// * `token` — Optional authentication token.
/// * `revision` — Optional revision (defaults to `"main"`).
/// * `preset_globs` — Optional include-glob list (typically returned by
///   [`crate::config::preset_globs`]). When supplied, governs the
///   `Excluded` distinction. When `None`, no `Excluded` entries are produced.
///
/// # Notes
///
/// Partial download detection is a repo-level heuristic: if any
/// `.chunked.part` file exists in the repo's `blobs/` directory, all
/// missing files are reported as [`FileStatus::Partial`] with the partial
/// file's size. This may overcount partials when multiple files are
/// missing but only one has an incomplete blob. Exact blob-to-file
/// mapping would require LFS metadata.
///
/// # Errors
///
/// Returns [`FetchError::Http`](crate::FetchError::Http) if the API request fails.
/// Returns [`FetchError::Io`](crate::FetchError::Io) if the cache directory cannot be read.
/// Returns [`FetchError::InvalidPattern`](crate::FetchError::InvalidPattern)
/// if any of the supplied `preset_globs` patterns fails to compile.
pub async fn repo_status(
    repo_id: &str,
    token: Option<&str>,
    revision: Option<&str>,
    preset_globs: Option<&[&str]>,
) -> Result<RepoStatus, FetchError> {
    let revision = revision.unwrap_or("main");
    let cache_dir = hf_cache_dir()?;
    let repo_dir = crate::cache_layout::repo_dir(&cache_dir, repo_id);

    // Read commit hash from refs file if available.
    let commit_hash = read_ref(&repo_dir, revision);

    // Fetch remote file list with sizes.
    let client = crate::chunked::build_client(token)?;
    let remote_files =
        crate::repo::list_repo_files_with_metadata(repo_id, token, Some(revision), &client).await?;

    // Determine snapshot directory.
    // BORROW: explicit .as_deref() for Option<String> → Option<&str>
    let snapshot_dir = commit_hash
        .as_deref()
        .map(|hash| crate::cache_layout::snapshot_dir(&repo_dir, hash));

    // Pre-check for .chunked.part files in blobs directory (avoids re-scanning
    // the blobs directory for every missing file in the loop below).
    let blobs_dir = crate::cache_layout::blobs_dir(&repo_dir);
    let has_any_partial = has_partial_blob(&blobs_dir);

    // Compile preset globs once, outside the per-file loop. `compile_glob_patterns`
    // already returns `Ok(None)` for empty slices, so an empty `Some(&[])`
    // collapses to the "no filter" case below.
    let preset_globset: Option<globset::GlobSet> = if let Some(patterns) = preset_globs {
        // BORROW: explicit .to_string() — compile_glob_patterns expects &[String]
        let owned: Vec<String> = patterns.iter().map(|s| (*s).to_string()).collect();
        crate::compile_glob_patterns(&owned)?
    } else {
        None
    };

    // Cross-reference remote files against local state.
    let mut files: Vec<(String, FileStatus)> = Vec::with_capacity(remote_files.len());

    for remote in &remote_files {
        let expected_size = remote.size.unwrap_or(0);

        let local_path = snapshot_dir
            .as_ref()
            // BORROW: explicit .as_str() for path construction
            .map(|dir| dir.join(remote.filename.as_str()));

        let status = if let Some(ref path) = local_path {
            if path.exists() {
                let local_size = std::fs::metadata(path).map_or(0, |m| m.len());

                if expected_size > 0 && local_size < expected_size {
                    FileStatus::Partial {
                        local_size,
                        expected_size,
                    }
                } else {
                    FileStatus::Complete { local_size }
                }
            } else if has_any_partial {
                // Blobs directory has .chunked.part temp files
                let part_size = find_partial_blob_size(&blobs_dir);
                FileStatus::Partial {
                    local_size: part_size,
                    expected_size,
                }
            } else if preset_globset
                .as_ref()
                // BORROW: explicit .as_str() — globset matches against &str
                .is_some_and(|gs| !gs.is_match(remote.filename.as_str()))
            {
                // File is absent AND the active preset deliberately excludes it.
                FileStatus::Excluded { expected_size }
            } else {
                FileStatus::Missing { expected_size }
            }
        } else {
            FileStatus::Missing { expected_size }
        };

        // BORROW: explicit .clone() for owned String
        files.push((remote.filename.clone(), status));
    }

    files.sort_by(|(a, _), (b, _)| a.cmp(b));

    // BORROW: explicit .to_owned() for &str → owned String field
    Ok(RepoStatus {
        repo_id: repo_id.to_owned(),
        commit_hash,
        cache_path: repo_dir,
        files,
    })
}

/// Summary of a single cached model (local-only, no API calls).
#[derive(Debug, Clone)]
pub struct CachedModelSummary {
    /// The repository identifier (e.g., `"RWKV/RWKV7-Goose-World3-1.5B-HF"`).
    pub repo_id: String,
    /// Number of files in the snapshot directory.
    pub file_count: usize,
    /// Total size on disk in bytes.
    pub total_size: u64,
    /// Whether there are incomplete `.chunked.part` temp files.
    pub has_partial: bool,
    /// Most recent modification time among files in the snapshot directory.
    ///
    /// `None` if no files were found or all metadata reads failed.
    pub last_modified: Option<std::time::SystemTime>,
}

/// Scans the entire HF cache and returns a summary for each cached model.
///
/// This is a local-only operation (no API calls). It lists all `models--*`
/// directories and counts files + sizes in each snapshot.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the cache directory cannot be read.
pub fn cache_summary() -> Result<Vec<CachedModelSummary>, FetchError> {
    let cache_dir = hf_cache_dir()?;

    if !cache_dir.exists() {
        return Ok(Vec::new());
    }

    let entries = std::fs::read_dir(&cache_dir).map_err(|e| FetchError::Io {
        path: cache_dir.clone(),
        source: e,
    })?;

    let mut summaries: Vec<CachedModelSummary> = Vec::new();

    for entry in entries {
        let Ok(entry) = entry else { continue };
        let dir_name = entry.file_name();
        // BORROW: explicit .to_string_lossy() for OsString → str conversion
        let dir_str = dir_name.to_string_lossy();

        let Some(repo_id) = repo_id_from_folder_name(&dir_str) else {
            continue;
        };

        let repo_dir = entry.path();

        // Count files and total size in snapshots.
        let (file_count, total_size, last_modified) = count_snapshot_files(&repo_dir);

        // Check for partial downloads.
        let has_partial = find_partial_blob_size(&crate::cache_layout::blobs_dir(&repo_dir)) > 0;

        summaries.push(CachedModelSummary {
            repo_id,
            file_count,
            total_size,
            has_partial,
            last_modified,
        });
    }

    summaries.sort_by(|a, b| a.repo_id.cmp(&b.repo_id));

    Ok(summaries)
}

/// Returns the file count and total size for a single cached repo.
///
/// Avoids scanning the entire cache when only one repo's metrics are needed
/// (e.g., for the `cache delete` preview).
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the cache directory cannot be determined.
pub fn repo_disk_usage(repo_id: &str) -> Result<(usize, u64), FetchError> {
    let cache_dir = hf_cache_dir()?;
    let repo_dir = crate::cache_layout::repo_dir(&cache_dir, repo_id);
    let (file_count, total_size, _) = count_snapshot_files(&repo_dir);
    Ok((file_count, total_size))
}

/// Checks whether a single cached repo has `.chunked.part` temp files.
///
/// Avoids scanning the entire cache when only one repo's partial status
/// is needed (e.g., for the `du <REPO>` partial-download hint).
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the cache directory cannot be determined.
pub fn repo_has_partial(repo_id: &str) -> Result<bool, FetchError> {
    let cache_dir = hf_cache_dir()?;
    let repo_dir = crate::cache_layout::repo_dir(&cache_dir, repo_id);
    let blobs_dir = crate::cache_layout::blobs_dir(&repo_dir);
    Ok(find_partial_blob_size(&blobs_dir) > 0)
}

/// Counts files, total size, and most recent modification time across all
/// snapshot directories for a repo.
fn count_snapshot_files(repo_dir: &Path) -> (usize, u64, Option<std::time::SystemTime>) {
    let snapshots_dir = crate::cache_layout::snapshots_dir(repo_dir);
    let Ok(snapshots) = std::fs::read_dir(snapshots_dir) else {
        return (0, 0, None);
    };

    let mut file_count: usize = 0;
    let mut total_size: u64 = 0;
    let mut latest: Option<std::time::SystemTime> = None;

    for snap_entry in snapshots {
        let Ok(snap_entry) = snap_entry else { continue };
        let snap_path = snap_entry.path();
        if !snap_path.is_dir() {
            continue;
        }
        count_files_recursive(&snap_path, &mut file_count, &mut total_size, &mut latest);
    }

    (file_count, total_size, latest)
}

/// Recursively counts files, accumulates sizes, and tracks the most recent
/// modification time in a directory.
fn count_files_recursive(
    dir: &Path,
    count: &mut usize,
    total: &mut u64,
    latest: &mut Option<std::time::SystemTime>,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if path.is_dir() {
            count_files_recursive(&path, count, total, latest);
        } else if let Ok(meta) = entry.metadata() {
            *count += 1;
            *total += meta.len();
            if let Ok(modified) = meta.modified() {
                match *latest {
                    Some(current) if modified <= current => {} // EXPLICIT: current mtime is more recent, keep it
                    _ => *latest = Some(modified),
                }
            }
        } else {
            *count += 1;
        }
    }
}

/// Reads the commit hash from a refs file, if it exists.
///
/// Looks for `<repo_dir>/refs/<revision>` and returns the trimmed contents
/// (a commit hash) or `None` if the file does not exist or is empty.
#[must_use]
pub fn read_ref(repo_dir: &Path, revision: &str) -> Option<String> {
    let ref_path = crate::cache_layout::ref_path(repo_dir, revision);
    std::fs::read_to_string(ref_path)
        .ok()
        // BORROW: explicit .to_owned() to convert trimmed &str → owned String
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
}

/// Checks whether any `.chunked.part` temp file exists in the blobs directory.
///
/// This is a repo-level heuristic: it cannot map a specific filename to its
/// blob without full LFS metadata, so it checks for any `.chunked.part` file.
/// A `true` result means *some* file in the repo has a partial download.
fn has_partial_blob(blobs_dir: &Path) -> bool {
    find_partial_blob_size(blobs_dir) > 0
}

/// Returns the size of the first `.chunked.part` file found in the blobs directory.
fn find_partial_blob_size(blobs_dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(blobs_dir) else {
        return 0;
    };

    for entry in entries {
        let Ok(entry) = entry else { continue };
        let name = entry.file_name();
        // BORROW: explicit .to_string_lossy() for OsString → str conversion
        if name.to_string_lossy().ends_with(".chunked.part") {
            return entry.metadata().map_or(0, |m| m.len());
        }
    }

    0
}

/// A `.chunked.part` temp file left by an interrupted chunked download.
#[derive(Debug, Clone)]
pub struct PartialFile {
    /// The repository identifier (e.g., `"meta-llama/Llama-3.2-1B"`).
    pub repo_id: String,
    /// The `.chunked.part` filename (e.g., `"abc123def456.chunked.part"`).
    pub filename: String,
    /// Absolute path to the `.chunked.part` file.
    pub path: PathBuf,
    /// Size of the partial file in bytes.
    pub size: u64,
}

impl PartialFile {
    /// Returns sibling sidecar paths that should be removed alongside this
    /// partial: the resume-state sidecar `{etag}.chunked.part.state` and
    /// any orphan write-tmp `{etag}.chunked.part.state.tmp` left by an
    /// interrupted atomic save.
    ///
    /// The paths are returned even when the underlying files do not exist
    /// — callers (`run_cache_clean_partial`) attempt removal best-effort.
    #[must_use]
    pub fn sidecar_paths(&self) -> Vec<PathBuf> {
        let Some(parent) = self.path.parent() else {
            return Vec::new();
        };
        // String concat (mirrors `cache_layout::temp_state_path`'s
        // rationale): the etag may itself contain periods, so
        // `Path::with_extension` would truncate at the wrong boundary.
        // BORROW: explicit .clone() for owned String → mutated copy
        let mut state_name = self.filename.clone();
        state_name.push_str(".state");
        // BORROW: explicit .clone() for owned String → mutated copy
        let mut tmp_name = self.filename.clone();
        tmp_name.push_str(".state.tmp");
        vec![parent.join(state_name), parent.join(tmp_name)]
    }
}

/// Finds all `.chunked.part` temp files in the `HuggingFace` cache.
///
/// Walks `models--*/blobs/` directories and collects partial files.
/// When `repo_filter` is `Some`, only the matching repo is scanned.
///
/// Returns an empty `Vec` if the cache directory does not exist.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the cache directory cannot be read.
pub fn find_partial_files(repo_filter: Option<&str>) -> Result<Vec<PartialFile>, FetchError> {
    let cache_dir = hf_cache_dir()?;

    if !cache_dir.exists() {
        return Ok(Vec::new());
    }

    let entries = std::fs::read_dir(&cache_dir).map_err(|e| FetchError::Io {
        // BORROW: explicit .clone() for owned PathBuf
        path: cache_dir.clone(),
        source: e,
    })?;

    let mut partials: Vec<PartialFile> = Vec::new();

    for entry in entries {
        let Ok(entry) = entry else { continue };
        let dir_name = entry.file_name();
        // BORROW: explicit .to_string_lossy() for OsString → str conversion
        let dir_str = dir_name.to_string_lossy();

        let Some(repo_id) = repo_id_from_folder_name(&dir_str) else {
            continue;
        };

        // Skip repos that don't match the filter.
        // BORROW: explicit .as_str() instead of Deref coercion
        if let Some(filter) = repo_filter {
            if repo_id.as_str() != filter {
                continue;
            }
        }

        let blobs_dir = crate::cache_layout::blobs_dir(&entry.path());
        let Ok(blob_entries) = std::fs::read_dir(&blobs_dir) else {
            continue;
        };

        for blob_entry in blob_entries {
            let Ok(blob_entry) = blob_entry else { continue };
            let name = blob_entry.file_name();
            // BORROW: explicit .to_string_lossy() for OsString → str conversion
            let name_str = name.to_string_lossy();
            if name_str.ends_with(".chunked.part") {
                let size = blob_entry.metadata().map_or(0, |m| m.len());
                partials.push(PartialFile {
                    // BORROW: explicit .clone() for owned String
                    repo_id: repo_id.clone(),
                    // BORROW: explicit .to_string() for Cow<str> → owned String
                    filename: name_str.to_string(),
                    path: blob_entry.path(),
                    size,
                });
            }
        }
    }

    Ok(partials)
}

/// Per-file disk usage entry within a cached repository.
#[derive(Debug, Clone)]
pub struct CacheFileUsage {
    /// Filename relative to the snapshot directory.
    pub filename: String,
    /// File size in bytes.
    pub size: u64,
}

/// Returns per-file disk usage for a specific cached repository.
///
/// Walks the snapshot directories under
/// `<cache_dir>/models--<org>--<name>/snapshots/` and collects each file's
/// relative path and size. Results are sorted by size descending.
///
/// Returns an empty `Vec` if the repository is not cached.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the cache directory cannot be determined.
pub fn cache_repo_usage(repo_id: &str) -> Result<Vec<CacheFileUsage>, FetchError> {
    let cache_dir = hf_cache_dir()?;
    let repo_dir = crate::cache_layout::repo_dir(&cache_dir, repo_id);

    if !repo_dir.exists() {
        return Ok(Vec::new());
    }

    let snapshots_dir = crate::cache_layout::snapshots_dir(&repo_dir);
    let Ok(snapshots) = std::fs::read_dir(&snapshots_dir) else {
        return Ok(Vec::new());
    };

    let mut files: Vec<CacheFileUsage> = Vec::new();

    for snap_entry in snapshots {
        let Ok(snap_entry) = snap_entry else { continue };
        let snap_path = snap_entry.path();
        if !snap_path.is_dir() {
            continue;
        }
        collect_snapshot_files(&snap_path, "", &mut files);
    }

    files.sort_by_key(|f| std::cmp::Reverse(f.size));

    Ok(files)
}

/// Recursively collects files from a snapshot directory into `CacheFileUsage` entries.
///
/// The `prefix` parameter tracks the relative path from the snapshot root,
/// so that files in subdirectories get paths like `"tokenizer/vocab.json"`.
fn collect_snapshot_files(dir: &Path, prefix: &str, files: &mut Vec<CacheFileUsage>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        // BORROW: explicit .to_string_lossy() for OsString → str conversion
        let name = entry.file_name().to_string_lossy().to_string();

        if path.is_dir() {
            let child_prefix = if prefix.is_empty() {
                name
            } else {
                format!("{prefix}/{name}")
            };
            collect_snapshot_files(&path, &child_prefix, files);
        } else {
            let filename = if prefix.is_empty() {
                name
            } else {
                format!("{prefix}/{name}")
            };
            let size = entry.metadata().map_or(0, |m| m.len());
            files.push(CacheFileUsage { filename, size });
        }
    }
}

/// Verification status for a single cached file.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum VerifyStatus {
    /// Local `SHA256` matches the expected hash from `HuggingFace` LFS metadata.
    Ok,
    /// Local `SHA256` does not match the expected hash — the cached file is
    /// corrupted (bit rot, interrupted write, or upstream blob changed).
    Mismatch {
        /// Expected `SHA256` hex digest from `HuggingFace` LFS metadata.
        expected: String,
        /// Actual `SHA256` hex digest computed from the local file.
        actual: String,
    },
    /// File has no LFS metadata (small git-stored file); verification skipped.
    Skipped,
    /// File is absent from the local snapshot directory.
    Missing,
}

/// Result of verifying a single cached file against `HuggingFace` LFS metadata.
#[derive(Debug, Clone)]
pub struct FileVerification {
    /// Filename within the repository.
    pub filename: String,
    /// File size in bytes — local size when the file is present, otherwise
    /// the expected size from the API (or `0` when neither is known).
    pub size: u64,
    /// Verification result.
    pub status: VerifyStatus,
}

/// Streaming progress event emitted by [`verify_cache_with_progress`] so
/// callers can render per-file feedback during a long verification.
///
/// Events fire in this order:
/// 1. [`VerifyEvent::Started`] — once, after the metadata fetch completes,
///    before any per-file work begins. Carries the total file count and a
///    pre-computed maximum filename length so callers can size display
///    columns up-front.
/// 2. For each file in alphabetical order:
///    - [`VerifyEvent::FileStart`] — before the per-file `SHA256`
///      computation kicks in.
///    - [`VerifyEvent::FileComplete`] — when the per-file result is known,
///      carrying the [`VerifyStatus`] outcome.
#[non_exhaustive]
#[derive(Debug)]
pub enum VerifyEvent<'a> {
    /// Fired once at the start of the run with summary stats useful for
    /// laying out a streamed table or progress display.
    Started {
        /// Total number of files that will be verified.
        total: usize,
        /// Maximum filename length across the verification list.
        max_filename_len: usize,
    },
    /// A file is about to be verified.
    FileStart {
        /// 1-based index of this file in the verification list.
        index: usize,
        /// Total number of files in the verification list.
        total: usize,
        /// Filename within the repository.
        filename: &'a str,
        /// File size in bytes (local size when present, else expected size).
        size: u64,
        /// `true` when the file has LFS metadata (a real `SHA256` computation
        /// is about to run); `false` when the file is git-stored and will be
        /// skipped near-instantly.
        has_lfs: bool,
    },
    /// A file's verification has completed.
    FileComplete {
        /// 1-based index of this file in the verification list.
        index: usize,
        /// Total number of files in the verification list.
        total: usize,
        /// Filename within the repository.
        filename: &'a str,
        /// File size in bytes (matches the `size` from the corresponding
        /// [`VerifyEvent::FileStart`]).
        size: u64,
        /// The per-file verification result.
        status: &'a VerifyStatus,
    },
}

/// Verifies `SHA256` digests of cached files against `HuggingFace` LFS metadata.
///
/// Fetches the expected hashes from the `HuggingFace` API and, for each file
/// that has an LFS `SHA256`, reads the local cached file and compares.
///
/// Files without LFS metadata (small git-stored files such as `config.json`)
/// are reported as [`VerifyStatus::Skipped`]; files absent from the snapshot
/// directory are reported as [`VerifyStatus::Missing`]. Both are
/// non-failures — only [`VerifyStatus::Mismatch`] indicates a corrupted file.
///
/// `revision` defaults to `"main"` when `None`. Requires network access for
/// the metadata fetch; the per-file digest computation is local-only.
///
/// For long verifications (multi-GiB safetensors files), prefer
/// [`verify_cache_with_progress`] so a CLI / GUI can render a spinner or
/// progress bar while each file is hashed.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the `HuggingFace` API request fails.
/// Returns [`FetchError::Io`] when a local cached file is present but
/// cannot be read.
pub async fn verify_cache(
    repo_id: &str,
    token: Option<&str>,
    revision: Option<&str>,
) -> Result<Vec<FileVerification>, FetchError> {
    verify_cache_with_progress(repo_id, token, revision, |_| {}).await
}

/// Same as [`verify_cache`] but emits [`VerifyEvent`]s through `on_event`
/// so callers can render streaming progress (e.g. a spinner per file).
///
/// The callback runs on the same task as the verification — keep it short.
/// Use interior mutability ([`std::cell::Cell`], [`std::cell::RefCell`]) if
/// you need to track state across events; the closure may capture by shared
/// reference because the API requires only [`Fn`].
///
/// Files are processed in alphabetical order by filename so that streamed
/// output remains stable across runs and matches the sort order of the
/// returned [`Vec<FileVerification>`].
///
/// # Errors
///
/// Same error conditions as [`verify_cache`].
pub async fn verify_cache_with_progress<F>(
    repo_id: &str,
    token: Option<&str>,
    revision: Option<&str>,
    on_event: F,
) -> Result<Vec<FileVerification>, FetchError>
where
    F: Fn(VerifyEvent<'_>),
{
    let revision = revision.unwrap_or("main");
    let cache_dir = hf_cache_dir()?;
    let repo_dir = crate::cache_layout::repo_dir(&cache_dir, repo_id);

    let commit_hash = read_ref(&repo_dir, revision);

    let client = crate::chunked::build_client(token)?;
    let mut remote_files =
        crate::repo::list_repo_files_with_metadata(repo_id, token, Some(revision), &client).await?;

    // Sort up-front so streamed output is stable across runs and matches the
    // returned `Vec<FileVerification>`'s order.
    remote_files.sort_by(|a, b| a.filename.cmp(&b.filename));

    // BORROW: explicit .as_deref() for Option<String> → Option<&str>
    let snapshot_dir = commit_hash
        .as_deref()
        .map(|hash| crate::cache_layout::snapshot_dir(&repo_dir, hash));

    let total = remote_files.len();
    let max_filename_len = remote_files
        .iter()
        .map(|f| f.filename.len())
        .max()
        .unwrap_or(0);

    on_event(VerifyEvent::Started {
        total,
        max_filename_len,
    });

    let mut results: Vec<FileVerification> = Vec::with_capacity(total);

    for (i, remote) in remote_files.iter().enumerate() {
        let index = i + 1;
        let local_path = snapshot_dir
            .as_ref()
            // BORROW: explicit .as_str() for path construction
            .map(|dir| dir.join(remote.filename.as_str()));

        let exists = local_path.as_ref().is_some_and(|p| p.exists());
        let local_size = local_path
            .as_ref()
            .filter(|_| exists)
            .and_then(|p| std::fs::metadata(p).ok().map(|m| m.len()))
            .unwrap_or(0);
        let expected_size = remote.size.unwrap_or(0);
        let display_size = if exists { local_size } else { expected_size };

        let has_lfs = remote.sha256.is_some();
        on_event(VerifyEvent::FileStart {
            index,
            total,
            // BORROW: explicit .as_str() for &String → &str argument
            filename: remote.filename.as_str(),
            size: display_size,
            has_lfs,
        });

        let status = match (remote.sha256.as_deref(), local_path.as_deref(), exists) {
            (None, _, _) => VerifyStatus::Skipped,
            (Some(_), None, _) | (Some(_), Some(_), false) => VerifyStatus::Missing,
            (Some(expected), Some(path), true) => {
                // BORROW: explicit .as_str() for &String → &str argument
                match crate::checksum::verify_sha256(path, remote.filename.as_str(), expected).await
                {
                    Ok(()) => VerifyStatus::Ok,
                    Err(FetchError::Checksum {
                        expected, actual, ..
                    }) => VerifyStatus::Mismatch { expected, actual },
                    Err(e) => return Err(e),
                }
            }
        };

        on_event(VerifyEvent::FileComplete {
            index,
            total,
            // BORROW: explicit .as_str() for &String → &str argument
            filename: remote.filename.as_str(),
            size: display_size,
            status: &status,
        });

        results.push(FileVerification {
            // BORROW: explicit .clone() for owned String
            filename: remote.filename.clone(),
            size: display_size,
            status,
        });
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::panic,
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing
    )]

    use super::*;

    fn sample_partial(filename: &str) -> PartialFile {
        PartialFile {
            repo_id: "org/model".to_owned(),
            filename: filename.to_owned(),
            path: PathBuf::from("/tmp/models--org--model/blobs").join(filename),
            size: 1024,
        }
    }

    #[test]
    fn sidecar_paths_returns_state_and_state_tmp() {
        let p = sample_partial("abc123.chunked.part");
        let sidecars = p.sidecar_paths();

        assert_eq!(sidecars.len(), 2);
        assert_eq!(
            sidecars[0],
            PathBuf::from("/tmp/models--org--model/blobs/abc123.chunked.part.state")
        );
        assert_eq!(
            sidecars[1],
            PathBuf::from("/tmp/models--org--model/blobs/abc123.chunked.part.state.tmp")
        );
    }

    #[test]
    fn sidecar_paths_handles_etag_with_periods() {
        // Same period-handling rationale as `cache_layout::temp_state_path`:
        // the etag may itself contain dots, so naive `Path::with_extension`
        // would chop at the wrong boundary.
        let p = sample_partial("abc.def.chunked.part");
        let sidecars = p.sidecar_paths();

        assert_eq!(
            sidecars[0],
            PathBuf::from("/tmp/models--org--model/blobs/abc.def.chunked.part.state")
        );
        assert_eq!(
            sidecars[1],
            PathBuf::from("/tmp/models--org--model/blobs/abc.def.chunked.part.state.tmp")
        );
    }

    #[test]
    fn snapshot_roundtrip_write_then_read_returns_equal_value() {
        // Use a freshly-created temp dir so the test is isolated from any
        // pre-existing cache state on the developer machine / CI runner.
        let tmp =
            std::env::temp_dir().join(format!("hf-fm-snapshot-roundtrip-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).expect("create temp dir");

        let original = Snapshot {
            version: SNAPSHOT_VERSION,
            revision: "main".to_owned(),
            preset: Some("safetensors".to_owned()),
            filter: vec!["*.json".to_owned()],
            exclude: vec!["*.md".to_owned()],
        };

        write_snapshot(&tmp, &original).expect("write_snapshot");
        let round_tripped = read_snapshot(&tmp)
            .expect("read_snapshot")
            .expect("snapshot present");

        assert_eq!(round_tripped, original);

        // Cleanup
        let _ = std::fs::remove_file(snapshot_path(&tmp));
        let _ = std::fs::remove_dir(&tmp);
    }

    #[test]
    fn snapshot_read_returns_none_when_absent() {
        let tmp =
            std::env::temp_dir().join(format!("hf-fm-snapshot-absent-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).expect("create temp dir");

        let result = read_snapshot(&tmp).expect("read_snapshot");
        assert!(result.is_none(), "expected None for absent sidecar");

        let _ = std::fs::remove_dir(&tmp);
    }
}
