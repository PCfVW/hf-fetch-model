// SPDX-License-Identifier: MIT OR Apache-2.0

//! `HuggingFace` cache directory resolution and local model family scanning.
//!
//! [`hf_cache_dir()`] locates the local HF cache. [`list_cached_families()`]
//! scans downloaded models and groups them by `model_type`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::error::FetchError;

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

/// Scans the local HF cache for downloaded models and groups them by `model_type`.
///
/// Looks for `config.json` files inside model snapshot directories:
/// `<cache>/models--<org>--<name>/snapshots/*/config.json`
///
/// Returns a map from `model_type` (e.g., `"llama"`) to a sorted list of
/// repository identifiers (e.g., `["meta-llama/Llama-3.2-1B"]`).
///
/// Models without a `model_type` field in their `config.json` are skipped.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the cache directory cannot be read.
pub fn list_cached_families() -> Result<BTreeMap<String, Vec<String>>, FetchError> {
    let cache_dir = hf_cache_dir()?;

    if !cache_dir.exists() {
        return Ok(BTreeMap::new());
    }

    let entries = std::fs::read_dir(&cache_dir).map_err(|e| FetchError::Io {
        path: cache_dir.clone(),
        source: e,
    })?;

    let mut families: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for entry in entries {
        let Ok(entry) = entry else { continue };

        let dir_name = entry.file_name();
        // BORROW: explicit .to_string_lossy() for OsString → str conversion
        let dir_str = dir_name.to_string_lossy();

        // Only process model directories (models--org--name)
        let Some(repo_part) = dir_str.strip_prefix("models--") else {
            continue;
        };

        // Reconstruct repo_id: replace first "--" with "/"
        let repo_id = match repo_part.find("--") {
            Some(pos) => {
                let (org, name_with_sep) = repo_part.split_at(pos);
                let name = name_with_sep.get(2..).unwrap_or_default();
                format!("{org}/{name}")
            }
            None => repo_part.to_string(),
        };

        // Find the newest snapshot's config.json
        let snapshots_dir = entry.path().join("snapshots");
        if !snapshots_dir.exists() {
            continue;
        }

        if let Some(model_type) = find_model_type_in_snapshots(&snapshots_dir) {
            families.entry(model_type).or_default().push(repo_id);
        }
    }

    // Sort repo lists within each family for stable output
    for repos in families.values_mut() {
        repos.sort();
    }

    Ok(families)
}

/// Searches snapshot directories for a `config.json` containing `model_type`.
///
/// Returns the first `model_type` value found, or `None`.
fn find_model_type_in_snapshots(snapshots_dir: &std::path::Path) -> Option<String> {
    let snapshots = std::fs::read_dir(snapshots_dir).ok()?;

    for snap_entry in snapshots {
        let Ok(snap_entry) = snap_entry else { continue };
        let config_path = snap_entry.path().join("config.json");

        if !config_path.exists() {
            continue;
        }

        if let Some(model_type) = extract_model_type(&config_path) {
            return Some(model_type);
        }
    }

    None
}

/// Reads a `config.json` file and extracts the `model_type` field.
fn extract_model_type(config_path: &std::path::Path) -> Option<String> {
    let contents = std::fs::read_to_string(config_path).ok()?;
    let value: serde_json::Value = serde_json::from_str(contents.as_str()).ok()?;
    // BORROW: explicit .as_str() on serde_json Value
    value.get("model_type")?.as_str().map(String::from)
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
}

/// Inspects the local cache for a repository and compares against the remote file list.
///
/// # Arguments
///
/// * `repo_id` — The repository identifier (e.g., `"RWKV/RWKV7-Goose-World3-1.5B-HF"`).
/// * `token` — Optional authentication token.
/// * `revision` — Optional revision (defaults to `"main"`).
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the API request fails.
/// Returns [`FetchError::Io`] if the cache directory cannot be read.
pub async fn repo_status(
    repo_id: &str,
    token: Option<&str>,
    revision: Option<&str>,
) -> Result<RepoStatus, FetchError> {
    let revision = revision.unwrap_or("main");
    let cache_dir = hf_cache_dir()?;
    let repo_folder = format!("models--{}", repo_id.replace('/', "--"));
    // BORROW: explicit .as_str() for path construction
    let repo_dir = cache_dir.join(repo_folder.as_str());

    // Read commit hash from refs file if available.
    let commit_hash = read_ref(&repo_dir, revision);

    // Fetch remote file list with sizes.
    let remote_files =
        crate::repo::list_repo_files_with_metadata(repo_id, token, Some(revision)).await?;

    // Determine snapshot directory.
    // BORROW: explicit .as_deref() for Option<String> → Option<&str>
    let snapshot_dir = commit_hash
        .as_deref()
        .map(|hash| repo_dir.join("snapshots").join(hash));

    // Also check for .chunked.part files in blobs directory.
    let blobs_dir = repo_dir.join("blobs");

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
                let local_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

                if expected_size > 0 && local_size < expected_size {
                    FileStatus::Partial {
                        local_size,
                        expected_size,
                    }
                } else {
                    FileStatus::Complete { local_size }
                }
            } else if has_partial_blob(&blobs_dir) {
                // Check blobs for .chunked.part temp files
                let part_size = find_partial_blob_size(&blobs_dir);
                FileStatus::Partial {
                    local_size: part_size,
                    expected_size,
                }
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

        let Some(repo_part) = dir_str.strip_prefix("models--") else {
            continue;
        };

        // Reconstruct repo_id: replace first "--" with "/"
        let repo_id = match repo_part.find("--") {
            Some(pos) => {
                let (org, name_with_sep) = repo_part.split_at(pos);
                let name = name_with_sep.get(2..).unwrap_or_default();
                format!("{org}/{name}")
            }
            None => repo_part.to_string(),
        };

        let repo_dir = entry.path();

        // Count files and total size in snapshots.
        let (file_count, total_size) = count_snapshot_files(&repo_dir);

        // Check for partial downloads.
        let has_partial = find_partial_blob_size(&repo_dir.join("blobs")) > 0;

        summaries.push(CachedModelSummary {
            repo_id,
            file_count,
            total_size,
            has_partial,
        });
    }

    summaries.sort_by(|a, b| a.repo_id.cmp(&b.repo_id));

    Ok(summaries)
}

/// Counts files and total size across all snapshot directories for a repo.
fn count_snapshot_files(repo_dir: &Path) -> (usize, u64) {
    let snapshots_dir = repo_dir.join("snapshots");
    let Ok(snapshots) = std::fs::read_dir(snapshots_dir) else {
        return (0, 0);
    };

    let mut file_count: usize = 0;
    let mut total_size: u64 = 0;

    for snap_entry in snapshots {
        let Ok(snap_entry) = snap_entry else { continue };
        let snap_path = snap_entry.path();
        if !snap_path.is_dir() {
            continue;
        }
        count_files_recursive(&snap_path, &mut file_count, &mut total_size);
    }

    (file_count, total_size)
}

/// Recursively counts files and accumulates sizes in a directory.
fn count_files_recursive(dir: &Path, count: &mut usize, total: &mut u64) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if path.is_dir() {
            count_files_recursive(&path, count, total);
        } else {
            *count += 1;
            *total += entry.metadata().map(|m| m.len()).unwrap_or(0);
        }
    }
}

/// Reads the commit hash from a refs file, if it exists.
pub(crate) fn read_ref(repo_dir: &Path, revision: &str) -> Option<String> {
    let ref_path = repo_dir.join("refs").join(revision);
    std::fs::read_to_string(ref_path)
        .ok()
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
            return entry.metadata().map(|m| m.len()).unwrap_or(0);
        }
    }

    0
}
