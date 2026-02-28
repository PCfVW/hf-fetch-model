// SPDX-License-Identifier: MIT OR Apache-2.0

//! `HuggingFace` cache directory resolution and local model family scanning.
//!
//! [`hf_cache_dir()`] locates the local HF cache. [`list_cached_families()`]
//! scans downloaded models and groups them by `model_type`.

use std::collections::BTreeMap;
use std::path::PathBuf;

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
