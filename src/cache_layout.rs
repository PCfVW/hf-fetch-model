// SPDX-License-Identifier: MIT OR Apache-2.0

//! Centralized `hf-hub` cache path construction.
//!
//! All paths follow the `hf-hub` 0.5 cache layout:
//! `{cache_root}/models--org--name/{snapshots,blobs,refs}/...`
//!
//! This module is the single source of truth for cache directory structure.
//! When `hf-hub` bumps, update this module and rerun the
//! `cache_layout_matches_hf_hub` integration test.

use std::path::{Path, PathBuf};

use hf_hub::{Repo, RepoType};

/// Repo folder name: `"models--org--name"`.
///
/// Delegates to [`hf_hub::Repo::folder_name()`].
#[must_use]
pub fn repo_folder_name(repo_id: &str) -> String {
    // BORROW: explicit .to_owned() for &str → owned String required by Repo::new
    Repo::new(repo_id.to_owned(), RepoType::Model).folder_name()
}

/// Repo root directory: `{cache_root}/models--org--name/`.
#[must_use]
pub fn repo_dir(cache_root: &Path, repo_id: &str) -> PathBuf {
    cache_root.join(repo_folder_name(repo_id))
}

/// Snapshots directory: `{repo_dir}/snapshots/`.
#[must_use]
pub fn snapshots_dir(repo_dir: &Path) -> PathBuf {
    repo_dir.join("snapshots")
}

/// Snapshot directory for a specific commit: `{repo_dir}/snapshots/{commit_hash}/`.
#[must_use]
pub fn snapshot_dir(repo_dir: &Path, commit_hash: &str) -> PathBuf {
    snapshots_dir(repo_dir).join(commit_hash)
}

/// Pointer path: `{repo_dir}/snapshots/{commit_hash}/{filename}`.
#[must_use]
pub fn pointer_path(repo_dir: &Path, commit_hash: &str, filename: &str) -> PathBuf {
    snapshot_dir(repo_dir, commit_hash).join(filename)
}

/// Blobs directory: `{repo_dir}/blobs/`.
#[must_use]
pub fn blobs_dir(repo_dir: &Path) -> PathBuf {
    repo_dir.join("blobs")
}

/// Blob path: `{repo_dir}/blobs/{etag}`.
#[must_use]
pub fn blob_path(repo_dir: &Path, etag: &str) -> PathBuf {
    blobs_dir(repo_dir).join(etag)
}

/// Temp blob path for chunked downloads: `{repo_dir}/blobs/{etag}.chunked.part`.
///
/// Uses string concatenation rather than [`Path::with_extension`] to handle
/// etags containing periods (e.g., `"abc.def"` → `"abc.def.chunked.part"`,
/// not `"abc.chunked.part"`).
#[must_use]
pub fn temp_blob_path(repo_dir: &Path, etag: &str) -> PathBuf {
    // BORROW: explicit .to_owned() for &str → owned String for path concatenation
    let mut name = etag.to_owned();
    name.push_str(".chunked.part");
    blobs_dir(repo_dir).join(name)
}

/// Resume-state sidecar path for chunked downloads:
/// `{repo_dir}/blobs/{etag}.chunked.part.state`.
///
/// Lives next to the [`temp_blob_path`] partial and tracks per-chunk
/// completion offsets so that an interrupted download can resume on the
/// next invocation. Cleaned up on successful finalization, kept alongside
/// the partial when the download is interrupted.
///
/// Same period-handling rationale as [`temp_blob_path`]: explicit string
/// concatenation rather than [`Path::with_extension`] so that etags
/// containing periods round-trip correctly.
#[must_use]
pub fn temp_state_path(repo_dir: &Path, etag: &str) -> PathBuf {
    // BORROW: explicit .to_owned() for &str → owned String for path concatenation
    let mut name = etag.to_owned();
    name.push_str(".chunked.part.state");
    blobs_dir(repo_dir).join(name)
}

/// Refs directory: `{repo_dir}/refs/`.
#[must_use]
pub fn refs_dir(repo_dir: &Path) -> PathBuf {
    repo_dir.join("refs")
}

/// Ref file path: `{repo_dir}/refs/{revision}`.
#[must_use]
pub fn ref_path(repo_dir: &Path, revision: &str) -> PathBuf {
    refs_dir(repo_dir).join(revision)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn blob_path_joins_repo_dir_and_etag() {
        let rd = Path::new("/tmp/models--x--y");
        assert_eq!(blob_path(rd, "abc123"), rd.join("blobs").join("abc123"));
    }

    #[test]
    fn temp_blob_path_preserves_periods_in_etag() {
        // Guards the specific behaviour called out in the `temp_blob_path`
        // doc comment: etags containing periods must not be truncated by
        // `Path::with_extension`. `"abc.def"` + `".chunked.part"` →
        // `"abc.def.chunked.part"`, NOT `"abc.chunked.part"`.
        let rd = Path::new("/tmp/models--x--y");
        assert_eq!(
            temp_blob_path(rd, "abc.def"),
            rd.join("blobs").join("abc.def.chunked.part")
        );
    }

    #[test]
    fn temp_state_path_lives_next_to_temp_blob() {
        let rd = Path::new("/tmp/models--x--y");
        let etag = "abc123";
        assert_eq!(
            temp_state_path(rd, etag),
            rd.join("blobs").join("abc123.chunked.part.state")
        );
        // The two helpers must agree on the directory so the sidecar
        // always sits right next to its partial.
        assert_eq!(
            temp_blob_path(rd, etag).parent(),
            temp_state_path(rd, etag).parent()
        );
    }

    #[test]
    fn temp_state_path_preserves_periods_in_etag() {
        let rd = Path::new("/tmp/models--x--y");
        assert_eq!(
            temp_state_path(rd, "abc.def"),
            rd.join("blobs").join("abc.def.chunked.part.state")
        );
    }
}
