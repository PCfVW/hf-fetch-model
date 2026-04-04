// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration tests for hf-fetch-model.
//!
//! These tests download small public repositories to verify end-to-end behavior.
//! They require network access and interact with the `HuggingFace` Hub.

#![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]

/// Downloads a small public model and verifies the returned path exists.
///
/// Uses `julien-c/dummy-unknown`, a tiny test repository on the `HuggingFace` Hub.
#[tokio::test]
async fn download_small_public_model() {
    let result = hf_fetch_model::download("julien-c/dummy-unknown".to_owned()).await;

    let path = result
        .unwrap_or_else(|e| panic!("download failed: {e}"))
        .into_inner();

    assert!(
        path.exists(),
        "cache directory should exist: {}",
        path.display()
    );
    assert!(
        path.is_dir(),
        "cache path should be a directory: {}",
        path.display()
    );
}

/// Verifies that our cache path construction matches what `hf-hub` actually writes.
///
/// Catches any drift between `repo_folder_name()`, `read_ref()`, and the
/// real on-disk layout produced by `hf-hub`.
#[tokio::test]
async fn cache_layout_matches_hf_hub() {
    let outcome = hf_fetch_model::download("julien-c/dummy-unknown".to_owned())
        .await
        .unwrap();
    let snapshot_dir = outcome.inner();

    // Our repo_folder_name() must match the actual directory.
    let cache_dir = hf_fetch_model::cache::hf_cache_dir().unwrap();
    let expected_folder = cache_dir.join("models--julien-c--dummy-unknown");
    assert!(expected_folder.exists(), "repo folder name mismatch");

    // refs/main must exist and contain a commit hash.
    let refs_main = expected_folder.join("refs").join("main");
    assert!(refs_main.exists(), "refs/main missing");
    let hash = std::fs::read_to_string(&refs_main).unwrap();
    assert!(!hash.trim().is_empty(), "refs/main is empty");

    // Snapshot dir must match refs/main.
    let expected_snapshot = expected_folder.join("snapshots").join(hash.trim());
    assert_eq!(snapshot_dir, &expected_snapshot, "snapshot path mismatch");

    // Files must exist in the snapshot directory.
    assert!(
        expected_snapshot.join("config.json").exists(),
        "config.json missing from snapshot"
    );
}
