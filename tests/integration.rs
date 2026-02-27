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

    let path = result.unwrap_or_else(|e| panic!("download failed: {e}"));

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
