// SPDX-License-Identifier: MIT OR Apache-2.0

//! Phase 2 reliability tests: checksum verification, retry config, timeouts.

#![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]

use std::time::Duration;

use hf_fetch_model::{FetchConfig, FetchError};

#[tokio::test]
async fn download_with_checksum_verification() {
    let config = FetchConfig::builder()
        .filter("*.json")
        .filter("*.txt")
        .filter("*.md")
        .verify_checksums(true)
        .build()
        .unwrap();

    let path = hf_fetch_model::download_with_config("julien-c/dummy-unknown".to_owned(), &config)
        .await
        .unwrap();

    assert!(path.exists(), "cache directory should exist");
    assert!(path.is_dir(), "cache path should be a directory");
}

#[tokio::test]
async fn download_with_checksums_disabled() {
    let config = FetchConfig::builder()
        .filter("*.json")
        .filter("*.txt")
        .filter("*.md")
        .verify_checksums(false)
        .build()
        .unwrap();

    let path = hf_fetch_model::download_with_config("julien-c/dummy-unknown".to_owned(), &config)
        .await
        .unwrap();

    assert!(path.exists());
}

#[tokio::test]
async fn download_with_retry_config() {
    let config = FetchConfig::builder()
        .filter("*.json")
        .filter("*.txt")
        .max_retries(1)
        .build()
        .unwrap();

    let path = hf_fetch_model::download_with_config("julien-c/dummy-unknown".to_owned(), &config)
        .await
        .unwrap();

    assert!(path.exists());
}

#[tokio::test]
async fn download_with_per_file_timeout() {
    // Generous timeout — should succeed.
    let config = FetchConfig::builder()
        .filter("*.json")
        .filter("*.txt")
        .timeout_per_file(Duration::from_secs(60))
        .build()
        .unwrap();

    let path = hf_fetch_model::download_with_config("julien-c/dummy-unknown".to_owned(), &config)
        .await
        .unwrap();

    assert!(path.exists());
}

#[tokio::test]
async fn download_with_total_timeout() {
    // Generous timeout — should succeed.
    let config = FetchConfig::builder()
        .filter("*.json")
        .filter("*.txt")
        .timeout_total(Duration::from_secs(120))
        .build()
        .unwrap();

    let path = hf_fetch_model::download_with_config("julien-c/dummy-unknown".to_owned(), &config)
        .await
        .unwrap();

    assert!(path.exists());
}

#[tokio::test]
async fn nonexistent_repo_returns_error() {
    let result =
        hf_fetch_model::download("this-repo-definitely-does-not-exist/nowhere".to_owned()).await;

    assert!(result.is_err());
    match result {
        Err(FetchError::RepoNotFound { .. }) => {}
        Err(other) => {
            // API errors are also acceptable (404 wrapped differently).
            assert!(
                other.to_string().contains("404") || matches!(other, FetchError::Api(_)),
                "unexpected error: {other}"
            );
        }
        Ok(_) => panic!("expected error for nonexistent repo"),
    }
}

#[tokio::test]
async fn download_files_returns_file_map() {
    let config = FetchConfig::builder()
        .filter("*.json")
        .filter("*.txt")
        .filter("*.md")
        .build()
        .unwrap();

    let files =
        hf_fetch_model::download_files_with_config("julien-c/dummy-unknown".to_owned(), &config)
            .await
            .unwrap();

    // The map should contain at least config.json (present in all HF repos).
    assert!(!files.is_empty(), "file map should not be empty");

    // Verify all returned paths exist on disk.
    for (filename, path) in &files {
        assert!(
            path.exists(),
            "file {filename} should exist at {}",
            path.display()
        );
    }
}

#[test]
fn download_files_blocking_returns_file_map() {
    let config = FetchConfig::builder()
        .filter("*.json")
        .filter("*.txt")
        .filter("*.md")
        .build()
        .unwrap();

    let files = hf_fetch_model::download_files_with_config_blocking(
        "julien-c/dummy-unknown".to_owned(),
        &config,
    )
    .unwrap();

    assert!(!files.is_empty(), "file map should not be empty");

    for (filename, path) in &files {
        assert!(
            path.exists(),
            "file {filename} should exist at {}",
            path.display()
        );
    }
}

#[test]
fn config_builder_defaults() {
    let config = FetchConfig::builder().build().unwrap();
    // Verify Phase 2 defaults are set correctly.
    let debug = format!("{config:?}");
    assert!(debug.contains("max_retries: 3"));
    assert!(debug.contains("verify_checksums: true"));
}
