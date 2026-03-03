// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration tests for single-file download API (`download_file`, `download_file_blocking`).

#![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]

use hf_fetch_model::{FetchConfig, FetchError};

#[tokio::test]
async fn download_single_file_async() {
    let config = FetchConfig::builder().build().unwrap();

    let path =
        hf_fetch_model::download_file("julien-c/dummy-unknown".to_owned(), "config.json", &config)
            .await
            .unwrap();

    assert!(path.exists(), "downloaded file should exist on disk");
    assert!(path.is_file(), "path should be a file, not a directory");
    assert!(
        path.file_name().is_some_and(|n| n == "config.json"),
        "filename should be config.json"
    );
}

#[test]
fn download_single_file_blocking() {
    let config = FetchConfig::builder().build().unwrap();

    let path = hf_fetch_model::download_file_blocking(
        "julien-c/dummy-unknown".to_owned(),
        "config.json",
        &config,
    )
    .unwrap();

    assert!(path.exists(), "downloaded file should exist on disk");
    assert!(path.is_file(), "path should be a file, not a directory");
}

#[tokio::test]
async fn download_single_file_nonexistent_returns_error() {
    let config = FetchConfig::builder().build().unwrap();

    let result = hf_fetch_model::download_file(
        "julien-c/dummy-unknown".to_owned(),
        "this-file-does-not-exist.bin",
        &config,
    )
    .await;

    assert!(
        result.is_err(),
        "downloading a nonexistent file should fail"
    );
    match result {
        Err(FetchError::Api(_) | FetchError::Http(_)) => {}
        Err(other) => {
            // 404 or similar HTTP error is also acceptable.
            assert!(
                other.to_string().contains("404") || other.to_string().contains("not found"),
                "unexpected error: {other}"
            );
        }
        Ok(path) => panic!(
            "expected error for nonexistent file, got: {}",
            path.display()
        ),
    }
}
