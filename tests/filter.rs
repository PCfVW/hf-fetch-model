// SPDX-License-Identifier: MIT OR Apache-2.0

//! Glob filtering tests for hf-fetch-model.

#![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]

use hf_fetch_model::{FetchConfig, Filter};

#[tokio::test]
async fn download_with_safetensors_filter() {
    let config = FetchConfig::builder()
        .filter("*.json")
        .filter("*.txt")
        .filter("*.md")
        .build()
        .unwrap();

    let path = hf_fetch_model::download_with_config("julien-c/dummy-unknown".to_owned(), &config)
        .await
        .unwrap()
        .into_inner();

    assert!(path.exists(), "cache directory should exist");
    assert!(path.is_dir(), "cache path should be a directory");
}

#[tokio::test]
async fn download_with_exclude_filter() {
    let config = FetchConfig::builder().exclude("*.bin").build().unwrap();

    let path = hf_fetch_model::download_with_config("julien-c/dummy-unknown".to_owned(), &config)
        .await
        .unwrap()
        .into_inner();

    assert!(path.exists());
}

#[test]
fn filter_presets_build_successfully() {
    Filter::safetensors().build().unwrap();
    Filter::gguf().build().unwrap();
    Filter::pth().build().unwrap();
    Filter::config_only().build().unwrap();
}

#[test]
fn invalid_glob_returns_error() {
    let result = FetchConfig::builder().filter("[invalid").build();
    assert!(result.is_err());
}

#[tokio::test]
async fn download_with_progress_callback() {
    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let events_clone = events.clone();

    let config = FetchConfig::builder()
        .on_progress(move |e| {
            events_clone.lock().unwrap().push(e.clone());
        })
        .build()
        .unwrap();

    let path = hf_fetch_model::download_with_config("julien-c/dummy-unknown".to_owned(), &config)
        .await
        .unwrap()
        .into_inner();

    assert!(path.exists());

    let captured = events.lock().unwrap();

    // If the model was already cached, no progress events fire (cache-first path).
    // If it was freshly downloaded, all events should report 100% (completed).
    for event in captured.iter() {
        assert!(
            (event.percent - 100.0).abs() < f64::EPSILON,
            "completed events should be 100%"
        );
    }
}
