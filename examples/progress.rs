// SPDX-License-Identifier: MIT OR Apache-2.0

//! Download with indicatif progress bars.
//!
//! Run: `cargo run --example progress --features indicatif`

#![allow(clippy::unwrap_used, clippy::expect_used)]

use hf_fetch_model::progress::IndicatifProgress;
use hf_fetch_model::FetchConfig;

#[tokio::main]
async fn main() {
    let progress = IndicatifProgress::new();

    let config = FetchConfig::builder()
        .on_progress(move |e| progress.handle(e))
        .build()
        .expect("config build failed");

    let path = hf_fetch_model::download_with_config("julien-c/dummy-unknown".to_owned(), &config)
        .await
        .expect("download failed");

    println!("Downloaded to: {}", path.display());
}
