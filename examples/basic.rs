// SPDX-License-Identifier: MIT OR Apache-2.0

//! Minimal download example.
//!
//! Run: `cargo run --example basic`

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[tokio::main]
async fn main() {
    let path = hf_fetch_model::download("julien-c/dummy-unknown".to_owned())
        .await
        .expect("download failed");

    println!("Downloaded to: {}", path.display());
}
