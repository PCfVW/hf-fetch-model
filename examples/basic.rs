// SPDX-License-Identifier: MIT OR Apache-2.0

//! Minimal download example.
//!
//! Run: `cargo run --example basic`

#[tokio::main]
async fn main() -> Result<(), hf_fetch_model::FetchError> {
    let path = hf_fetch_model::download("julien-c/dummy-unknown".to_owned()).await?;

    println!("Downloaded to: {}", path.display());
    Ok(())
}
