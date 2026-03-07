// SPDX-License-Identifier: MIT OR Apache-2.0

//! Minimal download example.
//!
//! Run: `cargo run --example basic`

#[tokio::main]
async fn main() -> Result<(), hf_fetch_model::FetchError> {
    let outcome = hf_fetch_model::download("julien-c/dummy-unknown".to_owned()).await?;

    if outcome.is_cached() {
        println!("Cached at: {}", outcome.inner().display());
    } else {
        println!("Downloaded to: {}", outcome.inner().display());
    }
    Ok(())
}
