// SPDX-License-Identifier: MIT OR Apache-2.0

//! Download with indicatif progress bars.
//!
//! Run: `cargo run --example progress --features indicatif`

use hf_fetch_model::progress::IndicatifProgress;
use hf_fetch_model::FetchConfig;

#[tokio::main]
async fn main() -> Result<(), hf_fetch_model::FetchError> {
    let progress = IndicatifProgress::new();

    let config = FetchConfig::builder()
        .on_progress(move |e| progress.handle(e))
        .build()?;

    let outcome =
        hf_fetch_model::download_with_config("julien-c/dummy-unknown".to_owned(), &config).await?;

    if outcome.is_cached() {
        println!("Cached at: {}", outcome.inner().display());
    } else {
        println!("Downloaded to: {}", outcome.inner().display());
    }
    Ok(())
}
