// SPDX-License-Identifier: MIT OR Apache-2.0

//! Benchmark: hf-fetch-model vs. plain hf-hub sequential download.
//!
//! This is a manual benchmark — run it with:
//! ```sh
//! cargo run --example bench --release
//! ```
//!
//! Results are network-dependent and should be documented in README.md.

use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repo_id = "julien-c/dummy-unknown";

    // Benchmark 1: plain hf-hub (sequential, file-by-file)
    println!("=== Plain hf-hub (sequential) ===");
    let start = Instant::now();
    let api = hf_hub::api::tokio::ApiBuilder::new().high().build()?;
    let repo = api.model(repo_id.to_owned());
    let info = repo.info().await?;
    for sibling in &info.siblings {
        // BORROW: explicit .as_str() instead of Deref coercion
        let _path = repo.get(sibling.rfilename.as_str()).await?;
    }
    let hf_hub_duration = start.elapsed();
    println!("hf-hub sequential: {hf_hub_duration:?}");

    // Benchmark 2: hf-fetch-model (concurrent)
    println!("\n=== hf-fetch-model (concurrent) ===");
    let start = Instant::now();
    let _path = hf_fetch_model::download(repo_id.to_owned()).await?;
    let hfm_duration = start.elapsed();
    println!("hf-fetch-model:    {hfm_duration:?}");

    // Summary
    println!("\n=== Summary ===");
    println!("hf-hub:         {hf_hub_duration:?}");
    println!("hf-fetch-model: {hfm_duration:?}");
    if hf_hub_duration > hfm_duration {
        let speedup = hf_hub_duration.as_secs_f64() / hfm_duration.as_secs_f64();
        println!("Speedup: {speedup:.1}x");
    }

    Ok(())
}
