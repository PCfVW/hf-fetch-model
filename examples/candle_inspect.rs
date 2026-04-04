// SPDX-License-Identifier: MIT OR Apache-2.0

//! Inspect a model's tensor layout before downloading weights.
//!
//! Shows tensor names, shapes, and dtypes from `.safetensors` headers
//! using HTTP Range requests — no weight data downloaded.
//!
//! Run: `cargo run --example candle_inspect`
//! With a specific model: `cargo run --example candle_inspect -- google/gemma-2-2b`

use hf_fetch_model::inspect;

#[tokio::main]
async fn main() -> Result<(), hf_fetch_model::FetchError> {
    let repo_id = std::env::args()
        .nth(1)
        // BORROW: explicit .to_owned() for &str → owned String
        .unwrap_or_else(|| "EleutherAI/pythia-1.4b".to_owned());

    println!("Inspecting {repo_id}...\n");

    // Read tensor metadata — only 2 HTTP requests per file, no weight download.
    let results = inspect::inspect_repo_safetensors(&repo_id, None, None).await?;

    let mut total_params: u64 = 0;
    for (filename, header, _source) in &results {
        println!("{filename}:");
        for t in &header.tensors {
            let params = t.num_elements();
            total_params = total_params.saturating_add(params);
            println!("  {:<60} {:>5}  {:?}", t.name, t.dtype, t.shape);
        }
        println!();
    }

    // Summary.
    let total_tensors: usize = results.iter().map(|(_, h, _)| h.tensors.len()).sum();
    println!("{total_tensors} tensors, {total_params} parameters");

    // To load into candle after inspecting:
    //
    //   let config = hf_fetch_model::Filter::safetensors().build()?;
    //   let outcome = hf_fetch_model::download_with_config(
    //       repo_id, &config,
    //   ).await?;
    //   let snapshot_dir = outcome.into_inner();
    //   // Pass safetensors paths to candle:
    //   //   candle_core::safetensors::load(&path, &device)?

    Ok(())
}
