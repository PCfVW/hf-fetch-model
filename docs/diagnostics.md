# Download Diagnostics

hf-fetch-model emits structured `tracing` events at `debug` level to help diagnose download performance.

## CLI

Use the `--verbose` / `-v` flag:

```sh
hf-fm google/gemma-2-2b-it -v
```

## Library

Initialize a `tracing-subscriber` at `debug` level:

```rust
// Via RUST_LOG environment variable
// RUST_LOG=hf_fetch_model=debug cargo run

// Or programmatically
tracing_subscriber::fmt()
    .with_env_filter("hf_fetch_model=debug")
    .init();
```

## Example output

```
DEBUG hf_fetch_model: listing repository files repo_id="allenai/OLMo-1B-hf"
DEBUG hf_fetch_model: metadata fetch succeeded files_with_size=8 total_files=8
DEBUG hf_fetch_model: merged plan-recommended settings concurrency=4 connections_per_file=8 chunk_threshold=104857600
DEBUG hf_fetch_model: download settings (after plan optimization) total_files=8 concurrency=4
DEBUG hf_fetch_model: chunked download (multi-connection) filename="model.safetensors" size_mib=2475 connections=8
DEBUG hf_fetch_model: single-connection download (below chunk threshold) filename="config.json" size_mib=0
DEBUG hf_fetch_model: download complete filename="model.safetensors" elapsed_secs="23.1" throughput_mbps="857.2"
DEBUG hf_fetch_model: download complete files_downloaded=8 files_failed=0 total_elapsed_secs="24.3"
```

## What to look for

| Message | Level | Meaning |
|---------|-------|---------|
| "metadata fetch failed" | warn | File sizes unknown — chunked downloads disabled, all files use single connection |
| "single-connection download" with reason "file size unknown" | debug | Metadata was not available for this file |
| "chunked download (multi-connection)" | debug | File exceeds `chunk_threshold`, using parallel HTTP Range connections |
| "download complete" (per-file) | debug | Includes `elapsed_secs` and `throughput_mbps` — useful for comparing single vs chunked |
| "download complete" (summary) | debug | Final tally: files downloaded, files failed, total elapsed time |
