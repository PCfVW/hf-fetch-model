# Configuration

All download functions accept a `FetchConfig` built via the builder pattern.

## Quick start

```rust
use hf_fetch_model::{FetchConfig, Filter};

// Preset — safetensors + config files
let config = Filter::safetensors()
    .on_progress(|e| println!("{}: {:.1}%", e.filename, e.percent))
    .build()?;

let outcome = hf_fetch_model::download_with_config(
    "google/gemma-2-2b-it".to_owned(),
    &config,
).await?;
```

## Builder methods

| Method | Description | Default |
|--------|-------------|---------|
| `.revision(rev)` | Git revision (branch, tag, SHA) | `"main"` |
| `.token(tok)` | Auth token | — |
| `.token_from_env()` | Read `HF_TOKEN` env var | — |
| `.filter(glob)` | Include pattern (repeatable) | all files |
| `.exclude(glob)` | Exclude pattern (repeatable) | none |
| `.concurrency(n)` | Parallel file downloads | 4 |
| `.output_dir(path)` | Custom cache directory | HF default |
| `.timeout_per_file(dur)` | Per-file timeout | 300s |
| `.timeout_total(dur)` | Overall timeout | unlimited |
| `.max_retries(n)` | Retries per file | 3 |
| `.verify_checksums(bool)` | SHA256 verification | true |
| `.chunk_threshold(bytes)` | Min file size for multi-connection download | 100 MiB |
| `.connections_per_file(n)` | Parallel connections per large file | 8 |
| `.on_progress(closure)` | Progress callback | — |

## Filter presets

Three presets cover common patterns:

```rust
// Safetensors weights + JSON configs
let config = Filter::safetensors().build()?;

// GGUF quantized models
let config = Filter::gguf().build()?;

// Config files only (no weights)
let config = Filter::config_only().build()?;
```

Presets are starting points — chain additional builder calls to customize:

```rust
let config = Filter::safetensors()
    .revision("v1.0")
    .token_from_env()
    .concurrency(2)
    .build()?;
```

## Single-file download

```rust
use hf_fetch_model::FetchConfig;

let config = FetchConfig::builder()
    .on_progress(|e| println!("{}: {:.1}%", e.filename, e.percent))
    .build()?;

let outcome = hf_fetch_model::download_file(
    "mntss/clt-gemma-2-2b-426k".to_owned(),
    "W_enc_0.safetensors",
    &config,
).await?;
```

## Blocking wrappers

For non-async callers, every download function has a `_blocking` variant:

```rust
let outcome = hf_fetch_model::download_blocking(
    "google/gemma-2-2b-it".to_owned(),
)?;

let outcome = hf_fetch_model::download_file_blocking(
    "mntss/clt-gemma-2-2b-426k".to_owned(),
    "config.yaml",
    &config,
)?;
```

## Download plan (dry-run API)

Before downloading, you can compute a plan that shows what needs downloading vs. what is already cached:

```rust
use hf_fetch_model::{FetchConfig, Filter, download_plan};

let config = Filter::safetensors().build()?;
let plan = download_plan("google/gemma-2-2b-it", &config).await?;

println!("Total: {} bytes", plan.total_bytes);
println!("Cached: {} bytes", plan.cached_bytes);
println!("To download: {} bytes ({} files)",
    plan.download_bytes, plan.files_to_download());
```

The plan also computes an optimized config based on the file size distribution:

```rust
// Get recommended settings (concurrency, connections/file, chunk threshold)
let optimized = plan.recommended_config()?;

// Or customize from the recommended baseline
let custom = plan.recommended_config_builder()
    .concurrency(2)
    .build()?;

// Execute with the optimized config
let outcome = hf_fetch_model::download_with_plan(&plan, &optimized).await?;
```

When no explicit plan is used, `download_with_config()` internally computes a plan and applies recommended settings for any fields the caller did not explicitly set.

## Cache and download outcome

All download functions return `DownloadOutcome<T>`, which distinguishes cache hits from network downloads:

```rust
let outcome = hf_fetch_model::download(
    "google/gemma-2-2b-it".to_owned(),
).await?;

if outcome.is_cached() {
    println!("Resolved from local cache (no network).");
}

let path = outcome.into_inner();
```
