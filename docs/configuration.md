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
| `.concurrency(n)` | Parallel file downloads | auto-tuned |
| `.output_dir(path)` | Custom cache directory | HF default |
| `.timeout_per_file(dur)` | Per-file timeout | 300s |
| `.timeout_total(dur)` | Overall timeout | unlimited |
| `.max_retries(n)` | Retries per file | 3 |
| `.verify_checksums(bool)` | SHA256 verification | true |
| `.chunk_threshold(bytes)` | Min file size for multi-connection download | auto-tuned |
| `.connections_per_file(n)` | Parallel connections per large file | auto-tuned |
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

## Safetensors header inspection

Read tensor metadata without downloading full weight files. The library checks the local cache first and only makes HTTP Range requests on cache miss.

```rust
use hf_fetch_model::inspect;

// Single file (cache-first, falls back to 2 HTTP Range requests)
let (info, source) = inspect::inspect_safetensors(
    "google/gemma-2-2b-it",
    "model-00001-of-00002.safetensors",
    None,  // token
    None,  // revision (defaults to "main")
).await?;

for tensor in &info.tensors {
    println!("{}: {:?} {}", tensor.name, tensor.shape, tensor.dtype);
}
println!("{} tensors, {} params", info.tensors.len(), info.total_params());

// Local file only (no network)
let info = inspect::inspect_safetensors_local(Path::new("model.safetensors"))?;

// Cache-only (fail if not cached)
let info = inspect::inspect_safetensors_cached("google/gemma-2-2b-it", "model.safetensors", None)?;

// Shard index for sharded models (1 request instead of 2×N)
if let Some(index) = inspect::fetch_shard_index("google/gemma-2-2b-it", None, None).await? {
    println!("{} shards, {} tensors", index.shards.len(), index.weight_map.len());
}
```

`TensorInfo` provides helpers: `num_elements()` (product of shape), `byte_len()` (data size from offsets), and `dtype_bytes()` (bytes per element for known dtypes).

All types derive `serde::Serialize` for JSON output.

## Cache disk usage

```rust
use hf_fetch_model::cache;

// Per-repo file breakdown (sorted by size descending)
let files = cache::cache_repo_usage("google/gemma-2-2b-it")?;
for f in &files {
    println!("{}: {} bytes", f.filename, f.size);
}

// Cache-wide summary (all repos)
let summaries = cache::cache_summary()?;
for s in &summaries {
    println!("{}: {} bytes, {} files", s.repo_id, s.total_size, s.file_count);
}
```
