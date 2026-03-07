# hf-fetch-model

[![CI](https://github.com/PCfVW/hf-fetch-model/actions/workflows/ci.yml/badge.svg)](https://github.com/PCfVW/hf-fetch-model/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/hf-fetch-model)](https://crates.io/crates/hf-fetch-model)
[![docs.rs](https://img.shields.io/docsrs/hf-fetch-model)](https://docs.rs/hf-fetch-model)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange)](https://www.rust-lang.org)
[![License](https://img.shields.io/crates/l/hf-fetch-model)](LICENSE-MIT)

*Download HuggingFace models at full speed with a single function call.*

## Features

- **Single-file download** — download one file by name, get its cache path
- **Repo-level download** — give it a model ID, get all files
- **Maximum throughput** — multi-connection parallel Range downloads for large files (≥100 MiB, 8 connections by default) enabled for **all** download functions, plus hf-hub's `.high()` mode
- **Download diagnostics** — structured `tracing` events at `debug` level report download plan, per-file chunked/single decisions, throughput, and completion summary
- **File filtering** — glob patterns (`*.safetensors`) and presets (`safetensors`, `gguf`, `config-only`)
- **HF cache compatible** — files stored in `~/.cache/huggingface/hub/`
- **Progress reporting** — per-file callbacks, optional `indicatif` progress bars
- **Checksum verification** — SHA256 against HuggingFace LFS metadata
- **Retry with backoff** — exponential backoff + jitter for flaky connections
- **Timeout control** — per-file and overall time limits
- **Cache diagnostics** — `status` command shows per-file download state (complete / partial / missing)
- **Model search** — `search` command queries the HuggingFace Hub by keyword
- **CLI included** — `hf-fetch-model` / `hf-fm` binary for command-line use

## Installation

### Library

```sh
cargo add hf-fetch-model
```

### CLI

```sh
cargo install hf-fetch-model --features cli
```

This installs two binaries: `hf-fetch-model` (explicit) and `hf-fm` (short alias).

## Quick Start (Library)

```rust
// Minimal — download everything
let path = hf_fetch_model::download("google/gemma-2-2b-it".to_owned()).await?;

// Configured — filter + progress
use hf_fetch_model::{FetchConfig, Filter};

let config = Filter::safetensors()
    .on_progress(|e| println!("{}: {:.1}%", e.filename, e.percent))
    .build()?;

let path = hf_fetch_model::download_with_config(
    "google/gemma-2-2b-it".to_owned(),
    &config,
).await?;
```

Blocking wrappers (`download_blocking()`, `download_with_config_blocking()`) are available for non-async callers.

### Single-file download

```rust
use hf_fetch_model::FetchConfig;

let config = FetchConfig::builder()
    .on_progress(|e| println!("{}: {:.1}%", e.filename, e.percent))
    .build()?;

// Download one file — returns the local cache path
let path = hf_fetch_model::download_file(
    "mntss/clt-gemma-2-2b-426k".to_owned(),
    "W_enc_0.safetensors",
    &config,
).await?;

// Blocking variant for non-async callers
let path = hf_fetch_model::download_file_blocking(
    "mntss/clt-gemma-2-2b-426k".to_owned(),
    "config.yaml",
    &config,
)?;
```

## CLI Usage

```sh
# Download all files
hf-fetch-model google/gemma-2-2b-it

# Download safetensors + config only
hf-fm google/gemma-2-2b-it --preset safetensors

# Custom filters
hf-fm google/gemma-2-2b-it --filter "*.safetensors" --filter "*.json"

# Download to a specific directory
hf-fm google/gemma-2-2b-it --output-dir ./models

# Download a single file
hf-fm download-file mntss/clt-gemma-2-2b-426k W_dec_0.safetensors

# Search for models on HuggingFace Hub
hf-fm search RWKV-7

# Check download status (per-repo or entire cache)
hf-fm status RWKV/RWKV7-Goose-World3-1.5B-HF
hf-fm status

# List model families in local cache
hf-fm list-families

# Discover new families from HuggingFace Hub
hf-fm discover

# Download with diagnostics (chunked/single decisions, throughput)
hf-fm google/gemma-2-2b-it -v
```

### Subcommands

| Command | Description |
|---------|-------------|
| *(default)* | Download a model: `hf-fm <REPO_ID>` |
| `download-file <REPO_ID> <FILENAME>` | Download a single file and print its cache path |
| `search <QUERY>` | Search the HuggingFace Hub for models (by downloads) |
| `status [REPO_ID]` | Show download status — per-repo detail, or cache-wide summary |
| `list-families` | List model families (`model_type`) in local cache |
| `discover` | Find new model families on the Hub not yet cached locally |

`<ARG>` = required, `[ARG]` = optional.

### Download Flags

These flags apply to the default download command (`hf-fm <REPO_ID>`) and `download-file`.

| Flag | Description | Default |
|------|-------------|---------|
| `-v`, `--verbose` | Enable download diagnostics (plan, per-file decisions, throughput) | off |
| `--chunk-threshold-mib` | Min file size (MiB) for multi-connection download | 100 |
| `--concurrency` | Parallel file downloads | 4 |
| `--connections-per-file` | Parallel HTTP connections per large file | 8 |
| `--exclude` | Exclude glob pattern (repeatable) | none |
| `--filter` | Include glob pattern (repeatable) | all files |
| `--output-dir` | Custom output directory | HF cache |
| `--preset` | Filter preset: `safetensors`, `gguf`, `config-only` | — |
| `--revision` | Git revision (branch, tag, SHA) | main |
| `--token` | Auth token (or set `HF_TOKEN` env var) | — |

### General Flags

| Flag | Description |
|------|-------------|
| `-h`, `--help` | Print help |
| `-V`, `--version` | Print version |

Subcommands accept their own flags (e.g., `--limit` for `search` and `discover`). Run `hf-fm <command> --help` for details.

## Download Diagnostics

hf-fetch-model emits structured `tracing` events at `debug` level to help diagnose download performance. In the CLI, use the `--verbose` / `-v` flag. For library users, initialize a `tracing-subscriber` at `debug` level (e.g., `RUST_LOG=hf_fetch_model=debug`):

```sh
# CLI — verbose flag (prints diagnostics to stderr)
hf-fm google/gemma-2-2b-it -v
```

Example output:

```
DEBUG hf_fetch_model: listing repository files repo_id="allenai/OLMo-1B-hf"
DEBUG hf_fetch_model: metadata fetch succeeded files_with_size=8 total_files=8
DEBUG hf_fetch_model: download plan total_files=8 concurrency=4 connections_per_file=8 chunk_threshold_mib=100 chunked_enabled=true
DEBUG hf_fetch_model: chunked download (multi-connection) filename="model.safetensors" size_mib=2475 connections=8
DEBUG hf_fetch_model: single-connection download (below chunk threshold) filename="config.json" size_mib=0
DEBUG hf_fetch_model: download complete filename="model.safetensors" elapsed_secs="23.1" throughput_mbps="857.2"
DEBUG hf_fetch_model: download complete files_downloaded=8 files_failed=0 total_elapsed_secs="24.3"
```

Key diagnostics:
- **"metadata fetch failed"** (warning): file sizes are unknown, so chunked downloads are disabled — all files use single-connection download.
- **"single-connection download" with reason "file size unknown"**: metadata was not available for this file.
- **"chunked download"**: file exceeds `chunk_threshold` and is being downloaded with multiple parallel HTTP Range connections.
- **throughput_mbps**: actual per-file throughput, useful for comparing single vs chunked performance.

## Architecture

```
candle-mi
  download_model() convenience fn
         │ optional dep (feature = "fast-download")
hf-fetch-model
  • repo file listing
  • file filtering (glob patterns)
  • parallel file orchestration
  • multi-connection Range downloads (large files)
  • progress callbacks
  • checksum verification
  • resume / retry
  • cache diagnostics & model search
         │ dep
hf-hub (tokio, .high())
  • single-connection download (.high() mode)
  • HF cache layout compatibility
  • auth token handling
```

## Configuration

| Builder method | Description | Default |
|---------------|-------------|---------|
| `.revision(rev)` | Git revision | `"main"` |
| `.token(tok)` | Auth token | — |
| `.token_from_env()` | Read `HF_TOKEN` env var | — |
| `.filter(glob)` | Include pattern (repeatable) | all files |
| `.exclude(glob)` | Exclude pattern (repeatable) | none |
| `.concurrency(n)` | Parallel downloads | 4 |
| `.output_dir(path)` | Custom cache directory | HF default |
| `.timeout_per_file(dur)` | Per-file timeout | 300s |
| `.timeout_total(dur)` | Overall timeout | unlimited |
| `.max_retries(n)` | Retries per file | 3 |
| `.verify_checksums(bool)` | SHA256 verification | true |
| `.chunk_threshold(bytes)` | Min file size for multi-connection download | 100 MiB |
| `.connections_per_file(n)` | Parallel connections per large file | 8 |
| `.on_progress(closure)` | Progress callback | — |

## Used by

- [candle-mi](https://github.com/PCfVW/candle-mi) — Mechanistic interpretability toolkit for transformer models, uses hf-fetch-model for fast model downloads (optional `fast-download` feature).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE)
or [MIT License](LICENSE-MIT) at your option.
