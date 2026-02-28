# hf-fetch-model

[![CI](https://github.com/PCfVW/hf-fetch-model/actions/workflows/ci.yml/badge.svg)](https://github.com/PCfVW/hf-fetch-model/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/hf-fetch-model)](https://crates.io/crates/hf-fetch-model)
[![docs.rs](https://img.shields.io/docsrs/hf-fetch-model)](https://docs.rs/hf-fetch-model)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange)](https://www.rust-lang.org)
[![License](https://img.shields.io/crates/l/hf-fetch-model)](LICENSE-MIT)

*Download HuggingFace models at full speed with a single function call.*

## Features

- **Repo-level download** — give it a model ID, get all files
- **Maximum throughput** — parallel chunked downloads via hf-hub's `.high()` mode
- **File filtering** — glob patterns (`*.safetensors`) and presets (`safetensors`, `gguf`, `config-only`)
- **Progress reporting** — per-file callbacks, optional `indicatif` progress bars
- **Checksum verification** — SHA256 against HuggingFace LFS metadata
- **Retry with backoff** — exponential backoff + jitter for flaky connections
- **Timeout control** — per-file and overall time limits
- **HF cache compatible** — files stored in `~/.cache/huggingface/hub/`
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

# List model families in local cache
hf-fm list-families

# Discover new families from HuggingFace Hub
hf-fm discover
```

### CLI Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--concurrency` | Parallel file downloads | 4 |
| `--exclude` | Exclude glob pattern (repeatable) | none |
| `--filter` | Include glob pattern (repeatable) | all files |
| `-h`, `--help` | Print help | — |
| `--output-dir` | Custom output directory | HF cache |
| `--preset` | Filter preset: `safetensors`, `gguf`, `config-only` | — |
| `--revision` | Git revision (branch, tag, SHA) | main |
| `--token` | Auth token (or set `HF_TOKEN` env var) | — |
| `-V`, `--version` | Print version | — |

## Architecture

```
candle-mi
  download_model() convenience fn
         │ optional dep (feature = "fast-download")
hf-fetch-model
  • repo file listing
  • file filtering (glob patterns)
  • parallel file orchestration
  • progress callbacks
  • checksum verification
  • resume / retry
         │ dep
hf-hub (tokio, .high())
  • HTTP chunked parallel download
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
| `.on_progress(closure)` | Progress callback | — |

## Used by

- [candle-mi](https://github.com/PCfVW/candle-mi) — Mechanistic interpretability toolkit for transformer models, uses hf-fetch-model for fast model downloads (optional `fast-download` feature).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE)
or [MIT License](LICENSE-MIT) at your option.
