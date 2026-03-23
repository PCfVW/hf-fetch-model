# hf-fetch-model

[![CI](https://github.com/PCfVW/hf-fetch-model/actions/workflows/ci.yml/badge.svg)](https://github.com/PCfVW/hf-fetch-model/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/hf-fetch-model)](https://crates.io/crates/hf-fetch-model)
[![docs.rs](https://img.shields.io/docsrs/hf-fetch-model)](https://docs.rs/hf-fetch-model)
[![Rust](https://img.shields.io/badge/rust-1.88%2B-orange)](https://www.rust-lang.org)
[![License](https://img.shields.io/crates/l/hf-fetch-model)](LICENSE-MIT)

A Rust library and CLI for downloading HuggingFace models at maximum speed. Multi-connection parallel downloads, file filtering, checksum verification, retry — and a search command to find models before you download them.

## Install

```sh
cargo install hf-fetch-model --features cli
```

## Commands

| Command | |
|---------|---|
| `hf-fm <REPO_ID>` | Download a model (multi-connection, auto-tuned) |
| `hf-fm diff <REPO_A> <REPO_B>` | Compare tensor layouts between two models |
| `hf-fm discover` | Find new model families on the Hub |
| `hf-fm download-file <REPO_ID> <FILE>` | Download a single file |
| `hf-fm du [REPO_ID]` | Show cache disk usage |
| `hf-fm inspect <REPO_ID> [FILE]` | Inspect safetensors headers (tensor names, shapes, dtypes) |
| `hf-fm list-families` | List model families in local cache |
| `hf-fm list-files <REPO_ID>` | List remote files (sizes, SHA256) without downloading |
| `hf-fm search <QUERY>` | Search the HuggingFace Hub for models |
| `hf-fm status [REPO_ID]` | Show download status (complete / partial / missing) |

See [CLI Reference](docs/cli-reference.md) for all flags and output examples.

## Try it

```
$ hf-fm search mistral,3B,instruct
Models matching "mistral,3B,instruct" (by downloads):

  hf-fm mistralai/Ministral-3-3B-Instruct-2512           (159,700 downloads)
  hf-fm mistralai/Ministral-3-3B-Instruct-2512-BF16      (62,600 downloads)
  hf-fm mistralai/Ministral-3-3B-Instruct-2512-GGUF      (32,700 downloads)
  ...

$ hf-fm search mistralai/Ministral-3-3B-Instruct-2512 --exact
Exact match:

  hf-fm mistralai/Ministral-3-3B-Instruct-2512           (159,700 downloads)

  License:      apache-2.0
  Pipeline:     text-generation
  Library:      vllm
  Languages:    en, fr, es, de, it, pt, nl, zh, ja, ko, ar

$ hf-fm list-files mistralai/Ministral-3-3B-Instruct-2512 --preset safetensors
  File                                               Size      SHA256
  model-00001-of-00002.safetensors                 3.68 GiB    a1b2c3d4e5f6
  model-00002-of-00002.safetensors                 2.88 GiB    f6e5d4c3b2a1
  config.json                                        856 B     —
  ...
  7 files, 6.57 GiB total

$ hf-fm mistralai/Ministral-3-3B-Instruct-2512 --preset safetensors --dry-run
  Repo:     mistralai/Ministral-3-3B-Instruct-2512
  Revision: main

  File                                               Size      Status
  model-00001-of-00002.safetensors                 3.68 GiB    to download
  model-00002-of-00002.safetensors                 2.88 GiB    to download
  ...
  Total: 6.57 GiB (7 files, 0 cached, 7 to download)

  Recommended config:
    concurrency:        2
    connections/file:   8
    chunk threshold:  100 MiB

$ hf-fm mistralai/Ministral-3-3B-Instruct-2512 --preset safetensors
Downloaded to: ~/.cache/huggingface/hub/models--mistralai--Ministral-3-3B.../snapshots/...
  6.57 GiB in 18.2s (369.1 MiB/s)
```

## Library quick start

```rust
let outcome = hf_fetch_model::download(
    "google/gemma-2-2b-it".to_owned(),
).await?;

println!("Model at: {}", outcome.inner().display());
```

Filter, progress, auth, and more via the builder — see [Configuration](docs/configuration.md).

## Documentation

| Topic | |
|-------|---|
| [CLI Reference](docs/cli-reference.md) | All subcommands, flags, and output examples |
| [Search](docs/search.md) | Comma filtering, `--exact`, model card metadata |
| [Configuration](docs/configuration.md) | Builder API, presets, progress callbacks |
| [Architecture](docs/architecture.md) | How hf-fetch-model relates to `hf-hub` and `candle-mi` |
| [Diagnostics](docs/diagnostics.md) | `--verbose` output, `tracing` setup for library users |
| [Changelog](CHANGELOG.md) | Release history and migration notes |

## Used by

- [candle-mi](https://github.com/PCfVW/candle-mi) — Mechanistic interpretability toolkit for transformer models

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE)
or [MIT License](LICENSE-MIT) at your option.

## Development

- Exclusively developed with [Claude Code](https://claude.com/product/claude-code) (dev) and [Augment Code](https://www.augmentcode.com/) (review)
- Git workflow managed with [Fork](https://fork.dev/)
- All code follows [CONVENTIONS.md](CONVENTIONS.md), derived from [Amphigraphic-Strict](https://github.com/PCfVW/Amphigraphic-Strict)'s [Grit](https://github.com/PCfVW/Amphigraphic-Strict/tree/master/Grit) — a strict Rust subset designed to improve AI coding accuracy.
