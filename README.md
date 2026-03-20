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

## Try it

```
$ hf-fm search mistral,3B,instruct
Models matching "mistral,3B,instruct" (by downloads):

  hf-fm mistralai/Ministral-3-3B-Instruct-2512           (159.7K downloads)
  hf-fm mistralai/Ministral-3-3B-Instruct-2512-BF16      (62.6K downloads)
  hf-fm mistralai/Ministral-3-3B-Instruct-2512-GGUF      (32.7K downloads)
  ...

$ hf-fm search mistralai/Ministral-3-3B-Instruct-2512 --exact
Exact match:

  hf-fm mistralai/Ministral-3-3B-Instruct-2512           (159.7K downloads)

  License:      apache-2.0
  Pipeline:     text-generation
  Library:      vllm
  Languages:    en, fr, es, de, it, pt, nl, zh, ja, ko, ar

$ hf-fm mistralai/Ministral-3-3B-Instruct-2512 --preset safetensors
Downloaded to: ~/.cache/huggingface/hub/models--mistralai--Ministral-3-3B.../snapshots/...
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
