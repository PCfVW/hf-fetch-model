# hf-fetch-model

[![CI](https://github.com/PCfVW/hf-fetch-model/actions/workflows/ci.yml/badge.svg)](https://github.com/PCfVW/hf-fetch-model/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/hf-fetch-model)](https://crates.io/crates/hf-fetch-model)
[![docs.rs](https://img.shields.io/docsrs/hf-fetch-model)](https://docs.rs/hf-fetch-model)
[![Rust](https://img.shields.io/badge/rust-1.88%2B-orange)](https://www.rust-lang.org)
[![License](https://img.shields.io/crates/l/hf-fetch-model)](LICENSE-MIT)

A Rust library and CLI for downloading and inspecting HuggingFace models. Multi-connection parallel downloads, file filtering, checksum verification, retry — plus safetensors header inspection and tensor layout comparison between models, all without downloading weight data.

## Table of contents

- [Install](#install)
- [Commands](#commands)
- [Try it](#try-it)
- [Inspect & compare](#inspect--compare)
- [Disk usage](#disk-usage)
- [Library quick start](#library-quick-start)
- [Documentation](#documentation)
- [Used by](#used-by)
- [License](#license)
- [Development](#development)

## Install

```sh
cargo install hf-fetch-model --features cli
```

## Commands

| Command | Description |
|---------|-------------|
| `hf-fm <REPO_ID>` *(default)* | Download a model (multi-connection, auto-tuned) |
| `hf-fm cache clean-partial` | Remove `.chunked.part` files from interrupted downloads |
| `hf-fm cache delete <REPO_ID\|N>` | Delete a cached model |
| `hf-fm cache path <REPO_ID\|N>` | Print snapshot directory path (for scripting) |
| `hf-fm diff <REPO_A> <REPO_B>` | Compare tensor layouts between two models |
| `hf-fm discover` | Find new model families on the Hub |
| `hf-fm download-file <REPO_ID> <FILE>` | Download a single file (or glob pattern) |
| `hf-fm du [REPO_ID\|N]` | Show cache disk usage (by name or `#` index) |
| `hf-fm inspect <REPO_ID> [FILE]` | Inspect safetensors headers (tensor names, shapes, dtypes) without downloading weights |
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

$ hf-fm search llama --tag gguf --limit 3
Models matching "llama" (by downloads):

  hf-fm bartowski/Llama-3.2-3B-Instruct-GGUF             (489,856 downloads)  [text-generation]
  hf-fm bartowski/Meta-Llama-3.1-8B-Instruct-GGUF        (237,791 downloads)  [text-generation]
  hf-fm MaziyarPanahi/Meta-Llama-3.1-8B-Instruct-GGUF    (184,847 downloads)  [text-generation]

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

# Download to flat layout (files directly in ./models/)
$ hf-fm mistralai/Ministral-3-3B-Instruct-2512 --preset safetensors --flat --output-dir ./models

# Download sharded PyTorch files by glob
$ hf-fm download-file org/model "pytorch_model-*.bin"
```

## Inspect & compare

```
$ hf-fm inspect EleutherAI/pythia-1.4b model.safetensors --cached --filter "layers.0."
  Repo:     EleutherAI/pythia-1.4b
  File:     model.safetensors
  Source:   cached

  Tensor                                             Dtype    Shape                  Size     Params
  gpt_neox.layers.0.attention.dense.weight           F16      [2048, 2048]       8.00 MiB       4.2M
  gpt_neox.layers.0.mlp.dense_h_to_4h.weight         F16      [8192, 2048]      32.00 MiB      16.8M
  ...
  ────────────────────────────────────────────────────────────────────────────────────────────────
  15/364 tensors, 54.6M/1.52B params (filter: "layers.0.")

$ hf-fm inspect google/gemma-4-E2B-it model.safetensors --tree --filter "embed"
  Repo:     google/gemma-4-E2B-it
  File:     model.safetensors
  Source:   remote (2 HTTP requests)

  └── model.
      ├── embed_audio.embedding_projection.weight   BF16  [1536, 1536]   4.50 MiB
      ├── embed_vision.embedding_projection.weight  BF16  [1536, 768]    2.25 MiB
      ├── language_model.
      │   ├── embed_tokens.weight            BF16  [262144, 1536]      768.00 MiB
      │   └── embed_tokens_per_layer.weight  BF16  [262144, 8960]        4.38 GiB
      └── vision_tower.patch_embedder.
          ├── input_proj.weight         BF16  [768, 768]        1.12 MiB
          └── position_embedding_table  BF16  [2, 10240, 768]  30.00 MiB
  6/2011 tensors, 2.77B/5.12B params (filter: "embed")

$ hf-fm diff RedHatAI/Llama-3.2-1B-Instruct-FP8 casperhansen/llama-3.2-1b-instruct-awq --cached --summary
  A: RedHatAI/Llama-3.2-1B-Instruct-FP8
  B: casperhansen/llama-3.2-1b-instruct-awq
  ──────────────────────────────────────────────────────────────────────────────────────────────
  A: 371 tensors | B: 370 tensors | only-A: 337 | only-B: 336 | differ: 34 | match: 0
```

Inspect reads tensor metadata via HTTP Range requests (2 requests per file) — no weight data downloaded. The `--tree` flag shows the hierarchical namespace with numeric sibling groups auto-collapsed to `[0..N]` for structural discovery. Diff compares tensor names, dtypes, and shapes between any two models (remote or cached).

## Disk usage

```
$ hf-fm du
   #        SIZE  REPO                                             FILES
   1    5.10 GiB  google/gemma-2-2b-it                                 8
   2    2.80 GiB  EleutherAI/pythia-1.4b                              12  ●
   3    1.20 GiB  google/gemma-scope-2b-pt-res                         3
  ─────────────────────────────────────────────────────────────────────────────
   9.10 GiB  total (3 repos, 23 files)
  ● = partial downloads

$ hf-fm du 2
  EleutherAI/pythia-1.4b:

   #        SIZE  FILE
   1    2.50 GiB  model-00001-of-00002.safetensors
   2    0.26 GiB  model-00002-of-00002.safetensors
   ...
  ──────────────────────────────────────────────────────────────────
   2.80 GiB  total (12 files)

  ● partial downloads — run `hf-fm status EleutherAI/pythia-1.4b` for details

$ hf-fm du --age
   #        SIZE  REPO                                             FILES  AGE
   1    5.10 GiB  google/gemma-2-2b-it                                 8  2 days ago
   2    2.80 GiB  EleutherAI/pythia-1.4b                              12  45 days ago     ●
   3    1.20 GiB  google/gemma-scope-2b-pt-res                         3  3 months ago
  ─────────────────────────────────────────────────────────────────────────────────────────
   9.10 GiB  total (3 repos, 23 files)
  ● = partial downloads

$ hf-fm cache path google/gemma-2-2b-it
/home/user/.cache/huggingface/hub/models--google--gemma-2-2b-it/snapshots/abc1234
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
| [FAQ](docs/FAQ.md) | Common questions — installation, auth, cache location, discovery, errors |
| [Search](docs/search.md) | Comma filtering, `--exact`, model card metadata |
| [Configuration](docs/configuration.md) | Builder API, presets, progress callbacks |
| [Architecture](docs/architecture.md) | How hf-fetch-model relates to `hf-hub` and `candle-mi` |
| [Diagnostics](docs/diagnostics.md) | `--verbose` output, `tracing` setup for library users |
| [Upstream differences](docs/upstream-differences.md) | Where hf-fetch-model diverges from Python `huggingface_hub`/`hf_transfer` |
| [Candle example](examples/candle_inspect.rs) | Inspect tensor layouts before downloading — for candle users |
| [Changelog](CHANGELOG.md) | Release history and migration notes |

## Used by

- [candle-mi](https://github.com/PCfVW/candle-mi) — Mechanistic interpretability toolkit for language models

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE)
or [MIT License](LICENSE-MIT) at your option.

## Development

- Exclusively developed with [Claude Code](https://claude.com/product/claude-code) (dev) and [Augment Code](https://www.augmentcode.com/) (review)
- Git workflow managed with [Fork](https://fork.dev/)
- All code follows [CONVENTIONS.md](CONVENTIONS.md), derived from [Amphigraphic-Strict](https://github.com/PCfVW/Amphigraphic-Strict)'s [Grit](https://github.com/PCfVW/Amphigraphic-Strict/tree/master/Grit) — a strict Rust subset designed to improve AI coding accuracy.
