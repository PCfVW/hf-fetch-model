# CLI Reference

hf-fetch-model installs two binaries: `hf-fetch-model` (explicit) and `hf-fm` (short alias).

```sh
cargo install hf-fetch-model --features cli
```

## Table of contents

- [Subcommands](#subcommands)
- [Download examples](#download-examples)
- [Dry-run example](#dry-run-example)
- [List-files examples](#list-files-examples)
- [Search examples](#search-examples)
- [Other commands](#other-commands)
- [Download flags](#download-flags)
- [List-files flags](#list-files-flags)
- [Search flags](#search-flags)
- [General flags](#general-flags)

## Subcommands

| Command | Description |
|---------|-------------|
| *(default)* | Download a model: `hf-fm <REPO_ID>` |
| `discover` | Find new model families on the Hub not yet cached locally |
| `download-file <REPO_ID> <FILENAME>` | Download a single file and print its cache path |
| `list-families` | List model families (`model_type`) in local cache |
| `list-files <REPO_ID>` | List files in a remote repo (filenames, sizes, SHA256) without downloading |
| `search <QUERY>` | Search the HuggingFace Hub for models (by downloads) |
| `status [REPO_ID]` | Show download status — per-repo detail, or cache-wide summary |

`<ARG>` = required, `[ARG]` = optional.

## Download examples

```sh
# Download all files
hf-fm google/gemma-2-2b-it

# Download safetensors + config only
hf-fm google/gemma-2-2b-it --preset safetensors

# Custom filters
hf-fm google/gemma-2-2b-it --filter "*.safetensors" --filter "*.json"

# Download to a specific directory
hf-fm google/gemma-2-2b-it --output-dir ./models

# Download a single file
hf-fm download-file mntss/clt-gemma-2-2b-426k W_dec_0.safetensors

# Download with diagnostics
hf-fm google/gemma-2-2b-it -v
```

## Dry-run example

Preview what would be downloaded before committing:

```sh
hf-fm google/gemma-2-2b-it --preset safetensors --dry-run
```

Output shows per-file status (cached / to download), total and download sizes, and a recommended config based on the file size distribution.

## List-files examples

```sh
# List all files in a repo
hf-fm list-files google/gemma-2-2b-it

# List only safetensors-related files
hf-fm list-files google/gemma-2-2b-it --preset safetensors

# Custom filter
hf-fm list-files google/gemma-2-2b-it --filter "*.safetensors"

# Hide SHA256 column
hf-fm list-files google/gemma-2-2b-it --no-checksum

# Show which files are already in local cache
hf-fm list-files google/gemma-2-2b-it --show-cached
```

## Search examples

See [Search](search.md) for the full feature set.

```sh
# Basic search
hf-fm search RWKV-7

# Multi-term filtering
hf-fm search mistral,3B,instruct

# Exact match with model card
hf-fm search mistralai/Ministral-3-3B-Instruct-2512 --exact
```

## Other commands

```sh
# Check download status (per-repo or entire cache)
hf-fm status RWKV/RWKV7-Goose-World3-1.5B-HF
hf-fm status

# List model families in local cache
hf-fm list-families

# Discover new families from HuggingFace Hub
hf-fm discover
```

## Download flags

These flags apply to the default download command (`hf-fm <REPO_ID>`). `download-file` shares the performance flags but not `--dry-run`, `--filter`, or `--preset`.

| Flag | Description | Default |
|------|-------------|---------|
| `-v`, `--verbose` | Enable download diagnostics (plan, per-file decisions, throughput) | off |
| `--dry-run` | Preview what would be downloaded (no actual download) | off |
| `--chunk-threshold-mib` | Min file size (MiB) for multi-connection download | 100 |
| `--concurrency` | Parallel file downloads | 4 |
| `--connections-per-file` | Parallel HTTP connections per large file | 8 |
| `--exclude` | Exclude glob pattern (repeatable) | none |
| `--filter` | Include glob pattern (repeatable) | all files |
| `--output-dir` | Custom output directory | HF cache |
| `--preset` | Filter preset: `safetensors`, `gguf`, `config-only` | — |
| `--revision` | Git revision (branch, tag, SHA) | main |
| `--token` | Auth token (or set `HF_TOKEN` env var) | — |

## List-files flags

| Flag | Description | Default |
|------|-------------|---------|
| `--exclude` | Exclude glob pattern (repeatable) | none |
| `--filter` | Include glob pattern (repeatable) | all files |
| `--no-checksum` | Suppress the SHA256 column | off |
| `--preset` | Filter preset: `safetensors`, `gguf`, `config-only` | — |
| `--revision` | Git revision (branch, tag, SHA) | main |
| `--show-cached` | Show cache status: complete (✓), partial, or missing (✗) | off |
| `--token` | Auth token (or set `HF_TOKEN` env var) | — |

## Search flags

| Flag | Description | Default |
|------|-------------|---------|
| `--exact` | Return only the exact model ID match; show model card metadata | off |
| `--limit` | Maximum number of results | 20 |

## General flags

| Flag | Description |
|------|-------------|
| `-h`, `--help` | Print help |
| `-V`, `--version` | Print version |

Subcommands accept their own flags. Run `hf-fm <command> --help` for details.
