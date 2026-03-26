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
- [Info examples](#info-examples)
- [Inspect examples](#inspect-examples)
- [Diff examples](#diff-examples)
- [Disk usage examples](#disk-usage-examples)
- [Other commands](#other-commands)
- [Diff flags](#diff-flags)
- [Download flags](#download-flags)
- [Info flags](#info-flags)
- [Inspect flags](#inspect-flags)
- [List-files flags](#list-files-flags)
- [Search flags](#search-flags)
- [General flags](#general-flags)

## Subcommands

| Command | Description |
|---------|-------------|
| *(default)* | Download a model: `hf-fm <REPO_ID>` |
| `diff <REPO_A> <REPO_B>` | Compare tensor layouts between two models |
| `discover` | Find new model families on the Hub not yet cached locally |
| `info <REPO_ID>` | Show model card metadata and README text |
| `download-file <REPO_ID> <FILENAME>` | Download a single file and print its cache path |
| `du [REPO_ID]` | Show cache disk usage — per-repo breakdown, or cache-wide summary |
| `inspect <REPO_ID> [FILENAME]` | Inspect safetensors file headers (tensor names, shapes, dtypes) |
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

After a successful download, a summary line shows total size, elapsed time, and throughput:

```
Downloaded to: ~/.cache/huggingface/hub/models--google--gemma-2-2b-it/snapshots/...
  4.89 GiB in 114.9s (43.5 MiB/s)
```

In non-TTY contexts (pipes, CI), periodic progress lines are emitted to stderr instead of progress bars:

```
[hf-fm] model-00002-of-00002.safetensors: 22.96 MiB/229.54 MiB (10%)
[hf-fm] model-00001-of-00002.safetensors: 475.71 MiB/4.65 GiB (10%)
```

A warning is emitted when `--filter` duplicates a pattern already included by `--preset`:

```
warning: --filter "*.safetensors" is redundant with --preset safetensors
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

# Filter by library
hf-fm search llama --library peft

# Filter by pipeline task
hf-fm search mistral --pipeline text-generation
```

Common quantization synonyms are normalized automatically: `8bit`, `8-bit`, `int8`, and `INT8` all produce the same results. Same for `4bit`/`4-bit`/`int4` and `fp8`/`float8`.

## Info examples

```sh
# Show metadata and first 40 lines of README
hf-fm info mistralai/Ministral-3-3B-Instruct-2512

# Show full README
hf-fm info mistralai/Ministral-3-3B-Instruct-2512 --lines 0

# JSON output
hf-fm info mistralai/Ministral-3-3B-Instruct-2512 --json

# Specific revision
hf-fm info mistralai/Ministral-3-3B-Instruct-2512 --revision v1.0
```

## Inspect examples

```sh
# Inspect a single safetensors file (cache-first, falls back to HTTP Range requests)
hf-fm inspect google/gemma-2-2b-it model-00001-of-00002.safetensors

# Inspect from cache only (no network)
hf-fm inspect google/gemma-2-2b-it model-00001-of-00002.safetensors --cached

# JSON output for programmatic consumption
hf-fm inspect google/gemma-2-2b-it model-00001-of-00002.safetensors --json

# Inspect all safetensors in a repo (uses shard index fast path when available)
hf-fm inspect google/gemma-2-2b-it

# Suppress metadata line
hf-fm inspect google/gemma-2-2b-it model-00001-of-00002.safetensors --no-metadata
```

## Diff examples

```sh
# Compare tensor layouts between two model variants (cache-first)
hf-fm diff RedHatAI/Llama-3.2-1B-Instruct-FP8 casperhansen/llama-3.2-1b-instruct-awq

# Cache-only (no network)
hf-fm diff RedHatAI/Llama-3.2-1B-Instruct-FP8 casperhansen/llama-3.2-1b-instruct-awq --cached

# Filter to specific layers
hf-fm diff RedHatAI/Llama-3.2-1B-Instruct-FP8 casperhansen/llama-3.2-1b-instruct-awq --filter "layers.0"

# Quick summary (counts only, no tensor listing)
hf-fm diff RedHatAI/Llama-3.2-1B-Instruct-FP8 casperhansen/llama-3.2-1b-instruct-awq --cached --summary

# JSON output for programmatic consumption
hf-fm diff RedHatAI/Llama-3.2-1B-Instruct-FP8 casperhansen/llama-3.2-1b-instruct-awq --cached --json
```

## Disk usage examples

```sh
# Show all cached repos sorted by size
hf-fm du

# Show per-file breakdown for a specific repo
hf-fm du google/gemma-2-2b-it
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

## Diff flags

| Flag | Description | Default |
|------|-------------|---------|
| `--cached` | Cache-only mode: fail if files are not cached locally | off |
| `--filter` | Show only tensors whose name contains this substring | — |
| `--json` | Output the full diff as JSON | off |
| `--revision-a` | Git revision for model A | main |
| `--revision-b` | Git revision for model B | main |
| `--summary` | Show only the summary line (counts per category) | off |
| `--token` | Auth token (or set `HF_TOKEN` env var) | — |

## Download flags

These flags apply to the default download command (`hf-fm <REPO_ID>`). `download-file` shares the performance flags but not `--dry-run`, `--filter`, or `--preset`.

| Flag | Description | Default |
|------|-------------|---------|
| `-v`, `--verbose` | Enable download diagnostics (plan, per-file decisions, throughput) | off |
| `--dry-run` | Preview what would be downloaded (no actual download) | off |
| `--chunk-threshold-mib` | Min file size (MiB) for multi-connection download | auto-tuned |
| `--concurrency` | Parallel file downloads | auto-tuned |
| `--connections-per-file` | Parallel HTTP connections per large file | auto-tuned |
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
| `--library` | Filter by library framework (e.g., `transformers`, `peft`, `vllm`) | — |
| `--limit` | Maximum number of results | 20 |
| `--pipeline` | Filter by pipeline task (e.g., `text-generation`, `text-classification`) | — |

## Info flags

| Flag | Description | Default |
|------|-------------|---------|
| `--json` | Output metadata and README as JSON | off |
| `--lines` | Maximum lines of README to display (0 = all) | 40 |
| `--revision` | Git revision (branch, tag, SHA) | main |
| `--token` | Auth token (or set `HF_TOKEN` env var) | — |

## Inspect flags

| Flag | Description | Default |
|------|-------------|---------|
| `--cached` | Cache-only mode: fail if the file is not cached locally | off |
| `--filter` | Show only tensors whose name contains this substring | — |
| `--json` | Output the full header as JSON instead of a human-readable table | off |
| `--no-metadata` | Suppress the `Metadata:` line in human-readable output | off |
| `--revision` | Git revision (branch, tag, SHA) | main |
| `--token` | Auth token (or set `HF_TOKEN` env var) | — |

## General flags

| Flag | Description |
|------|-------------|
| `-h`, `--help` | Print help |
| `-V`, `--version` | Print version |

Subcommands accept their own flags. Run `hf-fm <command> --help` for details.
