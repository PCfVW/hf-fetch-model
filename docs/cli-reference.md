# CLI Reference

hf-fetch-model installs two binaries: `hf-fetch-model` (explicit) and `hf-fm` (short alias).

```sh
cargo install hf-fetch-model --features cli
```

## Subcommands

| Command | Description |
|---------|-------------|
| *(default)* | Download a model: `hf-fm <REPO_ID>` |
| `download-file <REPO_ID> <FILENAME>` | Download a single file and print its cache path |
| `search <QUERY>` | Search the HuggingFace Hub for models (by downloads) |
| `status [REPO_ID]` | Show download status — per-repo detail, or cache-wide summary |
| `list-families` | List model families (`model_type`) in local cache |
| `discover` | Find new model families on the Hub not yet cached locally |

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

## Search flags

| Flag | Description | Default |
|------|-------------|---------|
| `--limit` | Maximum number of results | 20 |
| `--exact` | Return only the exact model ID match; show model card metadata | off |

## General flags

| Flag | Description |
|------|-------------|
| `-h`, `--help` | Print help |
| `-V`, `--version` | Print version |

Subcommands accept their own flags. Run `hf-fm <command> --help` for details.
