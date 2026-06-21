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
- [Du flags](#du-flags)
- [Other commands](#other-commands)
- [Cache commands](#cache-commands)
- [Cache clean-partial flags](#cache-clean-partial-flags)
- [Cache delete flags](#cache-delete-flags)
- [Cache gc flags](#cache-gc-flags)
- [Cache verify flags](#cache-verify-flags)
- [Diff flags](#diff-flags)
- [Download flags](#download-flags)
- [List-files flags](#list-files-flags)
- [Search flags](#search-flags)
- [List-families flags](#list-families-flags)
- [Status flags](#status-flags)
- [Info flags](#info-flags)
- [Inspect flags](#inspect-flags)
- [General flags](#general-flags)

## Subcommands

| Command | Description |
|---------|-------------|
| *(default)* | Download a model: `hf-fm <REPO_ID>` |
| `diff <REPO_A> <REPO_B>` | Compare tensor layouts between two models |
| `discover` | Find new model families on the Hub not yet cached locally |
| `info <REPO_ID>` | Show model card metadata and README text |
| `download-file <REPO_ID> <FILENAME>` | Download a single file (or glob pattern) and print its cache path |
| `du [REPO_ID\|N]` | Show cache disk usage — per-repo breakdown (by name or `#` index), or cache-wide summary |
| `cache clean-partial [REPO_ID\|N]` | Remove `.chunked.part` files from interrupted downloads |
| `cache delete <REPO_ID\|N>` | Delete a cached model (entire `models--org--name/` directory) |
| `cache path <REPO_ID\|N>` | Print the snapshot directory path for scripting |
| `cache verify <REPO_ID\|N>` | Re-verify SHA256 digests of cached files against HuggingFace LFS metadata |
| `inspect <REPO_ID> [FILENAME]` | Inspect `.safetensors` headers (remote or cached) and `.gguf` / `.npz` / `.pth` files (cached only) — tensor names, shapes, dtypes; auto-detects PEFT adapter config |
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

# Download sharded PyTorch files by glob pattern
hf-fm download-file org/model "pytorch_model-*.bin"

# Preview which files / how many bytes a glob would pull, without downloading
hf-fm download-file org/model "pytorch_model-*.bin" --dry-run

# Download to flat layout (files directly in target directory)
hf-fm google/gemma-2-2b-it --preset safetensors --flat --output-dir ./models

# Download a single file to flat layout
hf-fm download-file org/model config.json --flat --output-dir ./configs

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

# Emit JSON for scripting (size budgeting, checksum manifests)
hf-fm list-files google/gemma-2-2b-it --json | jq '.total_bytes'
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

# Filter by tag (useful for GGUF models without a library_name)
hf-fm search llama --tag gguf

# Combine a free-text query with a tag filter (text-match AND tag-match)
hf-fm search fp4 --tag bitsandbytes

# Enrich result rows with inline tag list (free) and total repo size (one extra HTTP request per row)
hf-fm search fp4 --tag bitsandbytes --show tags,size
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

For a narrative walkthrough using a real 4-shard model, see the [Inspect tutorial](tutorials/inspect-before-downloading.md). `inspect` covers `.safetensors` (remote or cached) plus `.gguf` (v0.10.2), `.npz`, and `.pth` (both v0.10.3) for **cached** files; remote inspect for those three is planned for v0.11. An unsupported extension is rejected with a clear error.

Gated repos (Meta Llama, Google Gemma, …) need an accepted license plus a token for `inspect`'s Range requests; on a 401/403 the error names the gate and the license URL instead of the raw status (v0.10.5). Note that a gated repo's file *listing* is public — `--list` or `list-files` succeeding does not prove content access. See the [FAQ entry on tokens and gated models](FAQ.md#how-do-i-pass-a-huggingface-token-why-does-a-gated-model-fail).

```sh
# Inspect a single safetensors file (cache-first, falls back to HTTP Range requests)
hf-fm inspect google/gemma-2-2b-it model-00001-of-00002.safetensors

# Inspect from cache only (no network)
hf-fm inspect google/gemma-2-2b-it model-00001-of-00002.safetensors --cached

# JSON output for programmatic consumption
hf-fm inspect google/gemma-2-2b-it model-00001-of-00002.safetensors --json

# Inspect all safetensors in a repo (uses shard index fast path when available)
hf-fm inspect google/gemma-2-2b-it

# Repo-level filter: list the matched tensor names nested under each shard/file
# (case-insensitive substring; add --limit N to cap a broad match)
hf-fm inspect google/gemma-2-2b-it --filter "layers.0."

# List the repo's tensor files (.safetensors / .gguf / .npz / .pth) — no headers read
hf-fm inspect google/gemma-2-2b-it --list

# Inspect file #2 from the --list numbering (1-based, alphabetical)
hf-fm inspect google/gemma-2-2b-it 2

# Pick the file interactively (numbered prompt on stderr; v0.10.5+)
hf-fm inspect google/gemma-2-2b-it --pick

# Narrow the picker by case-insensitive substring first; a unique match skips the prompt
hf-fm inspect little-lake-studios/demoncore-flux fluxV13 --pick --dtypes

# Suppress metadata line
hf-fm inspect google/gemma-2-2b-it model-00001-of-00002.safetensors --no-metadata

# Per-dtype summary (tensor count, params, size per dtype)
hf-fm inspect google/gemma-2-2b-it model-00001-of-00002.safetensors --dtypes

# Dtype summary for a subset of tensors
hf-fm inspect google/gemma-2-2b-it model-00001-of-00002.safetensors --dtypes --filter "layers.0"

# Dtype summary as JSON (for scripting / cross-model aggregation)
hf-fm inspect google/gemma-2-2b-it model-00001-of-00002.safetensors --dtypes --json

# Hierarchical tree view (numeric sibling groups auto-collapsed to [0..N])
hf-fm inspect google/gemma-2-2b-it model-00001-of-00002.safetensors --tree

# Tree view of a subset of tensors
hf-fm inspect google/gemma-2-2b-it model-00001-of-00002.safetensors --tree --filter "embed"

# Tree as JSON (tagged enum: leaf / branch / ranged)
hf-fm inspect google/gemma-2-2b-it model-00001-of-00002.safetensors --tree --json

# Show only the first 10 tensors (useful when you just want to peek)
hf-fm inspect google/gemma-2-2b-it model-00001-of-00002.safetensors --limit 10

# First 5 tensors matching a filter (JSON adds a `truncated` field so consumers detect the cap)
hf-fm inspect google/gemma-2-2b-it model-00001-of-00002.safetensors --filter "layers.0" --limit 5 --json

# Inspect a PEFT adapter repo (auto-detects adapter_config.json)
hf-fm inspect some-user/llama-2-7b-lora-adapter

# Will it fit on my GPU? (device 0 by default; pass --check-gpu N to pick another)
hf-fm inspect meta-llama/Llama-3.2-1B --cached --check-gpu

# Multi-GPU box: check device 1 instead of 0
hf-fm inspect meta-llama/Llama-3.2-1B --cached --check-gpu 1

# JSON composition: gpu_check rides alongside the existing header schema
hf-fm inspect meta-llama/Llama-3.2-1B --cached --check-gpu --json

# Will it fit *with* a 32K context? Folds the KV cache into the verdict (weights + KV)
hf-fm inspect meta-llama/Llama-3.2-3B --cached --check-gpu --context 32768

# Inspect a cached .gguf file (v0.10.2+, anamnesis-powered; remote .gguf planned for v0.11)
hf-fm inspect bartowski/Mistral-7B-Instruct-v0.3-GGUF Mistral-7B-Instruct-v0.3-Q4_K_M.gguf --cached
```

## Diff examples

Gated repos get the same 401/403 diagnosis as `inspect` (v0.10.5): each side is fetched separately, so the error names exactly which repo needs its license accepted.

```sh
# Compare tensor layouts between two model variants (cache-first)
hf-fm diff RedHatAI/Llama-3.2-1B-Instruct-FP8 casperhansen/llama-3.2-1b-instruct-awq

# Cache-only (no network)
hf-fm diff RedHatAI/Llama-3.2-1B-Instruct-FP8 casperhansen/llama-3.2-1b-instruct-awq --cached

# Filter to specific layers
hf-fm diff RedHatAI/Llama-3.2-1B-Instruct-FP8 casperhansen/llama-3.2-1b-instruct-awq --filter "layers.0"

# Quick summary (counts only, no tensor listing)
hf-fm diff RedHatAI/Llama-3.2-1B-Instruct-FP8 casperhansen/llama-3.2-1b-instruct-awq --cached --summary

# Per-dtype histograms side-by-side, with Δ Size column (ideal for scaled-sibling pairs)
hf-fm diff openai/gpt-oss-20b openai/gpt-oss-120b --dtypes

# Cap each section (only-A / only-B / differ) to the first 5 rows, like inspect --limit
hf-fm diff openai/gpt-oss-20b openai/gpt-oss-120b --cached --limit 5

# JSON output for programmatic consumption (includes byte_count on every tensor entry)
hf-fm diff RedHatAI/Llama-3.2-1B-Instruct-FP8 casperhansen/llama-3.2-1b-instruct-awq --cached --json
```

## Disk usage examples

```sh
# Show all cached repos sorted by size (numbered)
hf-fm du

# Drill into the 2nd largest repo by index
hf-fm du 2

# Show per-file breakdown for a specific repo
hf-fm du google/gemma-2-2b-it

# Show last-modified age column
hf-fm du --age

# Hierarchical tree of every cached repo + its files
hf-fm du --tree

# Tree view with last-modified column on each repo branch
hf-fm du --tree --age
```

## Du flags

| Flag | Description | Default |
|------|-------------|---------|
| `--age` | Show a last-modified age column (e.g., `2 days ago`, `3 months ago`) | off |
| `--json` | Output disk usage as JSON. Flat: `{repos:[{repo_id,size,file_count,has_partial,last_modified}],total_bytes,total_files,repo_count}` (`last_modified` is Unix epoch seconds, regardless of `--age`); `--tree` nests a `files` array per repo; `du <REPO_ID> --json` emits the per-file drill-down. | off |
| `--tree` | Hierarchical tree view: repos as branches, files as leaves, using box-drawing connectors. Composes with `--age` and `--json`; conflicts with the positional repo argument (the per-repo view is already covered by `du <REPO_ID>`). | off |

A repo with an in-progress or interrupted download carries a leading `●` marker in the `du` listing (`● = partial downloads`); run `hf-fm status <REPO_ID>` for the per-file breakdown.

## Other commands

```sh
# Check download status (per-repo or entire cache)
hf-fm status RWKV/RWKV7-Goose-World3-1.5B-HF
hf-fm status

# Re-evaluate "MISSING" through a preset's glob list — `.gitattributes` and `README.md`
# read `excluded` instead of `MISSING` for a `--preset safetensors` cache.
hf-fm status RWKV/RWKV7-Goose-World3-1.5B-HF --preset safetensors

# Scriptable JSON (jq-pipeable): disk budget and per-repo completeness
hf-fm du --json | jq '.total_bytes'
hf-fm du --tree --json                     # nested: files[] per repo
hf-fm status --json | jq '.repos[] | select(.has_partial)'
hf-fm status RWKV/RWKV7-Goose-World3-1.5B-HF --json

# List model families in local cache
hf-fm list-families

# Discover new families from HuggingFace Hub
hf-fm discover

# Discover families restricted to a specific tag (e.g. bitsandbytes, gguf)
hf-fm discover --tag bitsandbytes
```

## Cache commands

```sh
# Remove all partial downloads (interactive prompt)
hf-fm cache clean-partial

# Remove partials for a specific repo (by name or index)
hf-fm cache clean-partial meta-llama/Llama-3.2-1B
hf-fm cache clean-partial 29

# Preview what would be removed
hf-fm cache clean-partial --dry-run

# Skip confirmation prompt
hf-fm cache clean-partial --yes
```

## Cache clean-partial flags

| Flag | Description | Default |
|------|-------------|---------|
| `--dry-run` | Preview what would be removed without deleting | off |
| `--yes` | Skip confirmation prompt | off |

```sh
# Delete a cached model (interactive prompt)
hf-fm cache delete EleutherAI/pythia-1.4b

# Delete by numeric index from du output
hf-fm cache delete 3

# Skip confirmation prompt
hf-fm cache delete 3 --yes
```

## Cache delete flags

| Flag | Description | Default |
|------|-------------|---------|
| `--yes` | Skip confirmation prompt | off |

```sh
# Evict every repo last touched more than 30 days ago
hf-fm cache gc --older-than 30

# Trim the cache to fit under a budget (oldest-first)
hf-fm cache gc --max-size 20GiB

# Combined: age first, then trim further if still over budget
hf-fm cache gc --older-than 30 --max-size 20GiB

# Protect specific repos from eviction (repeatable)
hf-fm cache gc --max-size 20GiB --except google/gemma-2-2b-it

# Preview without deleting; show every kept repo for transparency
hf-fm cache gc --older-than 30 --dry-run --list-kept

# Skip the confirmation prompt
hf-fm cache gc --older-than 30 --yes
```

## Cache gc flags

| Flag | Description | Default |
|------|-------------|---------|
| `--older-than DAYS` | Evict repos with mtime older than this many days | unset |
| `--max-size SIZE` | Hard cap on total cache size (`B`, `KiB`, `MiB`, `GiB`, `TiB`) | unset |
| `--except REPO_ID` | Repository to protect from eviction (repeatable) | none |
| `--dry-run` | Preview the eviction plan without deleting anything | off |
| `--yes` | Skip the confirmation prompt | off |
| `--list-kept` | List every kept repo in the preview (default: hidden for terseness) | off |

At least one of `--older-than` or `--max-size` is required. When both are set, age eviction runs first; if the cache is still over budget, oldest non-protected repos are evicted next, oldest first. Repos with active partial downloads (mtime within the last hour) are skipped to avoid racing with `hf-fm download`; run `cache clean-partial` first to clear stale partials.

Decimal-prefixed size suffixes (`KB`, `MB`, `GB`, `TB`) are rejected — `hf-fm` reports sizes in binary units everywhere else and silent reinterpretation would mislead. Use `KiB`, `MiB`, `GiB`, `TiB`.

```sh
# Print snapshot path for shell substitution
hf-fm cache path google/gemma-2-2b-it

# By numeric index from du output
hf-fm cache path 2

# Use in shell scripts
cd $(hf-fm cache path google/gemma-2-2b-it)

# Pin to a specific branch / tag / commit SHA
hf-fm cache path google/gemma-2-2b-it --revision v1.0
```

```sh
# Re-verify SHA256 digests of cached files (requires network)
hf-fm cache verify google/gemma-2-2b-it

# By numeric index from du output
hf-fm cache verify 2

# Verify a specific revision
hf-fm cache verify google/gemma-2-2b-it --revision v1.0
```

## Cache verify flags

| Flag | Description | Default |
|------|-------------|---------|
| `--revision` | Git revision (branch, tag, SHA) | main |
| `--token` | Auth token (or set `HF_TOKEN` env var) | — |

`cache verify` fetches the expected SHA256 digests from the HuggingFace API and recomputes each cached file's digest locally. Per-file outcomes:

- `SHA256 OK` — the cached file matches the expected digest.
- `SHA256 MISMATCH` — the cached file's digest differs (corruption); both expected and actual hashes are printed for forensics.
- `no LFS hash` — the file has no LFS metadata (small git-stored files such as `config.json`); verification is skipped.
- `MISSING` — the file is listed remotely but not present in the local snapshot.

Exit code is non-zero only when at least one file mismatched; `skipped` and `missing` alone are non-failures (a partial cache is a legitimate state). This makes the command safe to compose into CI / cron-style integrity checks.

## Diff flags

| Flag | Description | Default |
|------|-------------|---------|
| `--cached` | Cache-only mode: fail if files are not cached locally | off |
| `--dtypes` | Show side-by-side per-dtype histograms instead of the per-tensor body (conflicts with `--summary`) | off |
| `--filter` | Show only tensors whose name contains this substring (case-insensitive) | — |
| `--json` | Output the full diff as JSON (per-tensor entries include `byte_count`; `--dtypes` adds a `dtype_histograms` field; `--limit` adds a `truncated` object) | off |
| `--limit` | Show only the first N tensors **per section** (only-A / only-B / differ), applied after `--filter`; the summary keeps true counts and `--json` adds a per-section `truncated {shown, total}` object | — |
| `--revision-a` | Git revision for model A | main |
| `--revision-b` | Git revision for model B | main |
| `--summary` | Show only the summary line (counts per category; conflicts with `--dtypes`) | off |
| `--token` | Auth token (or set `HF_TOKEN` env var) | — |

## Download flags

These flags apply to the default download command (`hf-fm <REPO_ID>`). `download-file` shares the performance and timeout flags (`--chunk-threshold-mib`, `--concurrency`, `--connections-per-file`, `--timeout-per-file-secs`, `--timeout-total-secs`), `--flat`, and `--dry-run`, but not `--filter` or `--preset`. `download-file` also accepts glob patterns (e.g., `"pytorch_model-*.bin"`) as the filename argument; with `--dry-run` it previews the matched file(s) and byte totals (a non-matching explicit filename errors; a non-matching glob prints a notice and exits 0) without downloading.

| Flag | Description | Default |
|------|-------------|---------|
| `-v`, `--verbose` | Enable download diagnostics (plan, per-file decisions, throughput) | off |
| `--dry-run` | Preview what would be downloaded (no actual download) | off |
| `--chunk-threshold-mib` | Min file size (MiB) for multi-connection download | auto-tuned |
| `--concurrency` | Parallel file downloads | auto-tuned |
| `--connections-per-file` | Parallel HTTP connections per large file | auto-tuned |
| `--exclude` | Exclude glob pattern (repeatable) | none |
| `--filter` | Include glob pattern (repeatable) | all files |
| `--flat` | Copy files to flat layout: `{output-dir}/{filename}` | off |
| `--output-dir` | Custom output directory (or flat copy target with `--flat`) | HF cache |
| `--preset` | Filter preset: `safetensors`, `gguf`, `npz`, `pth`, `config-only` | — |
| `--revision` | Git revision (branch, tag, SHA) | main |
| `--timeout-per-file-secs` | Per-file transfer timeout, in seconds. The ceiling on any single file. Raise it for large files on slow links — at ~10 MiB/s the 300 s default caps progress at roughly 3 GiB, so try `1800` for files in the 5–15 GiB range. | 300 |
| `--timeout-total-secs` | Overall wall-clock budget for the whole invocation, in seconds (including retries and, since v0.10.5, in-flight files — not just a between-files check). Independent of `--timeout-per-file-secs`; the effective cap on any file is whichever of the two elapses first. Applies to `download-file`'s single file too. | no limit |
| `--token` | Auth token (or set `HF_TOKEN` env var) | — |

## List-files flags

| Flag | Description | Default |
|------|-------------|---------|
| `--exclude` | Exclude glob pattern (repeatable) | none |
| `--filter` | Include glob pattern (repeatable) | all files |
| `--json` | Output the file list as JSON: `{repo_id, files[{filename, size, sha256}], total_bytes, file_count}` — full (untruncated) SHA256 regardless of `--no-checksum`; adds per-file `cached` and a `cached_count` with `--show-cached` | off |
| `--no-checksum` | Suppress the SHA256 column (human table only; `--json` always carries the full digest) | off |
| `--preset` | Filter preset: `safetensors`, `gguf`, `npz`, `pth`, `config-only` | — |
| `--revision` | Git revision (branch, tag, SHA) | main |
| `--show-cached` | Show cache status: complete (✓), partial, or missing (✗) | off |
| `--token` | Auth token (or set `HF_TOKEN` env var) | — |

## Search flags

| Flag | Description | Default |
|------|-------------|---------|
| `--exact` | Match a full repository ID exactly and show its metadata card | off |
| `--library` | Filter by library framework (e.g., `transformers`, `peft`, `vllm`) | — |
| `--limit` | Maximum number of results | 20 |
| `--pipeline` | Filter by pipeline task (e.g., `text-generation`, `text-classification`) | — |
| `--tag` | Filter by model tag (e.g., `gguf`, `conversational`, `imatrix`) | — |
| `--show` | Comma-separated columns to add: `tags` (free; from the existing API payload), `size` (one extra HTTP request per result, bounded to 8 concurrent). | — |

## List-families flags

| Flag | Description | Default |
|------|-------------|---------|
| `--show` | Comma-separated columns to add. Currently only `quant` — reads `quantization_config.quant_method` from each repo's cached `config.json`, falling back to `gguf` when any cached file ends in `.gguf`. | — |
| `--tag` | Filter cached repos by a HuggingFace tag (case-insensitive). Tags are fetched at query time via the HF model_info API, bounded to 8 concurrent requests. Per-repo fetch failures silently drop the row from the filter result. Empty families are pruned from the output. | — |

## Status flags

| Flag | Description | Default |
|------|-------------|---------|
| `--json` | Output the status report as JSON. All-repos: `{repos:[{repo_id,file_count,size,has_partial}],model_count}`. Per-repo: `{repo_id,revision,commit_hash,cache_path,files:[{filename,state,local_size?,expected_size?}],summary:{total,complete,partial,missing,excluded}}`, where `state` is `complete` / `partial` / `missing` / `excluded`. | off |
| `--preset` | Re-evaluate which remote files are deliberate skips. Files not matching this preset's glob list (`safetensors`, `gguf`, `npz`, `pth`, `config-only`) are reported as `excluded` instead of `MISSING`. Overrides the value persisted in `.hf-fm-snapshot.json` by `download --preset`. | sidecar value (or none) |
| `--revision` | Git revision (branch, tag, SHA) | main |
| `--token` | Auth token (or set `HF_TOKEN` env var) | — |

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
| `--check-gpu [N]` | Append a one-line GPU-fit verdict comparing model weight bytes against free VRAM on device `N` (default `0`). Reads device info via [`hypomnesis`](https://crates.io/crates/hypomnesis) (NVML on Linux/Windows, DXGI on Windows; falls back to `nvidia-smi`). On systems with no NVIDIA GPU detected, prints `GPU N: unavailable — <reason>` and skips the verdict (exit code stays `0` — the command is informational, not a gate). Uses the **unfiltered** model totals (so `--filter` / `--limit` affect only the printed table). Composes with `--json`: a `gpu_check` object is added to the per-file schema, the `--tree --json` schema, and the `--dtypes --json` schema; the repo-level plain `--json` schema becomes `{"files": [...], "gpu_check": {...}}` when `--check-gpu` is passed (the array schema is preserved when it is absent). At the whole-repo level, forces shard aggregation so the verdict reflects the total weight bytes across every shard. Conflicts with `--list` (no headers are read in `--list` mode). | off |
| `--context N` | KV-cache context length for the `--check-gpu` verdict (**requires `--check-gpu`**). Reads the model's `config.json`, computes the KV cache at sequence length `N`, and measures fit against `weights + KV` instead of weights alone — adding `KV cache @ ctx=N` and `Total` lines. Parameter-driven and architecture-aware: GQA, sliding-window (Gemma / Mistral, with mixed local/global blended), and hybrid Mamba/attention (Granite-4, Nemotron-H, Bamba, Qwen3-Next — a separate `Recurrent state` line for the Mamba2 state). MLA (DeepSeek) is **skipped** with a note; an absent / dimension-less `config.json` prints `KV cache: unavailable` and falls back to weights-only (exit code stays `0`). KV element size is the activation dtype (`torch_dtype`, bf16/fp16 = 2 B). Composes with `--json` (the `gpu_check` object gains a `kv_cache` sub-object and `model.total_bytes`). | — |
| `--dtypes` | Show a per-dtype summary (tensor count, params, size) instead of individual tensors. Composes with `--json` to emit `{ dtypes: [...], total_tensors, total_params }`. | off |
| `--filter` | Show only tensors whose name contains this substring (case-insensitive) | — |
| `--json` | Output the full header as JSON instead of a human-readable table | off |
| `--limit` | Show only the first N tensors (applied after `--filter`). JSON output gains a `truncated` field when the cap is reached. | — |
| `--list` | List the repo's supported tensor files (`.safetensors` / `.gguf` / `.npz` / `.pth`) as a numbered table (filename + size) and exit — no headers read. The `#` column doubles as the `FILENAME` argument on a follow-up run (`hf-fm inspect <repo> 3`); indices are alphabetical and stable while the repo does not change remotely (pin `--revision <sha>` on both sides to lock the view). Conflicts with `FILENAME`, the rendering flags, and `--pick`. | off |
| `--no-metadata` | Suppress the `Metadata:` line in human-readable output | off |
| `--pick` | Pick the file to inspect interactively from a numbered list (v0.10.5+). With no `FILENAME`, offers every supported tensor file; with a `FILENAME`, treats it as a **case-insensitive substring** filter — a unique match auto-resolves (with a `Resolving to <name>` note on stderr), several matches prompt `Pick [1..N]:` on stderr. Under `--pick` the positional is never a numeric index. Requires an interactive terminal (stdin + stderr); the prompt goes to stderr, so `--json` stdout can be redirected. Empty input cancels with a non-zero exit. Composes with every rendering flag; conflicts with `--list`. | off |
| `--tree` | Show a hierarchical tree view grouped by dotted namespace prefix; numeric sibling groups with identical sub-structure collapse to `[0..N]`. Composes with `--filter` and `--json`. Conflicts with `--dtypes` and `--limit`. | off |
| `--revision` | Git revision (branch, tag, SHA) | main |
| `--token` | Auth token (or set `HF_TOKEN` env var). Required for gated repos, together with an accepted license — each gated family (Llama 3.1 vs 3.2, …) is licensed separately. | — |

## General flags

| Flag | Description |
|------|-------------|
| `-h`, `--help` | Print help |
| `-V`, `--version` | Print version |

Subcommands accept their own flags. Run `hf-fm <command> --help` for details.
