# Cache Management Roadmap

**Date:** March 26, 2026
**Status:** Proposed
**Context:** hf-fetch-model can download, inspect, and search models, but managing the local HuggingFace cache remains manual. Users accumulate tens of GBs across dozens of models and have no way to reclaim space, clean up failed downloads, or even find where a model is stored — short of navigating the opaque `models--org--name` directory structure by hand.

---

## The pain

The HuggingFace cache (`~/.cache/huggingface/hub/`) grows silently. Every `hf-fm <REPO_ID>` adds a `models--org--name/` directory that persists indefinitely. There is no expiry, no size limit, no garbage collection. Users discover the problem when their disk fills up, then face a manual cleanup: figure out which `models--` directories correspond to which repos, guess which ones are safe to delete, and hope they don't break symlinks or leave orphaned refs.

hf-fetch-model already provides *visibility* (`du`, `status`, `list-files --show-cached`) and *integrity checking* infrastructure (`checksum.rs`). What's missing is the ability to **act** — delete, clean, verify, and organize the cache from the CLI.

---

## What Python's `huggingface-cli` provides

Python's `huggingface_hub` CLI offers cache management via `hf cache`:

| Command | Description |
|---------|-------------|
| `hf cache ls` | List cached repos with disk usage |
| `hf cache ls --revisions` | List cached revisions per repo |
| `hf cache rm <REVISION_HASH>` | Remove a specific revision by hash |
| `hf cache prune` | Remove unreferenced revisions (detached snapshots with no ref pointing to them) |
| `hf cache verify` | Verify cached file checksums |

**Gaps in the Python CLI** (from GitHub issues [#1065](https://github.com/huggingface/huggingface_hub/issues/1065) and [#2219](https://github.com/huggingface/huggingface_hub/issues/2219)):
- No `--older-than` age-based eviction
- No per-file deletion within a revision (all-or-nothing per revision hash)
- `rm` requires the opaque revision hash, not the human-readable repo ID
- No `--dry-run` on `rm`
- No `--except` to protect specific repos during bulk cleanup
- No dataset cache scanning (`scan-cache` only covers models — [#2218](https://github.com/huggingface/huggingface_hub/issues/2218))

hf-fetch-model can do better on all of these.

---

## Proposed features

### Tier 1 — Essential

#### `hf-fm cache delete <REPO_ID> [--yes]`

The most common operation: remove a specific cached model by its human-readable repo ID. Shows size before deleting, prompts for confirmation unless `--yes` is passed.

```
$ hf-fm cache delete EleutherAI/pythia-1.4b
  EleutherAI/pythia-1.4b  (2.80 GiB, 8 files)
  Delete? [y/N] y
  Deleted. Freed 2.80 GiB.
```

**Why not just `rm -rf`?** The HF cache uses symlinks and ref files. Naive deletion can leave orphaned refs or break other revisions of the same model. `cache delete` removes the entire `models--org--name/` directory cleanly.

**Implementation:** `cache::hf_cache_dir()` + `chunked::repo_folder_name()` to locate the directory, `cache_repo_usage()` for the size preview, `std::fs::remove_dir_all()` for deletion.

#### `hf-fm cache clean-partial`

Remove `.chunked.part` files and incomplete snapshots left by interrupted downloads. Safe and non-destructive to complete downloads.

```
$ hf-fm cache clean-partial
  Found 2 partial downloads:
    google/gemma-2-2b-it: model-00002-of-00002.safetensors.chunked.part  (1.30 GiB)
    meta-llama/Llama-3.2-1B: model.safetensors.chunked.part             (0.45 GiB)
  Clean? [y/N] y
  Removed 2 partial files. Freed 1.75 GiB.
```

**Implementation:** Walk the cache directory, identify `.chunked.part` files. The `has_partial` flag in `CachedModelSummary` already detects these at the repo level.

### Tier 2 — Quality of life

#### `hf-fm cache gc --older-than <DAYS>`

Age-based eviction — remove models not accessed in N days. Addresses the silent accumulation problem. Python's CLI has no equivalent.

```
$ hf-fm cache gc --older-than 30
  Will remove:
    EleutherAI/pythia-1.4b  (2.80 GiB, last accessed 45 days ago)
    RWKV/RWKV7-Goose-0.1B  (0.21 GiB, last accessed 62 days ago)
  Keep:
    google/gemma-2-2b-it    (5.10 GiB, accessed 2 days ago)
  Proceed? [y/N] y
  Removed 2 repos. Freed 3.01 GiB.
```

**Flags:** `--yes` (skip prompt), `--except <REPO_ID>` (protect specific repos), `--dry-run` (preview only).

**"Last accessed" heuristic:** Use the most recent modification time among files in the snapshot directory. This is an approximation (the HF cache doesn't track access times explicitly) but matches how `huggingface-cli` determines staleness.

#### `hf-fm cache verify [REPO_ID]`

Re-verify SHA256 checksums of cached files against HF LFS metadata. Detailed design already exists in the [v0.10.0 roadmap](v0.10.0-roadmap.md). Non-destructive, requires network to fetch expected hashes.

### Tier 3 — Scripting and convenience

#### `hf-fm cache path <REPO_ID>`

Print the snapshot directory path for a cached model. Useful for scripting and shell integration.

```
$ hf-fm cache path google/gemma-2-2b-it
/home/user/.cache/huggingface/hub/models--google--gemma-2-2b-it/snapshots/abc1234def5678

$ cd $(hf-fm cache path google/gemma-2-2b-it)
```

**Implementation:** `hf_cache_dir()` + `repo_folder_name()` + `read_ref()` to resolve the snapshot path. Returns non-zero exit code if the repo is not cached.

#### `hf-fm cache list [--sort size|age|name]`

List cached repos with last-access timestamps and sort options. Essentially `du` + `status` combined, with timestamp visibility. Could eventually replace the no-arg `status` and `du` commands.

```
$ hf-fm cache list --sort age
  REPO                                              SIZE        LAST ACCESS   FILES
  RWKV/RWKV7-Goose-0.1B                            0.21 GiB    62 days ago       3
  EleutherAI/pythia-1.4b                            2.80 GiB    45 days ago       8
  google/gemma-2-2b-it                              5.10 GiB     2 days ago       8
```

---

## Subcommand grouping

These features naturally group under a `cache` subcommand:

```
hf-fm cache delete <REPO_ID>
hf-fm cache clean-partial
hf-fm cache gc --older-than 30
hf-fm cache verify <REPO_ID>
hf-fm cache path <REPO_ID>
hf-fm cache list
```

The existing `du` and `status` commands could become aliases for `cache du` and `cache status` in a future version, with the top-level names kept for backwards compatibility.

**Implementation note:** Clap supports nested subcommands via `#[command(subcommand)]` on an enum field. Add a `Cache` variant to `Commands` with its own `CacheCommands` enum.

---

## Relationship to v0.10.0 roadmap

The [v0.10.0 roadmap](v0.10.0-roadmap.md) covers `verify`, `clean` (with `--older-than`, `--repo`, `--all`, `--except`), and `du --tree`. This roadmap extends that vision with:

- **`cache delete`** — simpler single-repo deletion (v0.10.0's `clean --repo` covers this but `cache delete` is more discoverable)
- **`cache clean-partial`** — targeted partial-download cleanup (v0.10.0's `clean` removes entire repos, not individual partial files)
- **`cache path`** — scripting helper (not in v0.10.0)
- **`cache list`** — unified listing with timestamps (not in v0.10.0)
- **Subcommand grouping** — architectural direction for organizing cache operations

These can be implemented incrementally across releases, starting with `cache delete` and `cache clean-partial` which address the most immediate pain.

---

## Implementation order

| Priority | Feature | Rationale |
|----------|---------|-----------|
| 1 | `cache delete` | Most requested operation. Simple, low risk, high value. |
| 2 | `cache clean-partial` | Safe cleanup of interrupted downloads. Unblocks re-downloads. |
| 3 | `cache verify` | Non-destructive integrity check. Detailed design in v0.10.0 roadmap. |
| 4 | `cache gc` | Age-based eviction. Requires the "last accessed" heuristic. |
| 5 | `cache path` | Small utility, useful for scripting. |
| 6 | `cache list` | Combines existing `du` and `status` with timestamps. |
