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

The Python CLI was revamped in v1.0/v1.1 (2025–2026, see [#1065](https://github.com/huggingface/huggingface_hub/issues/1065) and [PR #3439](https://github.com/huggingface/huggingface_hub/pull/3439)), adding sorting options and improved commands. Some long-standing gaps remain:
- `rm` operates on revision hashes, not human-readable repo IDs
- No `--older-than` age-based eviction
- No `--except` to protect specific repos during bulk cleanup
- No dataset cache scanning ([#2218](https://github.com/huggingface/huggingface_hub/issues/2218) — open since April 2024)

hf-fetch-model can offer a simpler, repo-ID-based UX with age-based eviction and partial-download cleanup.

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

#### `hf-fm cache gc --older-than <DAYS>` / `--max-size <SIZE>`

Two eviction strategies — by age or by budget. Users think in either "what haven't I used recently?" or "how much space can I spare?". Inspired by Cargo's `cargo clean gc --max-download-size=1GiB` ([Rust Blog](https://blog.rust-lang.org/2023/12/11/cargo-cache-cleaning/)).

**Age-based:** remove models not accessed in N days.
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

**Budget-based:** delete oldest repos until total cache size is under the target.
```
$ hf-fm cache gc --max-size 5GiB
  Cache is 9.31 GiB, target 5.00 GiB — need to free 4.31 GiB.
  Will remove (oldest first):
    RWKV/RWKV7-Goose-0.1B  (0.21 GiB, last accessed 62 days ago)
    EleutherAI/pythia-1.4b  (2.80 GiB, last accessed 45 days ago)
    google/gemma-scope-2b-pt-res  (1.20 GiB, last accessed 30 days ago)
  Proceed? [y/N] y
  Removed 3 repos. Freed 4.21 GiB. Cache now 5.10 GiB.
```

Both strategies can be combined: `--older-than 30 --max-size 20GiB` removes repos older than 30 days *and* trims further if still over budget.

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

**Limitation:** resolves the `main` ref only. Repos downloaded at a non-default revision (e.g., `--revision some-branch`) are not resolved. A future `--revision` flag will address this.

**Note:** `cache list` was considered as a separate command but dropped in favor of consolidating all cache visibility into `du` with progressive flags (`--age`, `--tree`). See the [du extensions roadmap](hf-fetch-model-du-extensions-roadmap.md) for details. One command to learn, fewer to remember.

---

## Subcommand grouping

Action commands group under a `cache` subcommand. Visibility stays in `du`:

```
# Seeing (du — one command, progressive flags)
hf-fm du                          # numbered list with partial markers
hf-fm du <N>                      # drill into Nth repo
hf-fm du --age                    # add last-access column
hf-fm du --tree                   # structural view

# Acting (cache — destructive operations)
hf-fm cache delete <REPO_ID|N>
hf-fm cache clean-partial
hf-fm cache gc --older-than 30
hf-fm cache gc --max-size 20GiB
hf-fm cache verify <REPO_ID>
hf-fm cache path <REPO_ID>
```

`du` for **seeing**, `cache` for **acting**. No `cache list` — `du` covers all visibility needs. Inspired by Docker's `docker system df` + `docker system prune` separation.

**Implementation note:** Clap supports nested subcommands via `#[command(subcommand)]` on an enum field. Add a `Cache` variant to `Commands` with its own `CacheCommands` enum.

---

## Relationship to v0.10.0 roadmap

The [v0.10.0 roadmap](v0.10.0-roadmap.md) covers `verify`, `clean` (with `--older-than`, `--repo`, `--all`, `--except`), and `du --tree`. This roadmap extends that vision with:

- **`cache delete`** — simpler single-repo deletion (v0.10.0's `clean --repo` covers this but `cache delete` is more discoverable)
- **`cache clean-partial`** — targeted partial-download cleanup (v0.10.0's `clean` removes entire repos, not individual partial files)
- **`cache gc --max-size`** — budget-based eviction (inspired by Cargo's `cargo clean gc --max-download-size`)
- **`cache path`** — scripting helper (not in v0.10.0)
- **`du` consolidation** — all cache visibility in `du` with `--age` and `--tree` flags, no separate `cache list`
- **Subcommand grouping** — `du` for seeing, `cache` for acting (inspired by Docker's `df` + `prune` separation)

These can be implemented incrementally across releases, starting with `cache delete` and `cache clean-partial` which address the most immediate pain.

---

## Release plan

Incremental delivery across patch and minor releases. Ship the highest-impact features first in small releases, reserve the minor bump for the more complex features that need the "last accessed" heuristic and network-based verification.

### v0.9.2 — CLI ergonomics (dogfooding) ✓

| Feature | Scope |
|---------|-------|
| Version in `--help` | `hf-fm --help` now shows the version number in the header |
| `--preset pth` | Filter preset for PyTorch `.bin` weight files (`pytorch_model*.bin`) |
| Glob in `download-file` | `hf-fm download-file org/model "pytorch_model-*.bin"` expands globs |
| `--flat` download flag | Copies files to flat `{output-dir}/{filename}` layout after download |

Addresses immediate friction from dogfooding. Ships fast, no architectural changes. **Shipped.**

### v0.9.3 — Immediate pain relief ✓

| Feature | Scope |
|---------|-------|
| `cache delete` | Single-repo deletion by repo ID or numeric index, confirmation prompt, `--yes` flag |
| `cache clean-partial` | Remove `.chunked.part` files from interrupted downloads, `--dry-run`, `--yes` |
| `du` numbered indexing | Add `#` column, accept `du <N>` for drill-down, `●` partial marker, dynamic column width |

Also shipped in v0.9.3 beyond the original plan: gated model pre-flight check, `du` cache path header, `candle_inspect` example, cache layout verification test. **Shipped.**

### v0.9.4 — Scripting and visibility ✓

| Feature | Scope |
|---------|-------|
| `cache path` | Print snapshot directory path for scripting |
| `du --age` | Add last-modified age column to `du` output (replaces the dropped `cache list`) |

Also shipped in v0.9.4 beyond the original plan: `--tag` search flag, binary name fix, em-dash legend, `--exact` help text, search cross-references, algorithmic fixes (targeted cache scans, BTreeSet dedup, filter-before-clone, TiB formatting). **Shipped.**

### v0.9.5 — Library hardening ✓

| Feature | Scope |
|---------|-------|
| Cache layout centralization | Extract all hf-hub cache path construction (`models--org--name`, `snapshots/`, `refs/`, `blobs/`) into a single `cache_layout.rs` module that delegates to `hf_hub::Repo::folder_name()`, `CacheRepo::pointer_path()`, etc. Replaces ~15 scattered `format!("models--{}", ...)` call sites across `chunked.rs`, `cache.rs`, `download.rs`, and `main.rs`. One file to audit when hf-hub bumps. |
| Watch-based progress channel | Add a `tokio::sync::watch::Receiver<ProgressEvent>` API alongside the existing `Arc<dyn Fn>` callback. Async GUI/TUI consumers can `.changed().await` on the receiver instead of bridging the callback into a channel themselves. The existing callback API stays unchanged (backward compatible). |

Also shipped in v0.9.5 beyond the original plan: 8 audit fixes (chunked download timeout, TCP connect timeout, Windows blob corruption, POSIX symlink TOCTOU race, inspect task cancellation, blocking I/O moved to `spawn_blocking`, CDN URL expiry detection with re-probe, temp file RAII cleanup), shared HTTP client for `list_repo_files_with_metadata`, `parse_header_json` zero-clone iteration, `check_disk_space` cache scan removal. **Shipped.**

### v0.10.0 — Cache maturity

| Feature | Scope |
|---------|-------|
| `cache verify` | SHA256 re-verification against HF LFS metadata (requires network). Detailed design in [v0.10.0 roadmap](v0.10.0-roadmap.md). |
| `cache gc` | Age-based (`--older-than`) and budget-based (`--max-size`) eviction, with `--except`, `--dry-run`. Requires the "last accessed" heuristic. |
| `du --tree` | Tree-view of cache directory structure with box-drawing characters. |

These are more complex: verify needs network + checksum comparison, gc needs the last-accessed heuristic and interactive prompt safety, `du --tree` is a display feature that benefits from the `du --age` timestamps added in v0.9.4. Bundle them as the "cache maturity" release.
