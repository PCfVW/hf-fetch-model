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

### v0.9.6 — Inspect discoverability ✓ ✓

| Feature | Scope |
|---------|-------|
| Dynamic table column widths | All 12 CLI tables (`inspect`, `diff`, `list-files`, `status`, `du`, `search`, `list-families`, `discover`, `--dry-run`, shard summary, multi-file summary, cache status) compute widths from actual data instead of hardcoded values. Fixes misalignment for long tensor/file names (e.g., multimodal models with `model.vision_tower.encoder.layers.*.mlp.*` prefixes). |
| `inspect --dtypes` | Per-dtype summary (tensor count, params, bytes) instead of individual tensors. Composes with `--filter` for subset breakdowns. |
| `inspect --dtypes --json` | Compact JSON schema (`{ dtypes: [...], total_tensors, total_params }`) for cross-model scripting (e.g., "find all cached models with FP8 tensors"). |
| `inspect --limit N` | Truncates tensor list to first N entries (applied after `--filter`). Solves the "wall of JSON" problem; the `--json` output gains a `truncated: { shown, total }` field so consumers can detect incomplete output. Schema-identical to v0.9.5 when not truncated. |
| `inspect --tree` | Hierarchical view grouped by dotted namespace. Single-child chains merge into one line; numeric sibling groups with structurally-identical sub-trees collapse to `layers.[0..N]   (×K)` with the template shown once. Composes with `--filter` and `--json` (tagged-enum schema: `leaf` / `branch` / `ranged`). Conflicts with `--dtypes` and `--limit` (clap parse-time rejection). |
| Cargo + GitHub description | Updated to reflect inspect/diff scope, not just downloads ("Download, inspect, and compare HuggingFace models from Rust..."). |

**Theme: structural discovery.** The HuggingFace ecosystem has many tools for *aggregating* (param counts, dtype distributions) but few for *exploring* (where do these tensors live? what's the namespace structure?). The `--tree` view in particular surfaces architectural variation (e.g., Gemma 4's per-layer shape differences) that flat listings hide.

Closest prior art: [EricLBuehler/safetensors_explorer](https://github.com/EricLBuehler/safetensors_explorer) — interactive TUI for local files. `hf-fm inspect --tree` differs by being remote-capable (HTTP Range), printable (pasteable into bug reports/comments), explicitly range-collapsing, and integrated with the same toolchain as downloads.

Critical for answering candle ecosystem issues ([#3448](https://github.com/huggingface/candle/issues/3448), [#2875](https://github.com/huggingface/candle/issues/2875)) where users are stuck because they can't see the tensor structure.

### v0.9.7 — Inspect discoverability & newbie-friendly UX ✓

| Feature | Scope |
|---------|-------|
| `inspect --list` | New discovery flag: numbered `.safetensors` table (filename + size) with `Repo:` and `Rev: <commit-sha>` header, alphabetically sorted so shards order naturally. No header parsing. Footer tip advertises the short SHA to pass via `--revision` for reproducibility. Composes with `--cached` (lists local snapshot, immune to remote changes) and `--revision` (locks both `--list` and the follow-up `inspect <n>` to the same commit). |
| `inspect <repo> <n>` numeric index | Bare-integer filenames resolve against the alphabetically-sorted safetensors listing (1-based). Literal filenames unchanged. Transparency line `Resolving index 3 → <name> (repo rev: <short-sha>)` printed to stderr before the inspect proceeds, so users catch mismatches immediately. Out-of-range indices produce a clear error pointing at `--list`. |
| `inspect` clear error on unsupported file types | Dogfooding Gap 1 resolved: passing a non-safetensors file (e.g. `.npz`) now emits `hf-fm inspect supports .safetensors only (got .npz for <path>)` instead of the misleading `failed to parse header JSON: expected value at line 1 column 1`. Driven by a new `FetchError::UnsupportedInspectFormat { filename, extension }` variant; retry classification updated accordingly. |
| `--preset npz` | Dogfooding Gap 3 resolved: new `Preset::Npz` variant plus `Filter::npz()` builder helper. Matches `*.npz`, `*.npy`, `config.yaml`, `*.json`, `*.txt` — tuned to NumPy-based weight repos such as Google's GemmaScope transcoders. Wired into the default download command, `--dry-run`, `list-files`, and `warn_redundant_filters`. |
| Alphabetical `--help` command listing | `hf-fm --help` and `hf-fm cache --help` now list subcommands alphabetically at runtime (via `display_order` re-assignment in `main`). Adding a new command lands in the right place automatically regardless of enum declaration order — no reviewer discipline required. |
| `list-families` line wrapping + cache header | Each repo prints on its own line, indented under the family column (the single run-on line for large families like `llama` is gone). Output now starts with `Cache: <absolute path>`, matching the header style used by `du`. |
| `inspect --help` examples block | `after_help` footer on the `inspect` subcommand with four concrete invocations (inspect-all / --list / index / --tree) plus a note on index stability and `--revision` pinning. Highest-ROI newbie-facing change: visible the moment the user types `--help`. |
| `indicatif` bumped `0.17 → 0.18` | Zero source changes needed — every API we call (`MultiProgress`, `ProgressBar`, `ProgressStyle`) was stable across the bump. Concrete build-tree win: `hf-hub` already pulled in `indicatif 0.18`, so we were compiling it **twice**; now we compile it once. |
| `cargo update` patch bumps | Eleven semver-compatible updates (tokio, openssl, clap, clap_derive, hyper-rustls, typenum, webpki-roots, and others). Routine hygiene. |

**Theme: discoverability + cache visibility.** v0.9.6 shipped structural discovery *inside* a safetensors file (`--tree`, `--dtypes`); v0.9.7 extends the principle *outward* to the repo level — "what can I inspect in this repo?" answered by `--list`, "which cache am I looking at?" answered by the new `Cache:` header on `list-families`, "which commit am I picking from?" answered by `Rev:` on `inspect --list`. The pattern unifies as: **every discovery command announces its scope before showing results**, so the user never has to guess at provenance.

**New public APIs.** Two additions to the library surface, both narrow and documented:

- `hf_fetch_model::repo::list_repo_files_with_commit(repo_id, token, revision, client) → (Vec<RepoFile>, Option<String>)` — same HTTP call as `list_repo_files_with_metadata` but also returns the resolved commit SHA. The existing `list_repo_files_with_metadata` is now a thin wrapper over this, so the seven existing callers in the crate see no breaking change.
- `hf_fetch_model::inspect::list_cached_safetensors(repo_id, revision) → (Vec<(String, u64)>, Option<String>)` — cheap name-and-size enumeration of cached safetensors, paired with the snapshot's commit SHA. Does **not** parse headers (unlike `inspect_repo_safetensors_cached`). Type-aliased as `CachedSafetensorsListing`.

**Known narrow gap surfaced.** Building the `--revision` hardening for `inspect --list` revealed a pre-existing limitation of every `--cached` code path: `cache::read_ref` looks for `refs/<revision>` on disk, which exists for branches/tags (`main`, `v1.0.0`) but **not** for raw commit SHAs — SHAs live in `snapshots/`, not `refs/`. So `--cached --revision <full-sha>` currently prints `Rev: (unknown)` and an empty listing even when the snapshot directory exists on disk. Not a regression — every existing `--cached` path has this limitation; `--list` just made it newly *visible*. ~10-line fix in `cache.rs`, deserves its own PR with tests. Detailed analysis in [`docs/dogfooding-feedbacks/hf-fm-dogfooding.md`](../dogfooding-feedbacks/hf-fm-dogfooding.md) under "Known narrow gap uncovered".

### v0.9.8 — Download durability ✓

| Feature | Scope |
|---------|-------|
| `--timeout-per-file-secs`, `--timeout-total-secs` CLI flags | Plumbs the existing `FetchConfigBuilder::timeout_per_file` / `timeout_total` builder methods through to the CLI. The 300 s default in `download.rs` is unchanged when the flag is omitted; users with slow connections on multi-GiB files can now extend it (e.g. `--timeout-per-file-secs 1800`). Wired through the default download command, `download-file`, and the `download-file` glob path via a shared `apply_timeout_overrides` helper so the per-call-site plumbing is DRY. |
| Partial state preservation on transient interruption | Inverts the `TempFileGuard` policy in `chunked.rs` from "wipe-by-default, `commit()` to keep" to "keep-by-default, `mark_corrupt()` to wipe". The intent split: transient failures (timeout-induced future drop, Ctrl-C, panic, retryable chunk error) leave the `.chunked.part` on disk and survive into the next invocation; confirmed corruption (etag/total-size/schema mismatch) wipes inline. The v0.9.5 RAII intent — clean cleanup of `.chunked.part` on JoinSet abort — is preserved through `mark_corrupt()`. |
| Resume after interruption via `.chunked.part.state` sidecar | New `chunked_state.rs` module storing per-chunk completion offsets as JSON next to the partial. `prepare_or_resume_temp_file` reuses an existing `(.chunked.part, .chunked.part.state)` pair when its `(schema_version, etag, total_size, connections)` quadruple matches the current download — bytes already downloaded are kept and each chunk sends `Range: bytes=<start+completed>-<end>` to skip them. On any invariant mismatch, both files are removed and a fresh state is written. The sidecar is updated atomically (write-tmp + rename) every 16 MiB of per-chunk progress and removed by `finalize_chunked_download` after the blob rename. End-to-end verified on the real Gemma 4 multimodal repo (9.54 GiB) at the slow-connection regime where the v0.9.7 binary was unable to complete. |
| `cache clean-partial` sweeps the new sidecars | `PartialFile::sidecar_paths()` enumerates the `.chunked.part.state` and `.chunked.part.state.tmp` siblings; `run_cache_clean_partial` removes them best-effort alongside the main partial. Without this, the sweep would leave kilobyte-sized orphans. |

**Theme: download durability.** v0.9.5 hardened the TCP/HTTP layer (TCP connect timeout, CDN expiry re-probe, TempFileGuard RAII). v0.9.8 hardens the wall-clock layer above it: configurable per-file and total budgets, preserved partial state on interruption, and resumeable downloads across invocations. The user-visible contract becomes: *if you see bytes on disk, they survive — and the next invocation continues where you left off, as long as the upstream file has not changed*.

**Concrete unblock.** Resolves the slow-connection lockup discovered while answering candle [#3448](https://github.com/huggingface/candle/issues/3448) (Gemma 4): a 9.54 GiB safetensors file at ~7 MiB/s effective throughput is uncompletable under the v0.9.7 CLI's 300 s/file ceiling regardless of retries (each retry truncated the prior partial via `prepare_temp_file`'s `File::create()`). With v0.9.8, `hf-fm download-file google/gemma-4-E2B-it model.safetensors --timeout-per-file-secs 1800` succeeds in one continuous run; if interrupted, the next invocation resumes via the sidecar.

**Test surface.** 119 tests passing (was 117 in v0.9.7, +2 sidecar-paths unit tests, +13 in the new `chunked_state` module, +3 for the TempFileGuard inversion, +3 CLI tests for the new flags, balanced against the 3 existing chunked.rs tests). `clippy --features cli --all-targets -- -D warnings -W clippy::pedantic` is clean — pre-existing `too_many_lines` warnings on six orchestration functions (`run`, `run_download`, `run_diff`, `run_inspect_single`, `run_list_files`, `download_all_files_map`) annotated with `#[allow(clippy::too_many_lines)]` + EXPLICIT reason comments, framed as a deliberate Phase-1-scope decision rather than fragmented refactors of unrelated code paths.

### v0.10.0 — Cache maturity & first docs (in progress)

| Feature | Scope |
|---------|-------|
| `cache verify` ✓ | SHA256 re-verification against HF LFS metadata (requires network). Detailed design in [v0.10.0 roadmap](v0.10.0-roadmap.md). Streaming spinner progress per file (rotating `\| / - \\` ASCII bar via `indicatif`) so multi-GiB safetensors hashes give continuous liveness feedback rather than a long blank pause. |
| `cache gc` ✓ | Age-based (`--older-than`) and budget-based (`--max-size`) eviction, with `--except`, `--dry-run`, `--list-kept`. Active partial downloads (mtime within the last hour) are skipped to avoid racing with `hf-fm download`. |
| `du --tree` ✓ | Tree-view of cache directory structure with box-drawing characters. Reuses the visual style established by `inspect --tree` in v0.9.6 (same `├──`, `└──`, `│   ` connectors and dynamic column-width approach). Each cached repo renders as a branch (size + file count, optional `--age` column, partial-download marker) with its files as leaves sorted by size descending. Composes with `--age`; conflicts at clap parse-time with the positional `du <REPO>` form (the per-repo view is already covered there). |
| Cache fast-path correctness ✓ | `download_all_files_map` previously short-circuited to `Cached` whenever the snapshot directory contained any include-filter-matching file, leading to a misleading "Cached at:" message when only the small config files were on disk and `model.safetensors` was absent. The fast-path now runs *after* the remote file listing and verifies every filtered remote file resolves to a real path under the snapshot dir. Cost: one cheap HTTP listing per `hf-fm <repo>` call when the cache happens to be complete. Discovered while dogfooding v0.9.8; landed on `main` immediately so future `hf-fm <repo>` invocations are correct. |
| Pipe-eats-exit-code FAQ entry ✓ | Wrapping `hf-fm download-file ... 2>&1 \| tail -20` masks hf-fm's exit code with `tail`'s, hiding download failures. New FAQ entry under "Errors and unexpected output" explains the mechanic and gives copy-paste recipes for `${PIPESTATUS[0]}` (bash/zsh) and `$LASTEXITCODE` (PowerShell). Not a code bug — pure documentation — but worth naming because long downloads invite the `\| tail` reflex. |
| **First docs effort** ◐ | **First tutorial shipped: [Inspect before you download](../tutorials/inspect-before-downloading.md) ✓ — establishes `docs/tutorials/` and lands the inspect-without-downloading walkthrough using `zed-industries/zeta-2` as the running example, pinned to a specific SHA for reproducibility, with a closing pivot to bartowski's GGUF Q4_K_M for the "doesn't fit on a 5060 Ti" case. Still pending: broader workflow tutorial (`search` → `inspect` → `download` → `cache` lifecycle); first batch of `docs/case-studies/` capturing real-world investigations (e.g., candle [#3448](https://github.com/huggingface/candle/issues/3448) Gemma 4 multimodal naming + per-layer shape variation, [#2875](https://github.com/huggingface/candle/issues/2875) Flux F8_E4M3 dtype audit). Case studies are written *after* the issue comments have had time to gather feedback, so the narrative is informed by real reception. Establishes the habit of shipping docs alongside features.** |

These are more complex than prior releases: verify needs network + checksum comparison, gc needs the last-accessed heuristic and interactive prompt safety, `du --tree` is a display feature that benefits from the `du --age` timestamps added in v0.9.4. The docs effort scopes to the new v0.10.0 cache features plus the first case studies (not a comprehensive rewrite of all existing docs). This is the project's coming-of-age release: where `hf-fetch-model` graduates from "useful tool with `--help`" to "documented, mature tool with narrative onboarding".

The cache fast-path correctness and pipe-FAQ ✓ items above are early increments toward v0.10.0 — UX bugs surfaced during v0.9.8 dogfooding, fixed on `main` immediately rather than waiting for a v0.9.9 patch release. They establish a small precedent: post-release dogfooding gaps land on the next-minor's branch, not the current-patch's. With `cache verify`, `cache gc`, and `du --tree` all marked ✓, the v0.10.0 feature work is complete; only the **First docs effort** remains before the version bump.

### v0.10.1 — `inspect --check-gpu` (hypomnesis adoption) ✓

| Feature | Scope |
|---------|-------|
| `inspect <repo> [FILE] --check-gpu [N]` ✓ | Adds a GPU-fit verdict to `inspect`: model weight size compared against free VRAM on device `N` (default 0). Works on both per-file (`inspect <repo> file.safetensors --check-gpu`) and whole-repo (`inspect <repo> --check-gpu`, forces shard aggregation so the verdict reflects the precise sum of tensor byte-lens across every shard, not the looser file-size proxy). Uses the **unfiltered** model totals (so `--filter` / `--limit` only affect the printed tensor table). On systems with no NVIDIA GPU detected, an out-of-range device index, or where neither NVML nor DXGI is usable, the verdict line reports the failure verbatim and the command still exits 0 — `--check-gpu` is informational, never a gate. |
| `--json` composition ✓ | `gpu_check` JSON object added to the existing per-file schema, the `--tree --json` schema, and the `--dtypes --json` schema (each gated by `#[serde(skip_serializing_if = "Option::is_none")]` so non-`--check-gpu` JSON output stays byte-identical to v0.10.0). The repo-level plain `--json` schema switches from a `Vec<(filename, header)>` array to `{"files": [...], "gpu_check": {...}}` when `--check-gpu` is passed (the array schema is preserved when `--check-gpu` is absent). |
| `hypomnesis = "0.2"` dependency ✓ | Bumped from the originally-planned `"0.1"` to the current crates.io tip; 0.2.0 was released after the v0.10.1 roadmap line was written and is API-additive (`device_info` / `device_count` unchanged). hf-fm v0.10.1 is hypomnesis's **first external consumer** — concrete dogfooding observations captured in [`docs/dogfooding-feedbacks/hypomnesis-adoption.md`](../dogfooding-feedbacks/hypomnesis-adoption.md). |
| `format_size` extracted to `src/format.rs` ✓ | The byte-formatter that has lived in `src/bin/main.rs` since v0.9.4 moves to a binary-internal module (via `#[path = "../format.rs"] mod format;`) so the new `src/gpu_check.rs` verdict renderer can reuse it. Five new unit tests cover the four bucket transitions (B / KiB / MiB / GiB / TiB). No public library API change. |
| Tutorial update ✓ | Tutorial §6 of [Inspect before you download](../tutorials/inspect-before-downloading.md) gains a real `--check-gpu` walkthrough against zeta-2 on an RTX 5060 Ti (verdict: `✗ short by 1.77 GiB` — exact numbers vary run-to-run with other GPU processes, the ✗ is stable). §7's "manual math" paragraph is compressed to a one-liner that points at the verdict already shown in §6. The tutorial's STYLE CONVENTIONS §4 prescribed this change in advance; v0.10.1 closes that forward reference. |

Sample output:

```
$ hf-fm inspect google/gemma-4-E2B-it model.safetensors --check-gpu

  Model weights:  9.54 GiB  (BF16, 5.12B params)
  GPU 0:          NVIDIA GeForce RTX 5060 Ti — 16.0 GiB VRAM
                  free: 14.2 GiB, used: 1.8 GiB
  Fit:            ✓ 4.66 GiB headroom for weights + KV cache + runtime

  Note: reports weights only. Large-context inference typically needs ~1.3–1.5×
  weight size for KV cache and activations.
```

Validates the `hypomnesis` API surface against a real consumer before `candle-mi` migrates in `hypomnesis` Phase 3 (candle-mi v0.2). Implementation-light on the hf-fm side: read the device info, format the verdict alongside the existing inspect output. The hard work — Windows DXGI per-process VRAM, NVML `u64::MAX` sentinel handling, `nvidia-smi` fallback — lives in `hypomnesis` and is already battle-tested code ported from candle-mi.

**Status:** shipped. `hypomnesis 0.2.0` (the current tip; the originally-planned 0.1.0 was bumped because 0.2 is API-additive over 0.1 and the upstream brief moved on). hf-fm v0.10.1 is the proof-of-concept consumer named in the [hypomnesis brief](https://github.com/PCfVW/hypomnesis/blob/main/docs/hypomnesis-brief.md) under *First consumer*. Concrete adoption feedback for the upstream maintainer (us) is captured in [`docs/dogfooding-feedbacks/hypomnesis-adoption.md`](../dogfooding-feedbacks/hypomnesis-adoption.md).

### v0.10.2 — GGUF inspect (cached) + anamnesis dep + hypomnesis 0.2.1 adoption

| Feature | Scope |
|---------|-------|
| `inspect <repo> file.gguf --cached` | Parse GGUF metadata block from locally-cached files. Implementation delegates to `anamnesis::parse_gguf(path).inspect()` — anamnesis is the format-knowledge crate; hf-fm is the HTTP/cache substrate. |
| `--tree` for GGUF | Same hierarchical view, applied to GGUF tensor names. |
| `--dtypes` for GGUF | Same per-dtype summary, applied to GGUF tensors (which use a different dtype encoding than safetensors — needs a small mapping layer). |
| `anamnesis = "0.4.3"` dependency | First adoption of [anamnesis](https://crates.io/crates/anamnesis) — a framework-agnostic Rust crate for parsing tensor formats and dequantising quantised weights. Pulled in here because anamnesis already has a battle-tested GGUF parser; v0.10.3 then extends the dep across the other formats with no further library work. |
| `hypomnesis 0.2 → 0.2.1` + `name_or_unknown()` adoption + `test-helpers` dev-feature | Cashes in the [hypomnesis-adoption.md](../dogfooding-feedbacks/hypomnesis-adoption.md) dogfooding report. Bumps the runtime dep to 0.2.1, replaces the inline `dev.name.as_deref().unwrap_or("unknown GPU")` at [src/gpu_check.rs:200](../../src/gpu_check.rs#L200) with the upstream convenience, and adds `hypomnesis = { version = "0.2.1", features = ["test-helpers"] }` under `[dev-dependencies]` so the `GpuDeviceInfo::builder()` becomes available to tests without leaking into the runtime binary. Writes the two fit-path / miss-path JSON unit tests that the comment block at [src/gpu_check.rs:475-482](../../src/gpu_check.rs#L475-L482) currently defers to manual smoke testing. **Pulled forward from v0.10.4** so the v0.10.1 hypomnesis adoption loop closes in the same patch that opens the anamnesis adoption loop. |
| `diff --dtypes` flag | Side-by-side per-dtype histograms for two repos plus a Δ row showing per-dtype byte/tensor deltas. Refactors `inspect --dtypes`'s aggregator into a shared helper, then renders two columns + a delta column. Composes with `--filter` (histogram aggregates over filtered tensors) and `--summary` (`--dtypes` replaces the per-tensor body; `--dtypes --summary` prints histogram + one-line totals). `--json` gains a `dtype_histograms: { a, b }` field alongside the existing per-tensor diff. **Direct enabler for the candle #3530 p2 reply** — if/when @sempervictus shares the crashing 80B repo and the working 35B sibling, this flag lands the side-by-side dtype evidence that distinguishes "scaled sibling, allocator-side bug" from "architectural variant" in one screenshot. Context: [docs/issues/candle-3530-p1.md](../issues/candle-3530-p1.md). |
| `diff --json` byte-count enrichment + jq recipe | Adds `byte_count: u64` to `DiffTensorSide` ([src/bin/main.rs:3507](../../src/bin/main.rs#L3507)) so a downstream analyzer can sum bytes per pattern bucket from the existing `diff --json` output. Documents a `jq` recipe in [docs/FAQ.md](../FAQ.md) collapsing `model.layers.{N}.*` into one bucket per pattern with a count and a summed byte total. **JSON-first by design**: the recipe is the structured-analysis hook for layer-index collapse; baking a `--collapse` flag into the tool waits on real-world cases informing the right heuristic (numeric-segment abstraction? template grouping? expert-routing-aware?). One-line struct change + a doc paragraph; the bake-in is deferred to v0.11 once two or three real cases land. |

Closes the format-coverage gap with `safetensors_explorer` for cached files. Lower lift than full remote support — local file reading only, leverages existing tree/dtypes infrastructure from v0.9.6. The dominant audience here is `llama.cpp` users who already have GGUF files in their HF cache.

**The dep enters here, not in v0.10.3.** v0.10.2 *needs* a GGUF parser to ship the cached-inspect feature; anamnesis already has one. Bringing it in at v0.10.2 gates v0.10.3's three-format extension on a single, focused dependency commit.

**Ecosystem-adoption release.** Hf-fm's two adjacent co-developed libraries — hypomnesis (GPU substrate, dep since v0.10.1) and anamnesis (format substrate, *new* dep here) — both land properly in one stroke. The release also ships the candle #3530 reply enabler (`diff --dtypes`) and opens the JSON-first analysis hook for future case studies.

**Commit order on the v0.10.2 work branch, smallest to largest, smallest-risk to highest-risk:**

1. **hypomnesis 0.2.1 bump** — `Cargo.toml` runtime + dev-deps lines, [src/gpu_check.rs:200](../../src/gpu_check.rs#L200) call-site swap, two unblocked fit/miss JSON unit tests replacing the deferred-tests comment block. ~15 LOC + tests. Zero new dispatch paths.
2. **`diff` enhancements** — refactor `inspect --dtypes` aggregator into a shared helper; add `--dtypes` flag to `diff` with side-by-side rendering and the `dtype_histograms` JSON field; add `byte_count` to `DiffTensorSide`. ~80 LOC + tests. Additive flag, no behavior change on default `diff`.
3. **jq recipe in [docs/FAQ.md](../FAQ.md)** — a new entry under the Discovery section: "How do I compare two HuggingFace models structurally?" pointing at `diff` / `diff --dtypes` / `diff --json | jq …`. Doc-only.
4. **anamnesis 0.4.3 adoption + GGUF inspect cached + `--tree` / `--dtypes` for GGUF** — new dep, new format, new dispatch path, dtype-mapping layer, adversarial-input caps. The bulk of the release.

The order is insurance — commits 1–3 are small, contained, and ship the candle #3530 enabler. If commit 4 hits an unforeseen GGUF / adversarial-input snag, v0.10.2 can still ship with only commits 1–3 and the GGUF work moves to v0.10.3 without renumbering or backing out.

### v0.10.3 — Cached-file format coverage via anamnesis

| Feature | Scope |
|---------|-------|
| `inspect <repo> file.npz --cached` | Parse `.npz` archive metadata via `anamnesis::inspect_npz(path)`. Tensor list, dtypes, shapes — uniform with the other three formats. |
| `inspect <repo> file.pth --cached` | Parse `.pth` (PyTorch state_dict) metadata via `anamnesis::parse_pth(path).inspect()`. Tensor list, dtypes, shapes. |
| Safetensors parser dedup | Replace hf-fm's in-tree JSON header parser with `anamnesis::parse_safetensors_header(&bytes)` on the cache-hit path. hf-fm continues to fetch the bytes; anamnesis owns the format knowledge. |
| Format-aware error message | Lift the `FetchError::UnsupportedInspectFormat` rejection introduced in v0.9.7: `inspect` now dispatches by extension across all four formats (`.safetensors` / `.npz` / `.pth` / `.gguf`) for cached files. |
| Quant-scheme display | `inspect` output gains a `Format: <QuantScheme>` line (FP8 / GPTQ INT4 g=128 / AWQ / BnB-NF4 / etc.) and a "Dequantised: X GB" line, both pulled from `anamnesis::InspectInfo`. |

**Theme: cashing in the anamnesis dep.** v0.10.2 introduced `anamnesis 0.4.3` as a dependency (it's the GGUF parser); v0.10.3 extends that adoption to the other three formats and retires hf-fm's duplicated safetensors header parser. **No new HTTP work, no new library work in anamnesis** — pure dispatch wiring on a dep that's already there. ~115 LOC + tests, low-risk patch release.

After v0.10.3, `hf-fm inspect <repo> <any-tensor-file> --cached` works uniformly across all four tensor formats anamnesis supports. Sets the stage for v0.11, where the same four formats become inspectable *remotely* via HTTP Range.

### v0.10.4 — `--check-gpu` follow-ups (multi-GPU + KV-cache budgeting)

| Feature | Scope |
|---------|-------|
| `inspect --check-gpu all` | Multi-GPU verdict. Iterates over `hypomnesis::device_count()`, prints one `GPU N:` line per visible device with its own free/used numbers, and picks the device with the most free VRAM for the `Fit:` summary line. The `all` token is parsed via clap's `value_parser`: a bare `--check-gpu` keeps the v0.10.1 single-device default, `--check-gpu N` keeps single-device targeting, `--check-gpu all` enables the iteration. JSON path gains a `devices: [...]` array under `gpu_check` and the existing `device` key becomes an alias for the auto-picked entry. Closes the "I have a 5060 Ti and a 5090 in the same box" case without breaking the v0.10.1 single-device default. |
| `inspect --check-gpu --context N` | KV-cache budgeting. Computes KV bytes from the model's `num_key_value_heads × head_dim × dtype_bytes × 2 (K+V) × num_hidden_layers × context` against the user-supplied context length, adds it to the weight bytes, and reports a *real* fit verdict instead of the v0.10.1 "weights only" disclaimer. Reads `config.json` from cache / API for the architectural parameters (already on the cache-first path used by `inspect`); falls back to a clear error when the config is missing or doesn't expose the GQA-related keys. Replaces the v0.10.1 closing note (`Note: reports weights only…`) with a `KV cache @ ctx=N: X.YZ GiB` line and a precise `Total: W + KV = X.YZ GiB` rollup. |

*(The `GpuDeviceInfo::name_or_unknown()` adoption originally scheduled here was pulled forward to v0.10.2 — see that section.)*

**Theme: extending `--check-gpu` with multi-GPU and KV-cache awareness.** v0.10.1 shipped single-device, weights-only `--check-gpu`; v0.10.4 spends the hypomnesis dependency on the two features users will ask for once they have used `--check-gpu` on real boxes. Lands after the anamnesis-driven v0.10.2 / v0.10.3 patches because KV-cache budgeting wants the same config-aware infrastructure those patches are building (the cache-first `config.json` read in particular).

**Out of scope for v0.10.4 (still deferred):** AMD ROCm support (waits on hypomnesis's `rocm-smi` backend, planned in hypomnesis 0.3); Apple Metal (likewise, future hypomnesis backend); multi-context "fit profile" sweep (e.g., printing fit at 4 K / 16 K / 64 K context in a table). All three are reasonable v0.11+ extensions.

### v0.10.5 — `inspect --pick` (interactive file selection)

| Feature | Scope |
|---------|-------|
| `inspect <repo> [FILE] --pick` | Interactive numbered picker for `inspect`'s file argument. With no positional, lists every supported tensor file in the repo (`.safetensors` / `.gguf` / `.npz` / `.pth` — multi-format from day one, inheriting v0.10.3's dispatcher). With a positional substring, narrows the listing first: if exactly one file matches, auto-runs against that file (printing `Resolving to: <name>` on stderr for transparency); if several match, prints a numbered table and prompts for selection. Composes with every existing rendering flag (`--dtypes`, `--tree`, `--filter`, `--limit`, `--check-gpu`, `--json`). Conflicts with `--list` at clap parse time — `--list` is the non-interactive form of the same listing, combining them is incoherent. |
| TTY safety + cancellation | Pre-flight check on `stdin.is_terminal() && stderr.is_terminal()` (stdout intentionally not checked — the prompt goes to stderr, so `hf-fm inspect ... --pick > out.json` works correctly with the picker on the terminal and JSON to the file). Non-interactive contexts (CI logs, piped stdin, captured stderr) get a clear error pointing at the existing `--list` + numeric-index workflow. Ctrl-C exits with the standard interrupt code; Ctrl-D / empty input bails with `cancelled — no file picked` and a non-zero exit code. Invalid input (`"foo"`, out-of-range integer, zero) loops the prompt without crashing. |
| FAQ entry | The *Discovery — finding what to inspect or download* section in [`docs/FAQ.md`](../FAQ.md) gains a third sub-question on file selection: *"A repo has many `.safetensors` files — can I pick one interactively?"* — pointing at `--pick` alongside the v0.9.7 `--list` + numeric-index path. The two coexist: power users continue running `inspect <repo> --list` then `inspect <repo> 3`; impatient users (and the maintainer) reach for `--pick`. |

Sample interactive session:

```
$ hf-fm inspect little-lake-studios/demoncore-flux demonCORE --pick --dtypes
Multiple .safetensors files match "demonCORE" in little-lake-studios/demoncore-flux:
  1  transformer/demonCORESFWNSFW_fluxV12.safetensors  15.40 GiB
  2  transformer/demonCORESFWNSFW_fluxV13.safetensors  15.40 GiB
  3  transformer/demonCORENSFW_fluxV11.safetensors     15.40 GiB
Pick [1..3]: 2
Resolving to transformer/demonCORESFWNSFW_fluxV13.safetensors

  Dtype    Tensors       Params       Size
  F8_E4M3      948       16.53B  15.40 GiB
  ...
```

**Theme: discovery UX for the long-filename case.** v0.9.7 added `--list` + numeric-index resolution for the same problem, but at the cost of a two-step workflow. `--pick` collapses that to one keystroke for unique matches and one numeric choice for ambiguous ones — and inherits format coverage from v0.10.3's multi-format dispatcher, so the picker works uniformly across `.safetensors`, `.gguf`, `.npz`, and `.pth` from its first release. No new library API, no new dependencies — `std::io::IsTerminal` is already in scope from other CLI paths, and the candidate-narrowing logic reuses the listing helper introduced in v0.9.7. ~80 LOC in `src/bin/main.rs`; candidate-narrowing is pure and unit-testable; the prompt loop itself gets a manual smoke test per shell.

**Concrete dogfooding origin.** Surfaced from the typed-the-full-filename pain documented in candle [#3448](https://github.com/huggingface/candle/issues/3448) (gemma-4-E2B-it's `model.safetensors` typed in full to inspect `--filter embed`) and candle [#2875](https://github.com/huggingface/candle/issues/2875) (demoncore-flux's `transformer/demonCORESFWNSFW_fluxV13.safetensors` — 50 characters of exact filename to get the dtype histogram). The fix is mechanical once you've felt the pain twice.

**Out of scope for v0.10.5 (deferred):**

- **Shell tab completion** (clap_complete-generated bash / zsh / fish / PowerShell scripts with a dynamic callback into `hf-fm inspect <repo> --list`). Native TAB behavior would be nicer than typing a substring, but each shell needs its own integration, dynamic completions on TAB feel slow when they trigger an HTTP listing, and the cross-shell maintenance burden is minor-release scope, not patch. Revisit for v0.11 if user demand surfaces.
- **Multi-selection.** One file per `inspect` invocation, same as today. Users who want all files run `inspect <repo>` (aggregated); users who want a subset can run the command twice. Multi-selection would force the rendering pipeline to handle a `Vec<picked>` instead of a single filename — a deeper refactor that exceeds the patch's UX-only ambition.
- **Fuzzy matching beyond substring** (Levenshtein, prefix-completion via lookahead, etc.). Substring is the cheapest contract that says what the user means: *"the filename contains these characters"*. Stronger matchers expand the surface for "did you mean" confusion that the explicit picker already solves.

---

The v0.11 minor is dedicated to **remote inspection**. v0.11.0 builds an `HttpRangeReader: Read + Seek` adapter once over `reqwest` Range requests; each subsequent patch wires one more tensor format through the same adapter. anamnesis owns format knowledge; hf-fm owns HTTP plumbing. Format order is risk-ascending: NPZ (anamnesis primitive already shipped in v0.4.3) → safetensors (small library work, retires the bespoke parser) → GGUF (medium library work, the originally-promised v0.11.0 feature) → PTH (largest library work, lowest demand).

### v0.11.0 — Remote inspect framework + NPZ remote

| Feature | Scope |
|---------|-------|
| `HttpRangeReader: Read + Seek` adapter | New module on top of `reqwest` Range requests. ~150 LOC including: prefetch + cache the EOCD region (last 64 KiB) on first seek-to-end; cache the central directory on first probe; small read-ahead buffer (4 KiB) for sequential-read patterns; translate `reqwest` errors into `std::io::Error`. The reusable substrate for every subsequent remote-inspect format. |
| `inspect <repo> file.npz` (no `--cached`) | Wire the NPZ dispatch path to `anamnesis::inspect_npz_from_reader(HttpRangeReader::new(url))`. ~7 small range requests fetch the ZIP central directory + per-entry NPY headers; total transfer well under 100 KiB on a typical Gemma Scope `params.npz` (vs the 288 MiB full download). **No new library work in anamnesis** — uses the v0.4.3 primitive directly. |
| Bespoke `fetch_header_bytes` retained | hf-fm's existing safetensors remote-inspect path stays on its bespoke two-Range-request implementation. v0.11.1 retires it. |

**Why NPZ first.** The anamnesis primitive is the only one already shipped — v0.11.0's risk concentrates on the new adapter (the genuinely new piece). NPZ also closes the original candle-mi GemmaScope dogfooding loop that motivated anamnesis's v0.4.3 work: the v0.11.0 ship makes that real end-to-end.

### v0.11.1 — Remote safetensors via anamnesis

| Feature | Scope |
|---------|-------|
| `parse_safetensors_header_from_reader<R: Read>` (anamnesis library work) | New ~30 LOC primitive in anamnesis. **No `Seek` needed** — the safetensors header is sequential at the start of the file. Reads the 8-byte length prefix, then the JSON header, then parses it. |
| Retire hf-fm's bespoke `fetch_header_bytes` | Replace with a small wrapper that issues two HTTP Range requests (length prefix + header bytes) and feeds them as a `Read` to the new anamnesis primitive. |

**User-facing:** no behavioural change — `hf-fm inspect <repo> file.safetensors` works identically. **Architecturally:** single source of truth for safetensors layout. The duplicated parser is gone.

### v0.11.2 — Remote GGUF inspect

| Feature | Scope |
|---------|-------|
| `inspect_gguf_from_reader<R: Read + Seek>` (anamnesis library work) | Refactor the existing `parse_gguf` cursor pattern off `memmap2::Mmap` and onto a `Read + Seek` cursor. Adversarial-input guards from the existing parser (caps on tensor count, KV count, string length, array length, nesting depth, dimension count, element product) carry over unchanged. |
| `inspect <repo> file.gguf` (no `--cached`) | Wire the GGUF dispatch path through `HttpRangeReader`. |

**The originally-promised v0.11.0 feature, now landing on a battle-tested adapter.** A 2 GiB quantised GGUF inspectable in a few range requests fetching the front-loaded metadata block + tensor info table — no weight data downloaded.

### v0.11.3 — Remote PTH inspect

| Feature | Scope |
|---------|-------|
| `inspect_pth_from_reader<R: Read + Seek>` (anamnesis library work) | Largest of the four library lifts. PTH's metadata lives in `data.pkl` mid-archive; the existing pickle VM zero-copies from `memmap2::Mmap`. Reader-based path materialises `data.pkl` (typically <100 KiB) via the ZIP central directory, then runs the existing pickle interpreter on that buffer — no zero-copy contract change for the local-file path. |
| `inspect <repo> file.pth` (no `--cached`) | Wire the PTH dispatch path through `HttpRangeReader`. |

**Closes the matrix.** Every tensor format anamnesis supports is now remotely inspectable via the same `HttpRangeReader` substrate. After v0.11.3, `hf-fm inspect <repo> <any-tensor-file>` works uniformly — cached or remote, regardless of format.
