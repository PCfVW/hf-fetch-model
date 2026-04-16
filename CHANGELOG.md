# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`inspect --dtypes` flag** — shows a per-dtype summary (tensor count, parameter count, byte size) instead of listing individual tensors. Composes with `--filter` to show dtype breakdown for a subset of tensors.
- **`inspect --limit N` flag** — caps the tensor list to the first N entries (applied after `--filter`). Solves the "wall of JSON" problem when peeking at a large model's schema. The human-readable footer shows `shown/total` or `shown/matched/total` when truncation occurs; the `--json` output gains a top-level `truncated: { shown, total }` field so downstream consumers can detect incomplete output. Non-truncated JSON output is schema-identical to v0.9.5.

### Fixed

- **Dynamic table column widths** — all CLI table outputs (`inspect`, `diff`, `list-files`, `status`, `du`, `search`, `list-families`, `discover`, `--dry-run`) now compute column widths from the actual data instead of using hardcoded values. Fixes misaligned columns when tensor names, filenames, or repo IDs exceed the previous fixed width (e.g., multimodal models with long tensor prefixes like `model.vision_tower.encoder.layers.*.mlp.*`).

## [0.9.5] — Library hardening

### Added

- **Watch-based progress channel** — `FetchConfigBuilder::progress_channel()` returns a `tokio::sync::watch::Receiver<ProgressEvent>` for async consumers. Call `.changed().await` to receive the latest progress update. Composes with the existing `on_progress()` callback — both can be active simultaneously.

### Fixed

- **Chunked download timeout** — chunked (multi-connection) downloads now respect `timeout_per_file` (default 300 s), matching the single-file download path. Previously, a silent network partition during a chunked download could stall indefinitely, holding the concurrency semaphore and blocking the entire batch.
- **TCP connect timeout** — both HTTP clients (`build_client`, `build_no_redirect_client`) now set a 30-second TCP connect timeout, bounding the connection handshake phase for all download, probe, inspect, and info operations that use these clients.
- **Windows blob corruption** — on Windows without symlink privileges, the pointer finalization step now copies the blob instead of renaming it. `rename` destroyed the `blobs/<etag>` entry, breaking cross-revision deduplication and causing full re-downloads when the same model was accessed at different revisions. Diverges from `hf-hub`'s `symlink_or_rename()` which has the same defect upstream.
- **POSIX symlink TOCTOU race** — `symlink_or_copy` now tolerates `EEXIST` on the POSIX `symlink()` call. Two concurrent downloaders racing to create the same pointer symlink no longer produce a spurious `FetchError::Io`.

### Changed

- **`parse_header_json` zero-clone iteration** — the safetensors header parser now consumes the intermediate `HashMap` via `into_iter()` instead of borrowing it, eliminating per-tensor `value.clone()` and `key.clone()` allocations.
- **`inspect_repo_safetensors` cancellation on failure** — replaced `Vec<JoinHandle>` with `JoinSet` and `abort_all()` on first error, preventing detached tasks from continuing HTTP requests after the function has already returned an error.
- **`check_disk_space` no longer walks the entire cache** — removed the `cache_summary()` call that scanned every cached model directory on the Tokio thread before every download. The disk space display now shows download size, available space, and projected remaining space — matching the Python `huggingface_hub` approach. Eliminates a multi-second blocking stall on large caches.
- **`try_resolve_repo_from_cache` moved to `spawn_blocking`** — the per-repo cache file scan (`collect_cached_files_recursive`) now runs on a blocking thread pool instead of the Tokio worker thread, preventing stalls on repositories with many files.
- **CDN URL expiry detection** — chunked downloads now parse `X-Amz-Expires` from the CDN signed URL to estimate when it will expire. If the estimated download time exceeds the remaining URL validity, a warning is logged and a fresh URL is probed before starting the download. Prevents silent failures on very large files over slow connections.
- **Temp file cleanup on abort** — chunked downloads now use an RAII `TempFileGuard` that removes the pre-allocated `.chunked.part` file on drop, including when tasks are aborted via `JoinSet::abort_all()`. Previously, aborted tasks left orphaned temp files consuming up to tens of GiB of disk space.
- **Shared HTTP client for `list_repo_files_with_metadata`** — the function now accepts a `&reqwest::Client` parameter instead of creating a disposable client per call. This reuses TCP connections and TLS sessions, eliminating redundant handshakes. `build_client` is now re-exported as a public API for library consumers. **Breaking:** callers must pass a `&reqwest::Client` (use `build_client(token)` to create one).
- **Cache layout centralization** — all hf-hub cache path construction (`models--org--name`, `snapshots/`, `blobs/`, `refs/`) is now centralized in a new `cache_layout` module. `repo_folder_name()` delegates to `hf_hub::Repo::folder_name()`. Replaces ~15 scattered `format!("models--{}", ...)` call sites and ~25 inline `.join("snapshots")` chains across 7 files. One module to audit when hf-hub bumps.

## [0.9.4] — Search tags, cache path & du age

### Added

- **`--tag` search flag** — `hf-fm search llama --tag gguf` filters models by tag (maps to the HF API `filter` parameter and applies client-side validation). Useful for GGUF models which typically lack a `library_name` but carry the `gguf` tag. `SearchResult` now includes a `tags` field.
- **`cache path <REPO_ID|N>`** — prints the snapshot directory path for a cached model. Output is a bare path for shell substitution: `cd $(hf-fm cache path google/gemma-2-2b-it)`. Accepts numeric index from `du` output. Currently resolves the `main` ref only.
- **`du --age`** — adds a last-modified age column (e.g., `"2 days ago"`, `"3 months ago"`) to the `du` summary. Uses the most recent file modification time in the snapshot directory. Sort order remains by size.
- **Em-dash legend in `list-files`** — when the SHA256 column shows `—` for non-LFS files, a footnote now explains: `— = not an LFS file (no SHA256 tracked by the Hub)`.
- **`search --help` cross-references** — search help text now mentions `list-families` and `discover` as related commands via a "See also" line.

### Changed

- **`--exact` help text** — reworded from "Return only the exact model ID match" to "Match a full repository ID exactly and show its metadata card" for clarity.
- **Binary name in usage line** — `--help` now shows `Usage: hf-fm [OPTIONS]` on all platforms (previously showed `hf-fm.exe` on Windows).
- **`format_size` TiB tier** — values >= 1000 GiB now display as TiB (e.g., `"2.00 TiB"`) instead of a four-digit GiB value.
- **`cache delete` targeted scan** — the deletion preview now scans only the target repo's snapshot directory instead of the entire cache, eliminating a full O(R) filesystem walk per delete.
- **`du <REPO>` targeted partial check** — the partial-download hint now checks only the target repo's blobs directory instead of rescanning the entire cache.
- **`repo_status` hoisted partial check** — partial-blob detection is now performed once before the per-file loop instead of re-scanning the blobs directory for every missing file.
- **`search` pre-normalized model IDs** — model IDs are now normalized once before client-side filtering instead of re-allocating per result per filter term.
- **`inspect --json --filter` filter-before-clone** — tensor filtering is now applied before cloning header metadata, avoiding O(T) clone-then-discard on large models.
- **`diff` uses `BTreeSet`** — tensor name deduplication now uses `BTreeSet` (sorted on insert) instead of `HashSet` → `Vec` → `sort`, eliminating the intermediate allocation and hashing overhead.

### Fixed

- **`format_age` future timestamps** — clock skew or future file timestamps now display `—` instead of the misleading `"< 1 hour"`.
- **`publish.yml` missing `--all-features` tests** — the publish workflow now runs `cargo test --all-features` (matching `ci.yml`), ensuring CLI-gated code is tested before crates.io release.
- **v0.8.2 CHANGELOG inaccuracy** — corrected the candle-mi auto-update entry which falsely claimed `publish.yml` automated the version bump (the step was implemented then removed; the process is manual per `CLAUDE.md`).

## [0.9.3] — Cache management, gated model detection & du numbered indexing

### Added

- **Gated model pre-flight check** — downloads now fail fast with a clear message when a repository is gated and either no token is configured or the token is rejected (invalid token or license not accepted), instead of producing per-file 401 errors.
- **`du` prints cache path** — `hf-fm du` and `hf-fm du <REPO_ID>` now display the absolute cache directory path as a header line.
- **`du` numbered indexing** — `hf-fm du` now shows a `#` column with 1-based numbering. `hf-fm du 2` drills into the 2nd largest cached repo (same as `hf-fm du org/model`). Partial downloads are marked with `●`, and the drill-down view hints to run `hf-fm status` for details.
- **`cache` subcommand group** — new `hf-fm cache` parent command for destructive cache operations. Future commands (`path`, `verify`, `gc`) will be added here.
- **`cache clean-partial`** — removes `.chunked.part` temp files from interrupted downloads. Supports whole-cache or single-repo scope (by repo ID or `#` index), `--yes` to skip confirmation, and `--dry-run` to preview.
- **`cache delete`** — deletes a cached model by repo ID or `#` index. Shows a size preview and prompts for confirmation (`--yes` to skip).
- **`candle_inspect` example** — runnable example showing how to inspect a model's tensor layout (names, shapes, dtypes) via HTTP Range requests before downloading weights. Run: `cargo run --example candle_inspect`.

### Fixed

- **`du` column alignment** — size values near 1 GiB (e.g., `1023.20 MiB`) no longer overflow the SIZE column; they are now displayed as GiB. The REPO column width adapts to the longest repo name.

## [0.9.2] — CLI ergonomics (dogfooding)

### Added

- **Version in `--help` output** — `hf-fm --help` now displays the version number in the header line (previously only available via `-V`/`--version`).
- **`--preset pth`** — filter preset for PyTorch `.bin` weight files (`pytorch_model*.bin` plus `*.json` and `*.txt`). Available on the download command and `list-files`.
- **Glob patterns in `download-file`** — `hf-fm download-file org/model "pytorch_model-*.bin"` now expands glob patterns against the remote file list and downloads all matches. Exact filenames are still supported (backward compatible).
- **`--flat` download flag** — copies downloaded files to a flat directory layout (`{output-dir}/{filename}`) after download. Defaults to the current directory when `--output-dir` is not set. Available on both the default download command and `download-file`.
- **`has_glob_chars()` public function** — detects glob metacharacters in a string, re-exported at the crate root.

## [0.9.1] — Search filtering, model card display & adapter config detection

### Added

- **`search --library` / `--pipeline` flags** — filter by library framework (e.g., `peft`, `transformers`) and pipeline task (e.g., `text-generation`). Filters are applied client-side for reliability (the HF search API does not honor them when combined with a search query). Search results now display library and pipeline metadata in brackets when available.
- **`SearchResult` now includes `library_name` and `pipeline_tag`** — populated from the HF API search response, enabling programmatic filtering.
- **`info` subcommand** — `hf-fm info <REPO_ID>` displays model card metadata (license, pipeline, library, tags, languages, gating status) and README text with YAML front matter stripped. Supports `--json`, `--lines` (default 40, 0 = all), `--revision`, and `--token` flags.
- **`fetch_readme()` API** — fetches raw README text from a `HuggingFace` repository. Returns `None` on 404.
- **Adapter config detection in `inspect`** — when inspecting a repository, `adapter_config.json` is automatically detected and its PEFT configuration (type, base model, rank, alpha, target modules, task type) is displayed alongside tensor metadata.
- **`AdapterConfig` type** — lightweight struct for parsed PEFT adapter configuration, re-exported at the crate root.
- **`fetch_adapter_config()` / `fetch_adapter_config_cached()` API** — fetches and parses `adapter_config.json` from a repository (cache-first or cache-only).

## [0.9.0] — Safetensors inspection, tensor diff & cache disk usage

### Added

- **`diff` subcommand** — compare tensor layouts between two models. Shows tensors only-in-A, only-in-B, dtype/shape differences, and matching count. Supports `--cached`, `--filter`, `--summary`, `--json`, and per-repo `--revision-a`/`--revision-b` flags.
- **`inspect` subcommand** — read safetensors tensor metadata (names, shapes, dtypes, offsets) from local cache or remote repos via HTTP Range requests, without downloading full files. Supports `--json`, `--no-metadata`, `--cached` flags. For sharded models, uses the shard index as a fast path (1 request instead of 2×N). Cross-validated against Python on 199 cached files (16,501 tensors, 0 discrepancies).
- **`inspect_safetensors()` / `inspect_safetensors_local()` / `inspect_safetensors_cached()` API** — single-file header inspection (cache-first, local-only, or cache-only).
- **`inspect_repo_safetensors()` / `inspect_repo_safetensors_cached()` API** — multi-file inspection with concurrent fetching.
- **`fetch_shard_index()` / `fetch_shard_index_cached()` API** — shard index parsing for sharded safetensors models.
- **`TensorInfo` / `SafetensorsHeaderInfo` / `ShardedIndex` types** — lightweight tensor metadata types with `Serialize` support for JSON output.
- **`FetchError::SafetensorsHeader` variant** — for malformed safetensors headers.
- **Disk space check before download** — shows current cache size, projected size after download, and available disk space. Warns if space is tight or insufficient.
- **`RepoNotFound` search hint** — when a repository is not found, suggests `hf-fm search <model-name>` to help find the correct name.
- **`inspect --filter <PATTERN>`** — show only tensors whose name contains the given substring. Works with all output modes (`--json`, shard index, multi-file summary). Summary line shows filtered/total counts.
- **`inspect` no-safetensors hint** — when a repo has no `.safetensors` files, suggests `hf-fm list-files <repo>` to see available file types.
- **`du` subcommand** — shows disk usage for cached models. Without arguments, lists all cached repos sorted by size. With a repo ID, shows per-file breakdown. Repos with incomplete downloads show a `PARTIAL` marker.
- **`cache_repo_usage()` API** — returns per-file disk usage for a specific cached repository.

## [0.8.2] — Download performance & observability

### Fixed

- **auto_plan never applied** — CLI default values for `--concurrency`, `--chunk-threshold-mib`, and `--connections-per-file` marked all three as explicit, preventing the data-driven download plan from optimizing settings. Now uses `Option` types so the plan optimizer applies automatically when flags are omitted.

### Added

- **Download summary line** — after a successful download, prints total size, elapsed time, and throughput (e.g., `923 MiB in 12.3s (75.0 MiB/s)`).
- **Non-TTY progress** — when stderr is not a terminal (pipes, CI), emits periodic progress lines to stderr every 5 seconds or 10% of total size.
- **Redundant filter warning** — warns when `--filter` globs duplicate patterns already included by `--preset`.
- **Search term normalization** — common quantization synonyms (`8bit`/`8-bit`/`int8`, `4bit`/`4-bit`/`int4`, `fp8`/`float8`) are normalized before searching the HuggingFace Hub API.

### Changed

- **CI: candle-mi post-publish step removed** — the `publish.yml` workflow originally included an automatic candle-mi version bump, but it was removed as the manual process (documented in `CLAUDE.md`) proved more reliable.
- **Download count formatting** — search results now display download counts with thousand separators (e.g., `1,234,567`) instead of abbreviated suffixes (`1.2M`).
- **Docs: auto-tuning** — updated rustdoc, CLI reference, and configuration docs to reflect that `concurrency`, `chunk_threshold`, and `connections_per_file` are now auto-tuned by the download plan optimizer when not explicitly set.

## [0.8.1] — Bug fixes, partial detection & CLI tests

### Added

- **CLI integration tests** — 16 tests exercising the `list-files` subcommand, `--dry-run` flag, help text, error handling, and output formatting. Includes a regression test for the chunk threshold display bug. CI now runs `cargo test --all-features` to include CLI tests.

### Fixed

- **Plan-to-config optimization** — the "many small files" strategy no longer triggers when large files (≥1 GiB) are present. A repo with 2 × 4 GiB safetensors + 8 small config files now correctly uses the mixed strategy (concurrency 4, 8 connections/file, 100 MiB chunk threshold) instead of disabling chunked downloads.
- **Dry-run display** — chunk threshold `u64::MAX` (disabled chunking) now displays as "disabled" instead of an astronomical MiB number.
- **`--show-cached` partial detection** — `list-files --show-cached` now compares local file size against expected size to detect partially downloaded files (shows "partial" instead of ✓). Previously, any existing file showed ✓ even if the download was interrupted.

## [0.8.0] — list-files, dry-run & download plan

### Added

- **`list-files` subcommand** — inspect remote repo contents (filenames, sizes, SHA256) without downloading. Supports `--filter`, `--exclude`, `--preset`, `--no-checksum`, and `--show-cached` flags.
- **`--dry-run` flag** — preview what would be downloaded, compare against local cache, and display recommended download settings. Available on the default download command (`hf-fm <REPO_ID> --dry-run`).
- **`DownloadPlan` type** — new public API (`download_plan()`) for computing a download plan (file list, sizes, cache status) without downloading. Includes `recommended_config()` for plan-based optimization of `FetchConfig`.
- **`FilePlan` type** — per-file entry within a `DownloadPlan`.
- **`download_with_plan()` / `download_with_plan_blocking()`** — execute a download using a precomputed plan and config.
- **`file_matches()` public function** — promoted from `pub(crate)` for use outside the download pipeline.
- **`compile_glob_patterns()` public function** — builds compiled glob filters from pattern strings.
- **`FetchConfig` accessors** — `concurrency()`, `connections_per_file()`, `chunk_threshold()` public const methods.

### Changed

- **Implicit plan optimization** — `download_with_config()` now internally computes a `DownloadPlan` and applies recommended settings for unset config fields. Every download benefits from plan-based tuning automatically.
- **Help text** — main command help now explains: "Downloads all files from a HuggingFace model repository. Use `--preset safetensors` to download only safetensors weights, config, and tokenizer files."
- **MSRV bumped to 1.88** — aligns with the actual dependency floor (`cookie_store`, `time` already require 1.88). Previously advertised 1.75 but compilation required 1.88 regardless.
- **Documentation** — README, CLI reference, configuration guide, and architecture doc updated for `list-files` and `--dry-run`. New download plan (dry-run API) section in configuration guide.

### Fixed

- **CI: upgrade `actions/checkout` from v4 to v5** — v4 runs on Node.js 20, which GitHub is deprecating in June 2026; v5 uses Node.js 24.

## [0.7.3] — Smarter Search & Documentation Overhaul

### Added

- **Search: slash normalization** — `/` in search queries is now replaced with a space before querying the HF API, so `hf-fm search mistralai/3B` works as expected.
- **Search: comma-separated multi-term filtering** — `hf-fm search mistral,3B,12` splits on `,`, sends the first term to the API, then filters results client-side to keep only models whose ID contains all terms.
- **Search: `--exact` flag** — `hf-fm search <model_id> --exact` returns only the exact match. On miss, shows "Did you mean:" suggestions from the fuzzy results.
- **Search: model card metadata** — when `--exact` finds a match, fetches and displays license, gating status, pipeline tag, library, tags, and languages from the HF model card API.
- `ModelCardMetadata` struct and `fetch_model_card()` function in `discover` module.
- `GateStatus` enum (`Open`, `Auto`, `Manual`) with `is_gated()` accessor and `Display` impl, re-exported at crate root.
- Re-exported `SearchResult`, `ModelCardMetadata`, and `GateStatus` at the crate root.

### Fixed

- Backtick hygiene: wrapped all `hf-hub` references in doc comments with backticks across `chunked.rs`, `download.rs`, and `error.rs` (14 occurrences).

### Changed

- Rewrote `README.md` as a short landing page (~70 lines) with install, try-it flow, and library quick start. Moved detailed content to topic-specific docs: `docs/cli-reference.md`, `docs/search.md`, `docs/configuration.md`, `docs/architecture.md`, `docs/diagnostics.md`.
- Added `homepage` and `documentation` fields to `Cargo.toml` for crates.io metadata links.
- Tailored `CONVENTIONS.md` for hf-fetch-model: removed candle-mi-specific sections (PROMOTE, CONTIGUOUS, Shape Documentation, Hook Purity Contract, Memory Doc Section, OOM-safe Decoder Loading Pattern), added Intra-Doc Link Safety rules, adapted all examples and error types to use `FetchError` instead of `MIError`.

## [0.7.2] — Cache Fallback & Download Refactor

### Fixed

- Downloads of gated models (e.g., `meta-llama/Llama-3.2-1B`) failed with "file(s) failed to download" even when the model was already cached. Root cause: hf-hub's `.high()` mode sends `Range: bytes=0-0` probes that fail for gated LFS files, and no cache check existed. Added full offline cache resolution: `download_all_files_map` now scans the local snapshot directory **before any network request** and returns immediately if all files are present. Single-file downloads (`download_file_by_name`) also check the cache first. Zero network calls for cached models.

### Added

- `DownloadOutcome<T>` enum (`Cached(T)` / `Downloaded(T)`) returned by all public download functions, so callers can distinguish cache hits from network downloads. Includes `into_inner()`, `inner()`, and `is_cached()` accessors. Re-exported from `hf_fetch_model::DownloadOutcome`.
- CLI now prints "Cached at:" when the model was resolved from local cache, and "Downloaded to:" when it was freshly downloaded.

### Changed

- Refactored `download_all_files_map` (291 → ~90 lines), `download_file_by_name` (162 → ~55 lines), and `download_chunked` (122 → ~80 lines) by extracting shared helpers:
  - `DownloadPlan` — resolved config parameters, avoiding repetitive option unpacking.
  - `dispatch_download()` — shared core download logic (method selection, 416 fallback, cache fallback, logging) used by both batch and single-file paths.
  - `collect_results()` — drains `JoinSet` with timeout checking and progress reporting.
  - `validate_download_results()` — checks for partial failures or empty file maps.
  - `build_shared_state()` — `Arc`-wrapped HTTP clients and cache paths for concurrent tasks.
  - `fetch_metadata_if_needed()` — conditional metadata fetching with logging.
  - `log_download_result()` — timing and throughput logging.
  - `prepare_temp_file()` — directory creation and temp file pre-allocation for chunked downloads.
  - `finalize_chunked_download()` — rename, symlink, and refs file creation.
- Made `cache::read_ref()` `pub(crate)` so `resolve_cached_file()` can look up commit hashes.
- Applied CONVENTIONS.md: fixed `# Errors` doc format on `download_file_by_name`, corrected CAST annotation in `progress.rs`.
- Fixed `// EXPLICIT:` → `// CAST:` annotations on 6 `as` casts in `main.rs` (`format_size`, `format_downloads`).
- Removed duplicate `cache_dir`/`repo_folder` resolution in `download_file_by_name` (was resolved twice: once for cache check, again for dispatch).
- All functions now pass `clippy::too_many_lines` (≤100 lines) under `clippy::pedantic`.

## [0.7.1] — Metadata & Progress Bar Fixes

### Fixed

- `list_repo_files_with_metadata()` did not pass `?blobs=true` to the HuggingFace API, so the response never included file sizes or LFS metadata. Without sizes, all files fell through to single-connection downloads regardless of the `chunk_threshold` setting. Now appends `?blobs=true` to the API URL, enabling chunked multi-connection downloads for large files as intended.
- `IndicatifProgress` overall file counter showed `9/9` instead of `8/8` for an 8-file repo with one chunked download: the 8-connection chunked download path fired a streaming event with `percent=100.0` when all chunks completed, then the orchestrator fired a second `completed_event` for the same file. Added a `completed_files` `HashSet` to deduplicate completion events.

### Changed

- `IndicatifProgress::handle()` now creates per-file progress bars for in-progress streaming events, showing bytes downloaded, throughput, and ETA. Previously it only tracked completed files via an overall counter, providing no visual feedback during large file downloads.

## [0.7.0] — Phase 7: Default Chunked Downloads & Download Diagnostics

### Changed

- `download()` and `download_files()` now delegate to their `_with_config` counterparts with a default `FetchConfig`, enabling multi-connection chunked downloads (≥100 MiB, 8 connections per file) by default. Previously, these functions bypassed `FetchConfig` entirely and used single-connection downloads via hf-hub's `.get()`, even though the multi-connection infrastructure existed.
- Eliminated duplicated `ApiBuilder` setup code in `download()` and `download_files()` — both are now 2-line delegating functions.

### Added

- `tracing` dependency (0.1) for structured download diagnostics at `debug` level:
  - **Download plan**: total files, concurrency, connections per file, chunk threshold, checksums enabled/disabled.
  - **Metadata fetch**: success with file count and size availability, or warning on failure (explains why chunked downloads may be disabled).
  - **Per-file decision**: whether each file uses chunked (multi-connection) or single-connection download, with file size and reason.
  - **Per-file completion**: elapsed time and throughput in Mbps (when file size is known).
  - **Overall summary**: total files downloaded, failures, and elapsed time.
- `--verbose` / `-v` CLI flag on the default download and `download-file` subcommands: initializes a `tracing-subscriber` at `debug` level for `hf_fetch_model`, printing download diagnostics to stderr. Respects `RUST_LOG` if set.
- Download diagnostics for single-file downloads (`download_file` / `download-file`): per-file chunked/single decision, elapsed time, and throughput.
- `tracing-subscriber` 0.3 dependency (optional, behind `cli` feature) with `env-filter` for `--verbose` support.

### Fixed

- `download()` and `download_files()` silently using single-connection downloads because they passed `config: None` to the internal orchestrator, which set `chunk_threshold = u64::MAX` — effectively disabling the chunked download path that was available since 0.5.0.
- `FetchError::RepoNotFound` was returned when a repository existed but had no files after filtering. Added `FetchError::NoFilesMatched` variant to distinguish "repo not found" from "repo exists but zero files matched".
- Single-file download (`download_file_by_name`) silently swallowed metadata fetch failures via `.unwrap_or_default()`. Now logs a `tracing::warn!` explaining that file size is unknown and chunked download is disabled.
- `search_models()` interpolated the query string directly into the URL without encoding. Now uses reqwest's `.query()` builder for proper URL encoding of special characters.
- `has_partial_blob()` accepted a `_filename` parameter it never used. Removed the unused parameter; added doc comment clarifying the repo-level heuristic.
- Public API docs on `download()`, `download_with_config()`, `download_files()`, and `download_files_with_config()` incorrectly promised `FetchError::Auth` for authentication failures. Auth errors currently surface as `FetchError::Api` via hf-hub; docs updated accordingly. The `Auth` variant is retained (reserved for future use).
- Added field-level documentation to all `pub(crate)` fields in `FetchConfig`.
- `retry_async()` used a `last_error` accumulator with an awkward synthetic fallback. Restructured with match guards so success and final-failure exits are explicit.
- Chunked download error message said "task panicked" for all `JoinError`s, which can also represent cancellation. Changed to "chunk task failed".
- `download_all_files()` used `.parent()` to derive the snapshot directory from a downloaded file path, which returned a wrong path for nested files (e.g., `subdir/file.bin` → `.../snapshots/<sha>/subdir` instead of `.../snapshots/<sha>`). Added `snapshot_root()` helper that strips the filename's path components to recover the true snapshot root. Also fixed `FetchError::PartialDownload.path` which stored a raw file path instead of the snapshot directory.
- `FileStatus::Partial` doc now notes that the `.chunked.part` detection is a repo-level heuristic (may not correspond to a specific file).
- Chunked downloads passed a fixed `total - 1` as `files_remaining` for every file's streaming progress events, regardless of how many files had actually completed. Replaced with a shared `AtomicUsize` counter incremented on each file completion, so in-flight tasks report an accurate remaining count.
- Examples (`basic.rs`, `bench.rs`, `progress.rs`) now return `Result` and use `?` instead of `.expect()`.

## [0.6.0] — Phase 6: Single-File Download API

### Added

- `download_file()` and `download_file_blocking()` public API for downloading a single named file from a HuggingFace repository and returning its cache path
- `download-file` CLI subcommand: `hf-fm download-file <REPO_ID> <FILENAME>` with `--revision`, `--token`, `--output-dir`, `--chunk-threshold-mib`, and `--connections-per-file` flags
- `download::download_file_by_name()` internal orchestration function reusing the existing download pipeline (chunked/standard, retry, checksum, 416 fallback) for a single file
- Single-file download integration tests (`tests/single_file.rs`)

## [0.5.0] — Phase 5: Multi-Connection Downloads, Search & Status

### Added

- Multi-connection HTTP Range-based parallel downloads for large files: files above `chunk_threshold` (default 100 MiB) are split into `connections_per_file` (default 8) concurrent Range requests for maximum throughput
- `--chunk-threshold-mib` and `--connections-per-file` CLI flags
- `FetchConfig::chunk_threshold()` and `FetchConfig::connections_per_file()` builder methods
- `search` subcommand: query the HuggingFace Hub for models matching a string (e.g., `hf-fm search RWKV-7`), sorted by downloads
- `status` subcommand: show per-file download state (complete / partial / missing) for a specific model (e.g., `hf-fm status RWKV/RWKV7-Goose-World3-1.5B-HF`), or scan the entire cache when no repo is given (`hf-fm status`)
- `cache::repo_status()` async API: cross-reference local cache against HF API for per-file status
- `cache::cache_summary()`: local-only scan of entire HF cache with file counts and sizes
- `cache::FileStatus` enum (`Complete`, `Partial`, `Missing`) with `#[non_exhaustive]`
- `cache::RepoStatus` and `cache::CachedModelSummary` structs
- `discover::search_models()` async API and `discover::SearchResult` struct
- `progress::streaming_event()` helper for mid-download progress reporting
- Direct HTTP GET fallback when hf-hub fails with HTTP 416 Range Not Satisfiable (small git-stored files)
- HF API commit hash resolution for fresh downloads (when `refs/main` does not yet exist)
- `futures-util` 0.3 dependency; `stream` feature on `reqwest`; `fs` feature on `tokio`

### Fixed

- Progress bar rendering twice on completion (shared via `Arc`, `AtomicBool` finish-once guard)
- Fresh download failure on first file when `refs/main` is absent (now resolved via HF API fallback)

## [0.4.0] — Phase 4: CLI & Publish

### Added

- CLI binary installed as both `hf-fetch-model` and `hf-fm` (behind `cli` feature)
- `--revision`, `--token`, `--filter`, `--exclude`, `--preset`, `--output-dir`, `--concurrency` CLI flags
- `list-families` subcommand: scan local HF cache, group models by `model_type`
- `discover` subcommand: query HF Hub API, show model families not yet cached locally
- `FetchConfig::output_dir()` builder method for custom download directory
- `cache` module: `hf_cache_dir()`, `list_cached_families()`
- `discover` module: `discover_new_families()`
- `examples/basic.rs`, `examples/progress.rs`, `examples/bench.rs`
- `benches/throughput.rs` benchmark placeholder
- `dirs` 6 dependency for cross-platform home directory resolution
- `serde_json` 1 dependency for `config.json` parsing
- `clap` 4 dependency (optional, behind `cli` feature)
- Full README with installation, usage, architecture, and configuration docs
- CI: `--all-features` clippy check; publish workflow: `cargo doc` check

### Changed

- `CONVENTIONS.md` renumbered to match upstream Grit (Rules 1–12); added Example column to annotation patterns table

## [0.3.1] — Concurrency & API Additions

### Added

- `download_files()` and `download_files_with_config()` async APIs returning a `HashMap<String, PathBuf>` (filename → path map)
- `download_files_blocking()` and `download_files_with_config_blocking()` sync wrappers
- Concurrent file downloads using `FetchConfig::concurrency` (default 4) via `tokio::task::JoinSet` + semaphore; previously files were downloaded sequentially
- `tokio` `sync` feature for semaphore-based concurrency limiting

### Fixed

- Broken rustdoc link to `IndicatifProgress` when building docs without the `indicatif` feature

## [0.3.0] — Phase 2: Reliability

### Added

- Retry with exponential backoff + jitter (base 300ms, cap 10s, configurable max retries, default 3)
- SHA256 checksum verification against `HuggingFace` LFS metadata via direct REST API call
- Per-file and overall timeout configuration on `FetchConfig`
- Structured error reporting: `FetchError::PartialDownload` with per-file `FileFailure` details (filename, reason, retryable flag)
- `FetchError::Checksum` variant for hash mismatches
- `FetchError::Timeout` variant for exceeded time limits
- `FetchError::Http` variant for direct API call failures
- `FileFailure` struct re-exported from public API
- `FetchConfig` builder methods: `timeout_per_file()`, `timeout_total()`, `max_retries()`, `verify_checksums()`
- `repo::list_repo_files_with_metadata()` for extended HF API metadata (file sizes, SHA256)
- New modules: `checksum.rs` (SHA256 verification), `retry.rs` (exponential backoff)
- `reqwest` 0.12, `serde` 1, `serde_json` 1, `sha2` 0.10 dependencies
- `tokio` `time` feature for timeout support
- Reliability integration tests (checksum, retry, timeout, nonexistent repo)

## [0.2.0] — Phase 1: Progress & Filtering

### Added

- `FetchConfig` builder with `revision`, `token`, `filter` (glob), `exclude` (glob), `concurrency`, and `on_progress` callback
- `download_with_config(repo_id, &config)` async API for configured downloads
- `download_blocking()` and `download_with_config_blocking()` sync wrappers for non-async callers
- `ProgressEvent` struct: `filename`, `bytes_downloaded`, `bytes_total`, `percent`, `files_remaining`
- `Filter` presets: `safetensors()`, `gguf()`, `config_only()`
- Optional `indicatif` feature gate with `IndicatifProgress` multi-bar helper
- `FetchError::InvalidPattern` variant for malformed glob patterns
- `globset` 0.4 dependency for file filtering
- Filter and progress integration tests

## [0.1.0] — Phase 0: Minimal Viable Download

### Added

- `download(repo_id)` async function — downloads all files from a HuggingFace model repository using high-throughput mode
- `FetchError` enum with `Api`, `Io`, `RepoNotFound`, and `Auth` variants (`#[non_exhaustive]`, `thiserror`-based)
- Repo file listing via `hf-hub`'s `info()` API (`repo::list_repo_files()`)
- Download orchestration using `hf-hub`'s `.get()` with `.high()` builder (`download::download_all_files()`)
- HuggingFace cache layout compatibility (`~/.cache/huggingface/hub/`)
- Authentication via `HF_TOKEN` environment variable (delegated to `hf-hub`)
- Integration test downloading `julien-c/dummy-unknown`
- CI pipeline: `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test`
- Publish workflow with `workflow_dispatch` for manual re-runs
- Grit coding conventions enforced via `[lints]` in `Cargo.toml`
- SPDX headers (`MIT OR Apache-2.0`) on all `.rs` files
