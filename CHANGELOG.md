# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`list-files` subcommand** — inspect remote repo contents (filenames, sizes, SHA256) without downloading. Supports `--filter`, `--exclude`, `--preset`, `--no-checksum`, and `--show-cached` flags.
- **`file_matches()` public function** — promoted from `pub(crate)` for use outside the download pipeline.
- **`compile_glob_patterns()` public function** — builds compiled glob filters from pattern strings.
- **`DownloadPlan` type** — new public API (`download_plan()`) for computing a download plan (file list, sizes, cache status) without downloading. Includes `recommended_config()` for plan-based optimization of `FetchConfig`.
- **`FilePlan` type** — per-file entry within a `DownloadPlan`.
- **`FetchConfig` accessors** — `concurrency()`, `connections_per_file()`, `chunk_threshold()` public const methods.
- **`--dry-run` flag** — preview what would be downloaded, compare against local cache, and display recommended download settings. Available on the default download command (`hf-fm <REPO_ID> --dry-run`).

### Changed

- **MSRV bumped to 1.88** — aligns with the actual dependency floor (`cookie_store`, `time` already require 1.88). Previously advertised 1.75 but compilation required 1.88 regardless.

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
