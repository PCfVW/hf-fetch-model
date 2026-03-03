# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
