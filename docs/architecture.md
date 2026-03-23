# Architecture

## Stack

```
candle-mi
  download_model() convenience fn
         │ optional dep (feature = "fast-download")
hf-fetch-model
  • repo file listing
  • file filtering (glob patterns)
  • download plan (dry-run, plan-to-config optimization)
  • parallel file orchestration
  • multi-connection Range downloads (large files)
  • progress callbacks
  • checksum verification
  • resume / retry
  • safetensors header inspection (HTTP Range)
  • cache diagnostics, disk usage & model search
         │ dep
hf-hub (tokio, .high())
  • single-connection download (.high() mode)
  • HF cache layout compatibility
  • auth token handling
```

## What hf-fetch-model adds over hf-hub

`hf-hub` provides single-file downloads with HuggingFace cache compatibility. hf-fetch-model wraps it and adds:

- **Repo-level orchestration** — download all files in a model repository with one call
- **Multi-connection chunked downloads** — large files are split into chunks downloaded in parallel via HTTP Range requests (concurrency and connection count auto-tuned by the download plan optimizer based on file size distribution)
- **File filtering** — glob patterns and presets to select which files to download
- **Progress reporting** — per-file callbacks with optional `indicatif` progress bars
- **Checksum verification** — SHA256 against HuggingFace LFS metadata
- **Retry with backoff** — exponential backoff + jitter for transient failures
- **Timeout control** — per-file and overall time limits
- **Safetensors header inspection** — read tensor metadata (names, shapes, dtypes, offsets) from local cache or remote repos via HTTP Range requests, without downloading full files
- **Cache diagnostics** — inspect download state (complete / partial / missing) per file; disk usage per repo and globally
- **Model search** — query the HuggingFace Hub API for models, with multi-term filtering and model card metadata

## Module layout

| Module | Visibility | Role |
|--------|-----------|------|
| `lib.rs` | public | Top-level download functions and re-exports |
| `config` | public | `FetchConfig` builder and `Filter` presets |
| `plan` | public | `DownloadPlan`, `FilePlan`, plan-to-config optimization |
| `download` | public | Download orchestration, `DownloadOutcome` |
| `discover` | public | Search, model card, `GateStatus` |
| `cache` | public | Cache inspection, status, and disk usage |
| `inspect` | public | Safetensors header parsing, `TensorInfo`, `SafetensorsHeaderInfo`, `ShardedIndex` |
| `checksum` | public | SHA256 verification |
| `error` | public | `FetchError` and `FileFailure` |
| `progress` | public | `ProgressEvent` and `IndicatifProgress` |
| `repo` | public | Repository file listing via HF API |
| `chunked` | private | Multi-connection Range download engine |
| `retry` | private | Retry policy with exponential backoff |
