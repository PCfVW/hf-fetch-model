# Architecture

## Stack

```
candle-mi
  download_model() convenience fn
         │ optional dep (feature = "fast-download")
hf-fetch-model
  • repo file listing
  • file filtering (glob patterns)
  • parallel file orchestration
  • multi-connection Range downloads (large files)
  • progress callbacks
  • checksum verification
  • resume / retry
  • cache diagnostics & model search
         │ dep
hf-hub (tokio, .high())
  • single-connection download (.high() mode)
  • HF cache layout compatibility
  • auth token handling
```

## What hf-fetch-model adds over hf-hub

`hf-hub` provides single-file downloads with HuggingFace cache compatibility. hf-fetch-model wraps it and adds:

- **Repo-level orchestration** — download all files in a model repository with one call
- **Multi-connection chunked downloads** — large files (≥100 MiB) are split into chunks downloaded in parallel via HTTP Range requests (8 connections by default)
- **File filtering** — glob patterns and presets to select which files to download
- **Progress reporting** — per-file callbacks with optional `indicatif` progress bars
- **Checksum verification** — SHA256 against HuggingFace LFS metadata
- **Retry with backoff** — exponential backoff + jitter for transient failures
- **Timeout control** — per-file and overall time limits
- **Cache diagnostics** — inspect download state (complete / partial / missing) per file
- **Model search** — query the HuggingFace Hub API for models, with multi-term filtering and model card metadata

## Module layout

| Module | Visibility | Role |
|--------|-----------|------|
| `lib.rs` | public | Top-level download functions and re-exports |
| `config` | public | `FetchConfig` builder and `Filter` presets |
| `download` | public | Download orchestration, `DownloadOutcome` |
| `discover` | public | Search, model card, `GateStatus` |
| `cache` | public | Cache inspection and status |
| `checksum` | public | SHA256 verification |
| `error` | public | `FetchError` and `FileFailure` |
| `progress` | public | `ProgressEvent` and `IndicatifProgress` |
| `repo` | public | Repository file listing via HF API |
| `chunked` | private | Multi-connection Range download engine |
| `retry` | private | Retry policy with exponential backoff |
