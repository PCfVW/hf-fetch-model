# Rust Ecosystem Comparison

**Date surveyed:** June 2026
**hf-fetch-model version at survey:** v0.10.6
**Scope:** Rust crates and binaries only. Python tooling (`huggingface-cli`/`hf`, `accelerate`) is out of scope for the matrix by design — this comparison exists to help downstream Rust consumers evaluate alternatives in their own ecosystem. A few Python analogs are named in prose as cross-ecosystem context (the way `vramio` was in the original survey), but they are never scored in the matrix.

> **Revision note (June 2026).** This survey was re-baselined from v0.10.0 to v0.10.6. The whole v0.10.1–v0.10.6 line shipped in the interim — most consequentially, the GPU-fit verdict (`inspect --check-gpu`) went from *planned* to *shipped and deepened* (KV-cache + hybrid-Mamba budgeting). Competitor facts were re-verified against crates.io / lib.rs / GitHub as of June 2026; several versions and two strategic claims moved. Items that could not be verified to a primary source are flagged inline.

---

## What hf-fetch-model does (the comparison reference)

`hf-fetch-model` (binary alias `hf-fm`) is a Rust CLI binary AND embeddable library for working with HuggingFace model repos. The 13 capability areas used as comparison axes:

1. **MultiDL** — multi-connection chunked downloads with auto-tuned concurrency, resumable across interruptions via `.chunked.part.state` sidecars, configurable per-file/total timeouts (hard wall-clock cap, including in-flight files, since v0.10.5), glob filters (`--filter`/`--exclude`), filter presets (`safetensors`/`gguf`/`pth`/`npz`/`config-only`), gated-model preflight auth.
2. **DLFile** — single-file downloads with glob expansion (`download-file org/model "pytorch_model-*.bin"`).
3. **Inspect** — header reading via two HTTP Range requests for remote `.safetensors` (no weight bytes), plus **cached inspection of four formats** — `.safetensors`, `.gguf`, `.npz`, `.pth` (the latter three local-cache only, via the `anamnesis` parser crate; remote support is the v0.11 line). Tensor names, shapes, dtypes, `__metadata__`. `--tree` (hierarchical view, numeric sibling collapsing `[0..N]`), `--dtypes` (per-dtype histogram), `--limit`, `--filter` (case-insensitive since v0.10.6), `--cached`, `--list`, **`--pick`** (interactive file selection, v0.10.5), `--json`. **Quantization detection** for cached safetensors (`Format:`/`Size:` for FP8 / GPTQ / AWQ / BnB, v0.10.3). Aggregation across sharded repos (`model.safetensors.index.json`), with repo-level `--filter` listing matched tensor names (v0.10.6).
4. **Diff** — tensor-layout comparison between two models (only-in-A, only-in-B, dtype/shape diffs, matching), plus **`--dtypes`** side-by-side per-dtype histograms with a Δ column (v0.10.2).
5. **Du** — cache disk usage with numbered list, drill-down by index, `--age` (last-modified), `--tree` (full cache tree with box-drawing connectors), partial-download markers (`●`).
6. **CacheMgmt** — `cache delete`, `cache clean-partial`, `cache gc --older-than/--max-size --except --dry-run`, `cache verify` (re-checks SHA256 against HF LFS metadata, with streaming spinner progress), `cache path` (prints snapshot path for shell substitution).
7. **Search** — Hub search with multi-term filtering, `--library`, `--pipeline`, `--tag`, `--exact` for full-id match with model card display.
8. **Info** — model card + README display.
9. **Status** — per-repo file-level cache state (Complete / Partial / Missing / Excluded), with per-file partial attribution (v0.10.5).
10. **ListFiles** — remote file listing with sizes + SHA256, optional `--show-cached` cross-reference.
11. **Discover** — `model_type` grouping over local cache, Hub discovery of new families.
12. **Lib** — embeddable Rust library API. Existing consumers: `candle-mi`, `anamnesis`. Uses `hf-hub 0.5` as a transitive dep but adds significant orchestration on top.
13. **VRAMFit** — "does this model fit on my GPU?" verdict from metadata only (no weight download). **Shipped v0.10.1**; deepened in v0.10.4 with KV-cache budgeting at a user-supplied `--context N` and hybrid Mamba/attention accounting. See the dedicated section below.

The bin is `hf-fm` (alias `hf-fetch-model`); installed via `cargo install hf-fetch-model --features cli`. GPU detection is delegated to `hypomnesis 0.2.3` (NVML / DXGI / `nvidia-smi`, plus macOS Metal since 0.2.3).

---

## Per-tool detail

### A. Direct competitors (Rust HuggingFace clients/CLIs)

**hf-hub** — [github.com/huggingface/hf-hub](https://github.com/huggingface/hf-hub) | [crates.io/crates/hf-hub](https://crates.io/crates/hf-hub)
Latest **stable v0.5.0 (2026-02-19)**, but a **1.0 line is in release-candidate: `1.0.0-rc.1` (2026-05-07)** is the newest published version. ~306 stars, official HuggingFace-maintained. Library-first; ships an `hfrs` CLI binary (terminal interface to the Hub). Covers repo metadata (model/dataset/space CRUD), file upload/download, list-tree, commit history, branch/tag mgmt, whoami, async streaming pagination, **Xet transfers** (HF's dedup storage backend), sync + async APIs. Does NOT cover: safetensors inspection, hub search, cache gc/verify/du, diff, dtype histograms. Used as a transitive dep by hf-fm itself, plus `tokenizers`, `fastembed`, `lancedb`, `candle-*`, etc. — **~423 reverse deps** (crates.io API; the original "~70" figure was badly undercounted). *Downstream note: 1.0 is RC, not stable on crates.io — the project's deferred `reqwest 0.13` bump is still gated on a stable 1.0 release.*

**huggingface-hub** (a.k.a. `huggingface_hub_rust`) — [github.com/huggingface/huggingface_hub_rust](https://github.com/huggingface/huggingface_hub_rust) | [crates.io/crates/huggingface-hub](https://crates.io/crates/huggingface-hub) — **NEW, not in prior survey**
Crate **v0.0.0 (first published 2026-03-26)**, by an HF Xet engineer. A **second, official** Rust Hub client distinct from `hf-hub` — a typed, ergonomic interface aiming at full parity with the Python `huggingface_hub` (repo/file/commit ops, branch/tag mgmt, user/org info, streaming pagination, optional Xet). Pre-alpha (placeholder `0.0.0`, minimal downloads) — the **repo** is the real artifact, not yet the crate. Strategically important to track: if HF promotes this over `hf-hub`, the downstream Rust dependency landscape shifts. Library-only; no inspect/search/cache/diff.

**rust-hf-downloader** — [crates.io/crates/rust-hf-downloader](https://crates.io/crates/rust-hf-downloader) | [docs.rs](https://docs.rs/crate/rust-hf-downloader/latest)
**v1.4.0 (2026-02-13)** — unchanged since the prior survey. Binary-only (not a library; confirmed by docs.rs). Closest functional competitor to hf-fm on the download path: parallel chunked downloads with adaptive chunk sizing, resume, SHA256 verification, gated-model auth, hub search with downloads/likes/recency filters, multi-part GGUF grouping, rate-limit token bucket, queue management. Has BOTH a TUI (vim-like keys, mouse) and a headless CLI mode, persistent config in `~/.config/jreb/config.toml`. Lacks: safetensors inspection, diff, cache gc/verify/du subcommands, library API.

**model-hub** — [github.com/waitsalt/model-hub](https://github.com/waitsalt/model-hub) | [crates.io/crates/model-hub](https://crates.io/crates/model-hub) — **NEW, not in prior survey**
**v0.1.1 (2026-04-15)**, very new. Async Rust downloader for **both HuggingFace and ModelScope** — the dual-provider angle is novel among the surveyed downloaders. Concurrent transfers, exponential-backoff retries, resume, path-traversal protection, token auth, file filtering. No cache management, inspect, or search.

**rustyface** — [crates.io/crates/rustyface](https://crates.io/crates/rustyface) | [github.com/AspadaX/RustyFace](https://github.com/AspadaX/RustyFace)
**v0.1.3 (2025-07-30)**, MIT, CLI-only — unchanged. Lightweight pure-Rust whole-repo downloader: concurrent downloads (`--tasks N`), SHA-256 verification, mirror support (default `hf-mirror.com`, base URL configurable via env var). No glob filters, no inspect, no search, no cache mgmt. Niche: works in restricted networks.

**facecrab** — [crates.io/crates/facecrab](https://crates.io/crates/facecrab) | [github.com/tmzt/rusty-genius](https://github.com/tmzt/rusty-genius)
**v0.1.6 (2026-02-08)**, MIT, library only. Component of the `rusty-genius` orchestration system ("The Supplier"). Resolves HF repo IDs to local cached file paths, maintains a `registry.toml` index, async. Used by ogenius. No inspect/diff/search/gc. *(The prior survey's "~565 LOC / async-std" internals could not be re-verified to a primary source — treat as indicative.)*

**ogenius** — [crates.io/crates/ogenius](https://crates.io/crates/ogenius) | [github.com/tmzt/rusty-genius](https://github.com/tmzt/rusty-genius)
**v0.1.6 (2026-02-08)** (prior survey said 0.1.5). CLI binary — "a simpler alternative to Ollama" — with subcommands `chat`, `download <repo>`, `serve` (OpenAI-compatible API server, default port 8080; CUDA/Metal/Vulkan accel). Downloads via facecrab. No inspect/diff/cache/search.

**hugging-face-client** — [crates.io/crates/hugging-face-client](https://crates.io/crates/hugging-face-client)
**v0.6.0 (2025-05-14)**, library-only, **inactive** (no 2026 releases). A separate small Rust implementation of the Hub API. Minimal documentation/usage.

**hf-hub-enfer** — [crates.io/crates/hf-hub-enfer](https://crates.io/crates/hf-hub-enfer)
**v0.3.2 (2024-12-06)**, a fork/variant of hf-hub targeting parity with Python `huggingface_hub`. **Effectively abandoned** (no activity since Dec 2024).

### B. Tensor format inspectors in Rust

**safetensors_explorer** — [github.com/EricLBuehler/safetensors_explorer](https://github.com/EricLBuehler/safetensors_explorer) | [crates.io](https://crates.io/crates/safetensors_explorer)
**v0.2.0 (2025-07-28)**, ~58 stars, MIT, by Eric Buehler (also `mistral.rs` author). Closest analog to hf-fm's `inspect`. Interactive TUI (crossterm) with hierarchical tree view, fuzzy search (`/`), expandable groups, glob patterns, sharded model index detection, multi-file unified view, supports BOTH `.safetensors` AND `.gguf`. CLI-only (no library API documented). Lacks: dtype histograms, JSON output, diff between two files, hub downloads, remote inspection over HTTP Range, sibling-collapsing tree compaction. Reads metadata only (memory-efficient). No release since the prior survey.

**safetensors-browser** — [crates.io/crates/safetensors-browser](https://crates.io/crates/safetensors-browser) — **NEW, not in prior survey**
**v0.2.0 (2026-02-19)**, dual MIT/Apache-2.0, by Daniël de Kok (a HuggingFace engineer). A compact Rust CLI binary (~956 LOC) described simply as a "Browser for Safetensors checkpoints" — browses and searches safetensors metadata. *(crates.io lists no repository and only a one-line description. The prior research pass's richer claims — an interactive `ratatui` TUI and **remote/Hub** inspection via `hf-hub` — could **not** be confirmed against any primary source and are NOT asserted here; scored as a local safetensors browser only. Worth re-checking as it matures, given the author's HF affiliation.)*

**llm_hunter** — [github.com/jpegleg/llm_hunter](https://github.com/jpegleg/llm_hunter) — **NEW, not in prior survey**
**v0.3.5 (created 2026-04-07)**. Rust library + CLI for **forensic *identification*** of model files — GGUF-focused, plus generic binary entropy profiling. Detects entropy *transitions* (byte offsets where randomness shifts), recognises container/format signatures (GGUF / ZIP / Pickle / PyTorch / HDF5 *structure*), fingerprints model family and quantisation scheme by scanning text/byte regions, and emits JSON reports (`quick` vs `deep` modes). Its README **explicitly disclaims** malware scanning, pickle-opcode analysis, and tamper detection — so it is identification/profiling, **not** security threat assessment, and it does not overlap hf-fm's structural `inspect`.

**safetensors-cli** — [github.com/gzsombor/safetensors-cli](https://github.com/gzsombor/safetensors-cli) | [crates.io](https://crates.io/crates/safetensors-cli)
**v0.1.0 (2023-06-17)**, 0 stars, Apache-2.0. Tiny (~57 LOC) inspect tool using memmap2. No feature work since 2023, but **bot-maintained** (Renovate dependency bumps through 2026-02-15) — "unmaintained" is too strong; "dormant on features, deps kept current" is accurate. Local files only.

**safetensors** (canonical Rust crate) — [github.com/huggingface/safetensors](https://github.com/huggingface/safetensors)
**v0.8.0 (2026-06-09)**, ~3.8k stars. Library only, no CLI. Foundation for many other tools (the repo focuses on the format spec + Python bindings). v0.8.0 added direct-to-Metal loading on Apple Silicon, GIL-free serialization, a `TensorSpec` metadata class, Windows ARM64 / RISC-V64 support, and new fp8 dtypes. *(The canonical repo is `huggingface/safetensors`; the prior survey's `safetensors/safetensors` path redirects to it.)*

**gguf-rs-lib / `gguf-cli`** (ThreatFlux) — [github.com/ThreatFlux/gguf](https://github.com/ThreatFlux/gguf) | [crates.io/crates/gguf-rs-lib](https://crates.io/crates/gguf-rs-lib)
**v0.2.5 (2025-09-02)**, ~5 stars. Library (**published as `gguf-rs-lib`**, repo `threatflux/gguf_rs`) + `gguf-cli` binary. Subcommands: `info`, `tensors`, `metadata --format json`, `validate`. Local files only — no HF integration. Zero-copy parsing, optional mmap, async feature. **Naming caution:** the prior survey called this "gguf-rs" and gave a `cargo install gguf …` command — both are stale. The crate was renamed (`gguf` → `gguf-rs` → `gguf-rs-lib`), and **`gguf-rs` on crates.io is now a *different, more active* crate** (see below).

**gguf-rs** (Zack Shen) — [github.com/zackshen/gguf](https://github.com/zackshen/gguf) | [crates.io/crates/gguf-rs](https://crates.io/crates/gguf-rs) — **NEW (disambiguation)**
**v0.1.8 (2026-06-15)**, an actively-maintained GGUF parsing **library** (format versions 1–3, mmap, async) — unrelated to the ThreatFlux tool above despite the colliding name. Library, not a CLI inspector.

**inspector-gguf** — [github.com/FerrisMind/inspector-gguf](https://github.com/FerrisMind/inspector-gguf) | [docs.rs/inspector-gguf](https://docs.rs/inspector-gguf)
**v0.3.1 (2025-12-15)**, Apache-2.0 (was MIT at 0.3.0), ~3 stars. GUI (egui drag-and-drop) + CLI + library. Exports analysis as CSV/YAML/Markdown/HTML/PDF. Tokenizer + chat-template analysis. Operates on local files (makes network calls only for GitHub update-checking). Heavyweight compared to alternatives.

### C. Adjacent: ML inference frameworks with download tooling

**candle** — [github.com/huggingface/candle](https://github.com/huggingface/candle)
candle-core **v0.10.2 (2026-04-01)**, ~20.5k stars, ~1.6k forks, ~2,643 commits. Active. Pulls models via `hf-hub` per-example — **no centralized CLI for downloads or inspection**. Each example binary (`cargo run --example <name>`) handles its own download. Overlap is incidental, via library composition.

**mistral.rs** — [github.com/EricLBuehler/mistral.rs](https://github.com/EricLBuehler/mistral.rs)
**v0.8.5 (2026-06-16)**, ~7.3k stars (the `mistralrs` crate on crates.io lags at v0.8.1). The `mistralrs` CLI handles auto-download (`mistralrs run -m user/model`), `tune` (auto-config), `quantize` (UQFF creation), `doctor` (CUDA/Metal/HF connectivity diagnostics), `serve` (web UI is now **default-on**, disabled with `--no-ui`), plus newer `from-config` / `bench`. Downloads happen as a side-effect of running models — **no standalone download/inspect/diff/cache subcommands**. Same author as `safetensors_explorer`. *(The exact `--ui`/`--no-ui` flag wording should be confirmed against live `--help` before quoting.)*

**apr-cli** — [docs.rs/crate/apr-cli/latest](https://docs.rs/crate/apr-cli/latest) | part of [paiml/aprender](https://github.com/paiml/aprender)
**v0.41.0 (2026-06-11)** on crates.io (the `aprender` monorepo itself is at v0.49.1; the CLI crate publishes behind it). Rust monorepo (~81 workspace crates, **~103 CLI commands**). Inference orchestration tool centered on a custom "APR" model format with its own Q4K quantization. Relevant overlap: `apr serve plan` accepts a HuggingFace model ID (`hf://org/model`) and emits a **READY / WARNINGS / BLOCKED** verdict on whether the model fits the user's GPU before downloading, fetching only metadata (the `~2 KiB config.json` via `huggingface.co/{repo}/raw/main/config.json`, no weights). Supports SafeTensors and GGUF for inference (loaded locally) and ships a `quantize` subcommand that streams SafeTensors → APR Q4K. **A Rust VRAM-fit peer** — see the VRAMFit section. Mechanism: a **config.json parameter-count heuristic** (`estimate_param_string()` from `hidden_size`/`num_hidden_layers`/etc.), not summed tensor byte-sizes; live VRAM via **`nvidia-smi --query-gpu=memory.total`** (nameplate capacity, not free), with a 5%-margin fit contract. *(Source reads were against repo HEAD/v0.49.1; published 0.41.0 may differ slightly.)*

**burn** — [github.com/tracel-ai/burn](https://github.com/tracel-ai/burn)
**v0.21.0 (2026-05-07)**, ~15.4k stars. No CLI for HF model management. Library-first; weights loaded via in-code APIs (PyTorch/Safetensors/ONNX import). Out of scope.

**llm crate** — [crates.io/crates/llm](https://crates.io/crates/llm) | [github.com/graniet/llm](https://github.com/graniet/llm)
**Correction:** the prior survey's "effectively superseded; replaced by `mistral.rs` / `candle`" described the **archived `rustformers/llm`** GGML library. The crates.io `llm` name is now owned by **`graniet/llm` (v1.3.8, 2026-04-19)** — an *active* multi-backend LLM API client (OpenAI/Anthropic/Gemini/Ollama/…), not a local inference engine. Neither does HF download/inspect/cache; listed only to correct the stale identity.

### D. Adjacent: GPU-fit / VRAM estimation

**llmfit** — [github.com/AlexsJones/llmfit](https://github.com/AlexsJones/llmfit) | [crates.io/crates/llmfit](https://crates.io/crates/llmfit) — **NEW, not in prior survey; the key VRAMFit miss**
**v0.9.31 (2026-06-09)**, MIT, very active (Rust 2024 edition). A **Rust** "which models fit my hardware?" tool — TUI + CLI — that emits an explicit fit verdict (**Perfect / Good / Marginal / Too Tight**) across 200+ models, picks the highest-quality quantization that fits available RAM/VRAM (Q8_0 → Q2_K hierarchy), and is **MoE-aware** (active-vs-total params). KV-cache enters its tok/s estimate. VRAM detection is broader than apr-cli's: NVIDIA `nvidia-smi`, AMD `rocm-smi`, Intel Arc sysfs, Apple `system_profiler`, Ascend `npu-smi`, plus a `--memory` override. **Falsifies the prior survey's claim that apr-cli was the only Rust fit tool.** (By the same author as `llmserve`, below.)

### E. Adjacent: cache management

**llmserve** — [crates.io/crates/llmserve](https://crates.io/crates/llmserve) | [github.com/AlexsJones/llmserve](https://github.com/AlexsJones/llmserve)
**v0.0.7 (2026-04-12)**, MIT. TUI for orchestrating *inference servers* (llama-server, KoboldCpp, LocalAI, MLX, Ollama, vLLM, LM Studio). Discovers models in the LM Studio cache, HF hub cache (`~/.cache/huggingface/hub/`), and llama.cpp cache. Three-panel layout, vim keys, themes. Overlap: cache *discovery* only. Not gc/verify/diff.

**diskard** — [github.com/connectwithprakash/diskard](https://github.com/connectwithprakash/diskard) | [crates.io/crates/diskard](https://crates.io/crates/diskard) — **NEW, not in prior survey**
**v0.2.0 (2026-02-19)**, MIT OR Apache-2.0. A **general-purpose** Rust disk cleaner that scans and deletes the HF cache (`~/.cache/huggingface`) as 1 of 18 recognizers. It does du/scan + bulk delete (overlapping hf-fm's `du` and a crude `gc`), but **no integrity verify** and **no revision-aware pruning**. It weakens — without breaking — the "no Rust HF-cache tool" claim: the *dedicated, HF-aware* niche (revision-gc + LFS verify + indexed du) still has no Rust competitor.

**No *dedicated, HF-cache-aware* Rust tool** for `~/.cache/huggingface/hub/` lifecycle (gc with `--older-than`/`--max-size`/`--except`, verify against LFS metadata, du with tree view, partial-download cleanup) was found. hf-fm's `cache` subcommand group remains unique on the Rust side; `diskard` deletes but cannot verify or prune by revision. The closest cross-ecosystem analog is the Python `hf cache` group (`ls` / `rm` / `prune` / `verify`), which is itself under active redesign ([RFC #3432](https://github.com/huggingface/huggingface_hub/issues/3432)).

### Cross-ecosystem context (Python — not scored in the matrix)

- **`hf cache verify`** — the Python `huggingface_hub` CLI now checksums cached (or local-dir) files against the Hub, warns on missing/extra files, and exits non-zero on mismatch. The direct analog to hf-fm's `cache verify`.
- **`hf-mem`** — [github.com/alvarobartt/hf-mem](https://github.com/alvarobartt/hf-mem) reads safetensors/GGUF metadata via **HTTP Range requests (~first 100 KB), no download**, to estimate memory — the *same* header-range trick hf-fm's `inspect` uses. It is the upstream of `vramio` (the Python web service the prior survey already cited).
- **`accelerate estimate-memory`** — the canonical official fit estimator: loads the model on a meta device (no weights) and reports largest-layer / total / training memory per dtype. The reference point any VRAM-fit tool is measured against.

### Notable observations

- The `hf-hub` crate has **~423 reverse deps**, dominated by libraries (`tokenizers`, `fastembed`, `lancedb`, `candle-*`) — mostly library-on-library composition, not user-facing tools.
- HuggingFace now has **two** official Rust Hub clients in flight: the established `hf-hub` (1.0-rc) and the experimental, parity-focused `huggingface_hub_rust` / `huggingface-hub` crate. Their eventual relationship is unresolved and worth watching.
- Two of EricLBuehler's projects (`mistral.rs`, `safetensors_explorer`) cover adjacent territory but never merged the inspect tooling into the inference CLI.
- One author (**AlexsJones**) ships *both* a cache-discovery TUI (`llmserve`) and a VRAM-fit TUI (`llmfit`) — the closest single-author footprint to two of hf-fm's axes, though split across two binaries and neither a library.
- `rust-hf-downloader` is still the closest direct competitor on download/search, but ships no inspect/diff/cache subcommands and is binary-only (no library reuse for downstream Rust apps like `candle-mi` / `anamnesis`).

---

## Comparison matrix

Cells: ✓ = present · ◐ = partial · — = absent.

Capability headers (the 13 hf-fm areas above):

| # | Header | Capability |
|---|---|---|
| 1 | MultiDL | multi-connection chunked download with resume + filters/presets |
| 2 | DLFile | single-file download with glob expansion |
| 3 | Inspect | tensor-header inspection (tree, dtypes, sharded aggregation, JSON; 4 formats cached) |
| 4 | Diff | tensor-layout diff between two models |
| 5 | Du | cache disk-usage drill-down + tree |
| 6 | CacheMgmt | cache delete / clean-partial / gc / verify (SHA256) / path |
| 7 | Search | Hub search with library/pipeline/tag filters |
| 8 | Info | model card / README display |
| 9 | Status | per-repo file-level cache state |
| 10 | ListFiles | remote file listing with sizes + SHA256 |
| 11 | Discover | `model_type` grouping / Hub family discovery |
| 12 | Lib | embeddable Rust library API |
| 13 | VRAMFit | "does this model fit my GPU?" verdict from metadata only (no weight download) |

| Tool | MultiDL | DLFile | Inspect | Diff | Du | CacheMgmt | Search | Info | Status | ListFiles | Discover | Lib | VRAMFit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **hf-fetch-model v0.10.6** *(reference)* | ✓ | ✓ | ✓ 4 fmts, tree, dtypes, quant, sharded, JSON | ✓ +dtypes | ✓ tree | ✓ delete/clean-partial/gc/verify/path | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ +KV-cache/ctx |
| hf-hub 1.0.0-rc.1 / 0.5.0 | ◐ resume, single-conn | ◐ no glob | — | — | — | — | — | ◐ metadata only | — | ◐ list-tree | — | ✓ | — |
| huggingface-hub 0.0.0 *(official, experimental)* | ◐ | ◐ | — | — | — | — | — | ◐ metadata | — | ◐ | — | ✓ | — |
| model-hub 0.1.1 | ✓ concurrent+resume | ◐ no glob | — | — | — | — | — | — | — | — | — | ✓ (HF+ModelScope) | — |
| rust-hf-downloader 1.4.0 | ✓ adaptive chunks | ◐ no glob | — | — | — | — | ✓ TUI search | — | — | — | — | — bin only | — |
| rustyface 0.1.3 | ◐ concurrent, no chunks | — repo-only | — | — | — | — | — | — | — | — | — | — bin only | — |
| facecrab 0.1.6 | — single-conn | ✓ resolve-by-id | — | — | — | ◐ registry.toml | — | — | — | — | — | ✓ | — |
| ogenius 0.1.6 | — | ✓ via facecrab | — | — | — | — | — | — | — | — | — | ◐ via facecrab | — |
| hugging-face-client 0.6.0 | ◐ basic | ◐ basic | — | — | — | — | — | — | — | ◐ | — | ✓ | — |
| hf-hub-enfer 0.3.2 *(abandoned)* | ◐ | ◐ | — | — | — | — | — | — | — | ◐ | — | ✓ | — |
| safetensors_explorer 0.2.0 | — | — | ✓ TUI tree, sharded, fuzzy | — | — | — | — | — | — | — | — | — bin only | — |
| safetensors-browser 0.2.0 | — | — | ✓ local browser | — | — | — | — | — | — | — | — | — bin only | — |
| llm_hunter 0.3.5 | — | — | ◐ forensic ID (GGUF + entropy) | — | — | — | — | — | — | — | — | ✓ | — |
| safetensors-cli 0.1.0 | — | — | ◐ minimal | — | — | — | — | — | — | — | — | — bin only | — |
| safetensors 0.8.0 | — | — | — no CLI | — | — | — | — | — | — | — | — | ✓ format only | — |
| gguf-rs-lib 0.2.5 *(ThreatFlux)* | — | — | ◐ GGUF only (info/tensors/meta/validate) | — | — | — | — | — | — | — | — | ✓ | — |
| gguf-rs 0.1.8 *(Zack Shen)* | — | — | — parse lib, no CLI | — | — | — | — | — | — | — | — | ✓ format only | — |
| inspector-gguf 0.3.1 | — | — | ◐ GGUF only, GUI+CLI | — | — | — | — | — | — | — | — | ✓ | — |
| candle 0.10.2 | — via hf-hub | — via hf-hub | — | — | — | — | — | — | — | — | — | ✓ framework | — |
| mistral.rs 0.8.5 | ◐ via hf-hub | ◐ via hf-hub | — | — | — | ◐ doctor only | — | — | — | — | — | ✓ | — |
| apr-cli 0.41.0 | ◐ for inference | — | ◐ APR-format-centric | — | — | — | — | — | — | — | — | ✓ via aprender | ✓ `serve plan`, config.json-based |
| llmfit 0.9.31 | — | — | — | — | — | — | — | — | — | — | ◐ model browser | ✓ lib+bin | ✓ multi-vendor, quant/MoE |
| burn 0.21.0 | — | — | — | — | — | — | — | — | — | — | — | ✓ framework | — |
| llmserve 0.0.7 | — | — | — | — | ◐ discovery | ◐ multi-cache discovery | — | — | ◐ status | — | ◐ across caches | — bin only | — |
| diskard 0.2.0 | — | — | — | — | ◐ scan | ◐ delete-only (no verify/gc-by-rev) | — | — | — | — | — | — bin only | — |

---

## Gaps unique to hf-fm in the surveyed Rust ecosystem

- **Cache `gc` / `verify` / `clean-partial`** — no other Rust tool offers HF-aware revision-gc + LFS-checksum verify. `diskard` can delete the HF cache but cannot verify or prune by revision; `llmserve` only discovers.
- **Hub-side `inspect` over HTTP Range** (no full download) combined with sharded `model.safetensors.index.json` aggregation — `safetensors_explorer`, `safetensors-browser`, and the GGUF inspectors read local files only; none reads metadata over HTTP Range.
- **Four-format cached inspect under one CLI** (`.safetensors` / `.gguf` / `.npz` / `.pth`) with shared tree/dtypes/quant rendering — the other inspectors are single-format or single-format-family.
- **`diff` between two tensor layouts** (with `--dtypes` histograms) — no Rust analog found.
- **`du --tree` with box-drawing connectors over the HF cache layout** with index drill-down — only `llmserve`/`diskard` do coarser cache discovery/scan.
- **Library + CLI parity** — `rust-hf-downloader`, `safetensors_explorer`, `safetensors-browser`, `llmserve`, `diskard` are all bin-only; hf-fm exposes the same orchestration to embedders (`candle-mi`, `anamnesis`).

The closest competitor footprint is now "hf-hub (lib) + rust-hf-downloader (download CLI) + safetensors_explorer / safetensors-browser (inspect) + llmfit (VRAM-fit) + diskard (cache delete)" — five+ separate tools whose union still does not cover `diff`, HF-aware `gc`/`verify`, four-format unified inspect, remote Range-based inspection, or library-and-CLI parity.

## VRAMFit — now shipped (and the niche is contested)

The prior survey listed VRAMFit as forward-looking and conceded "apr-cli got there first." Both framings are now outdated: **hf-fm shipped `inspect --check-gpu` in v0.10.1 and deepened it in v0.10.4**, and **apr-cli is no longer the only other Rust tool** — `llmfit` is an active peer. The three Rust tools occupy distinct points:

| | hf-fm `inspect --check-gpu` | apr-cli `serve plan` | llmfit |
|---|---|---|---|
| Status | shipped v0.10.1, deepened v0.10.4 | shipped | shipped, very active |
| Metadata source | **safetensors header — exact tensor byte-lengths** for weights; `config.json` for KV-cache geometry | `config.json` — param-count heuristic | param-count heuristic + quant hierarchy |
| KV-cache | **✓ at `--context N`** — GQA, sliding-window, hybrid Mamba/attention, MLA-skip | not modeled (weights-fit gate) | enters tok/s estimate |
| Quantization | exact (reads quantized tensor footprints; quant-scheme detection) | APR Q4K target | quant hierarchy, picks best fit |
| Live VRAM source | `hypomnesis` (NVML / DXGI / `nvidia-smi` / macOS Metal) | `nvidia-smi memory.total` (nameplate, not free) | multi-vendor (NVIDIA/AMD/Intel/Apple/Ascend) |
| Shape | thin flag on HF-focused `inspect` | gate inside an inference pipeline | standalone TUI "which models fit?" browser |
| Primary question | "should I download this model?" | "should I serve this model?" | "which quant of which model fits?" |

The defensible differentiator for hf-fm is **precision**: it budgets from *actual* on-disk tensor footprints plus an *architecture-aware* KV-cache model (hybrid Mamba, sliding-window, MLA), where apr-cli and llmfit use param-count heuristics. `llmfit`'s strengths are breadth (multi-vendor VRAM, quant-hierarchy auto-selection, MoE) and its model-browser UX. The niche is genuinely contested now — hf-fm's claim is "most precise single-model pre-download verdict," not "only one."

---

## Strategic read

The `hf-hub` reverse-dependency graph (~423 crates, mostly libraries) confirms a healthy ecosystem of *library composition* over the official client, but still **little user-facing tool development** above it. hf-fm remains largely alone in the "one CLI + library, all capabilities" gap — but the surrounding landscape has thickened since the prior survey: a second official Rust Hub client is incubating (`huggingface_hub_rust`), a Rust VRAM-fit peer shipped (`llmfit`), another local safetensors browser appeared (`safetensors-browser`), and a general cleaner now touches the HF cache (`diskard`). None of these is a library-and-CLI superset; each occupies one or two of hf-fm's axes.

Two structural choices still stand out. **EricLBuehler** maintains both `mistral.rs` and `safetensors_explorer` but kept inspect a standalone TUI rather than bolting it onto the inference CLI. **AlexsJones** likewise splits cache-discovery (`llmserve`) and VRAM-fit (`llmfit`) into two binaries. hf-fm goes the other way: one CLI, one library, all capabilities — which is the position to defend as the individual niches attract dedicated single-purpose tools.

---

## Sources

- [hf-hub GitHub](https://github.com/huggingface/hf-hub) / [crates.io](https://crates.io/crates/hf-hub) (v0.5.0 stable, 1.0.0-rc.1) / [reverse deps API](https://crates.io/api/v1/crates/hf-hub/reverse_dependencies)
- [huggingface_hub_rust](https://github.com/huggingface/huggingface_hub_rust) / [huggingface-hub crate](https://crates.io/crates/huggingface-hub)
- [model-hub](https://github.com/waitsalt/model-hub) / [crates.io](https://crates.io/crates/model-hub)
- [rust-hf-downloader on lib.rs](https://lib.rs/crates/rust-hf-downloader) / [docs.rs](https://docs.rs/crate/rust-hf-downloader/latest)
- [rustyface](https://github.com/AspadaX/RustyFace) / [crates.io](https://crates.io/crates/rustyface)
- [facecrab](https://crates.io/crates/facecrab) / [ogenius](https://crates.io/crates/ogenius) / [rusty-genius](https://github.com/tmzt/rusty-genius)
- [hugging-face-client](https://crates.io/crates/hugging-face-client) / [hf-hub-enfer](https://crates.io/crates/hf-hub-enfer)
- [safetensors_explorer GitHub](https://github.com/EricLBuehler/safetensors_explorer) / [lib.rs](https://lib.rs/crates/safetensors_explorer)
- [safetensors-browser](https://crates.io/crates/safetensors-browser) / [llm_hunter](https://github.com/jpegleg/llm_hunter)
- [safetensors-cli GitHub](https://github.com/gzsombor/safetensors-cli) / [lib.rs](https://lib.rs/crates/safetensors-cli)
- [safetensors canonical](https://github.com/huggingface/safetensors) / [crates.io](https://crates.io/crates/safetensors)
- [gguf-rs-lib / ThreatFlux GitHub](https://github.com/ThreatFlux/gguf) / [crates.io](https://crates.io/crates/gguf-rs-lib) — distinct from [gguf-rs (Zack Shen)](https://github.com/zackshen/gguf)
- [inspector-gguf GitHub](https://github.com/FerrisMind/inspector-gguf) / [docs.rs](https://docs.rs/inspector-gguf)
- [candle GitHub](https://github.com/huggingface/candle) / [candle-core crates.io](https://crates.io/crates/candle-core)
- [mistral.rs GitHub](https://github.com/EricLBuehler/mistral.rs) / [releases](https://github.com/EricLBuehler/mistral.rs/releases)
- [apr-cli on docs.rs](https://docs.rs/crate/apr-cli/latest) / [aprender monorepo](https://github.com/paiml/aprender)
- [llmfit GitHub](https://github.com/AlexsJones/llmfit) / [crates.io](https://crates.io/crates/llmfit)
- [burn GitHub](https://github.com/tracel-ai/burn) / [releases](https://github.com/tracel-ai/burn/releases)
- [llmserve](https://github.com/AlexsJones/llmserve) / [crates.io](https://crates.io/crates/llmserve)
- [diskard](https://github.com/connectwithprakash/diskard) / [crates.io](https://crates.io/crates/diskard)
- [llm crate (graniet/llm)](https://github.com/graniet/llm) — correcting the archived rustformers/llm identity
- VRAMFit landscape context (Python): [hf-mem](https://github.com/alvarobartt/hf-mem) (HTTP-Range header trick, upstream of vramio), [accelerate estimate-memory](https://huggingface.co/docs/accelerate/main/en/usage_guides/model_size_estimator), [vramio](https://github.com/ksingh-scogo/vramio)
- Cross-ecosystem cache context (Python): [hf cache CLI guide](https://github.com/huggingface/huggingface_hub/blob/main/docs/source/en/guides/cli.md) / [Revamp RFC #3432](https://github.com/huggingface/huggingface_hub/issues/3432)
