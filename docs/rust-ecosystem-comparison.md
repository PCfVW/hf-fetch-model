# Rust Ecosystem Comparison

**Date surveyed:** May 2026
**hf-fetch-model version at survey:** v0.10.0 (in progress)
**Scope:** Rust crates and binaries only. Python tooling (`huggingface-cli`, `accelerate`) is out of scope by design — this matrix exists to help downstream Rust consumers evaluate alternatives in their own ecosystem.

---

## What hf-fetch-model does (the comparison reference)

`hf-fetch-model` (binary alias `hf-fm`) is a Rust CLI binary AND embeddable library for working with HuggingFace model repos. The 12 capability areas used as comparison axes:

1. **MultiDL** — multi-connection chunked downloads with auto-tuned concurrency, resumable across interruptions via `.chunked.part.state` sidecars, configurable per-file/total timeouts, glob filters (`--filter`/`--exclude`), filter presets (`safetensors`/`gguf`/`pth`/`npz`/`config-only`), gated-model preflight auth.
2. **DLFile** — single-file downloads with glob expansion (`download-file org/model "pytorch_model-*.bin"`).
3. **Inspect** — `.safetensors` header reading via two HTTP Range requests (no weight bytes). Tensor names, shapes, dtypes, `__metadata__`. Local cache fast path. `--tree` (hierarchical view, numeric sibling collapsing `[0..N]`), `--dtypes` (per-dtype histogram), `--limit`, `--filter`, `--cached`, `--list`, `--json`. Aggregation across sharded repos (`model.safetensors.index.json`).
4. **Diff** — tensor-layout comparison between two models (only-in-A, only-in-B, dtype/shape diffs, matching).
5. **Du** — cache disk usage with numbered list, drill-down by index, `--age` (last-modified), `--tree` (full cache tree with box-drawing connectors), partial-download markers.
6. **CacheMgmt** — `cache delete`, `cache clean-partial`, `cache gc --older-than/--max-size --except --dry-run`, `cache verify` (re-checks SHA256 against HF LFS metadata, with streaming spinner progress), `cache path` (prints snapshot path for shell substitution).
7. **Search** — Hub search with multi-term filtering, `--library`, `--pipeline`, `--tag`, `--exact` for full-id match with model card display.
8. **Info** — model card + README display.
9. **Status** — per-repo file-level cache state (Complete / Partial / Missing).
10. **ListFiles** — remote file listing with sizes + SHA256, optional `--show-cached` cross-reference.
11. **Discover** — `model_type` grouping over local cache, Hub discovery of new families.
12. **Lib** — embeddable Rust library API. Existing consumers: `candle-mi`, `anamnesis`. Uses `hf-hub 0.5` as transitive dep but adds significant orchestration on top.

The bin is `hf-fm` (alias `hf-fetch-model`); installed via `cargo install hf-fetch-model --features cli`.

---

## Per-tool detail

### A. Direct competitors (Rust HuggingFace clients/CLIs)

**hf-hub** — [github.com/huggingface/hf-hub](https://github.com/huggingface/hf-hub) | [crates.io/crates/hf-hub](https://crates.io/crates/hf-hub)
v0.5.0 (2026-02-19), ~292 stars, official HuggingFace-maintained. Library-first; ships an `hfrs` CLI binary added recently. Covers repo metadata (model/dataset/space CRUD), file upload/download, list-tree, commit history, branch/tag mgmt, whoami, async streaming pagination, Xet transfers, sync + async APIs. Does NOT cover: safetensors inspection, hub search, cache gc/verify/du, diff, dtype histograms. Used as a transitive dep by hf-fm itself, plus `tokenizers`, `fastembed`, `lancedb`, `candle-examples`, etc. (~70 reverse deps).

**rust-hf-downloader** — [crates.io/crates/rust-hf-downloader](https://crates.io/crates/rust-hf-downloader) | [docs.rs](https://docs.rs/crate/rust-hf-downloader/latest)
v1.4.0 (2026-02-13). Binary-only (not a library). Closest functional competitor to hf-fm on the download path: parallel chunked downloads with adaptive chunk sizing, resume, SHA256 verification, gated-model auth, hub search with downloads/likes/recency filters, multi-part GGUF grouping, rate-limit token bucket, queue management. Has BOTH a TUI (vim-like keys, mouse) and a headless CLI mode, persistent config in `~/.config/jreb/config.toml`. Lacks: safetensors inspection, diff, cache gc/verify/du subcommands, library API.

**rustyface** — [crates.io/crates/rustyface](https://crates.io/crates/rustyface) | [lib.rs](https://lib.rs/crates/rustyface)
v0.1.3 (2025-07-30), MIT, CLI-only. Lightweight pure-Rust whole-repo downloader: concurrent downloads, SHA-256 verification, mirror support (`hf-mirror.com` env var). No glob filters, no inspect, no search, no cache mgmt. Niche: works in restricted networks.

**facecrab** — [crates.io/crates/facecrab](https://crates.io/crates/facecrab) | [lib.rs](https://lib.rs/crates/facecrab)
v0.1.6 (2026-02-08), MIT, library only. Component of the `rusty-genius` orchestration system ("The Supplier"). Resolves HF repo IDs to local cached file paths, maintains a `registry.toml` index. Tiny (~565 LOC), async-std-based, used by ogenius. No inspect/diff/search/gc.

**ogenius** — [crates.io/crates/ogenius](https://crates.io/crates/ogenius/0.1.5) | [github.com/tmzt/rusty-genius](https://github.com/tmzt/rusty-genius)
v0.1.5, CLI binary plus components. Three subcommands: `chat`, `download <repo>`, `serve`. Downloads via facecrab. No inspect/diff/cache/search.

**hugging-face-client** — [crates.io/crates/hugging-face-client](https://crates.io/crates/hugging-face-client)
A separate Rust implementation of the Hub API — extremely small footprint, library-only, low recent activity. Listed in keyword search results but minimal documentation/usage.

**hf-hub-enfer** — [crates.io/crates/hf-hub-enfer](https://crates.io/crates/hf-hub-enfer)
A fork/variant of hf-hub. Low download counts, likely a niche fork; documentation not retrievable.

### B. Tensor format inspectors in Rust

**safetensors_explorer** — [github.com/EricLBuehler/safetensors_explorer](https://github.com/EricLBuehler/safetensors_explorer) | [crates.io](https://crates.io/crates/safetensors_explorer)
v0.2.0 (2025-07-28), 51 stars, 5 forks, MIT, by Eric Buehler (also `mistral.rs` author). Closest analog to hf-fm's `inspect`. Interactive TUI (crossterm) with hierarchical tree view, fuzzy search (`/`), expandable groups, glob patterns, sharded model index detection, multi-file unified view, supports BOTH `.safetensors` AND `.gguf`. CLI-only (no library API documented). Lacks: dtype histograms, JSON output, diff between two files, hub downloads, remote inspection over HTTP Range, sibling-collapsing tree compaction. Reads metadata only (memory-efficient).

**safetensors-cli** — [github.com/gzsombor/safetensors-cli](https://github.com/gzsombor/safetensors-cli) | [crates.io](https://crates.io/crates/safetensors-cli)
v0.1.0 (2023-06-17), 0 stars, 21 commits, Apache-2.0. Tiny (57 LOC) inspect tool using memmap2. Effectively unmaintained; useful only as a minimal example. Local files only.

**safetensors** (canonical Rust crate) — [github.com/safetensors/safetensors](https://github.com/safetensors/safetensors)
v0.7.0 (2025-11-19), 3,700+ stars. Library only, no CLI. Foundation for many other tools. The repo focuses on the format spec + Python bindings; no inspect binary.

**gguf-rs / gguf-cli** — [github.com/ThreatFlux/gguf](https://github.com/ThreatFlux/gguf)
v0.2.5 (2025-09-02), 5 stars. Library + `gguf-cli` binary (`cargo install gguf --features=cli`). Subcommands: `info`, `tensors`, `metadata --format json`, `validate`. Local files only — no HF integration. Zero-copy parsing, optional mmap, async feature.

**inspector-gguf** — [docs.rs/inspector-gguf](https://docs.rs/inspector-gguf)
v0.3.1, Apache-2.0. GUI (egui drag-and-drop) + CLI (structopt) + library. Exports analysis as CSV/YAML/Markdown/HTML/PDF. Tokenizer + chat-template analysis. Local files only. Heavyweight compared to alternatives.

### C. Adjacent: ML inference frameworks with download tooling

**candle** — [github.com/huggingface/candle](https://github.com/huggingface/candle)
v0.x, ~20.1k stars, 1.5k forks, 2,614 commits. Active. Pulls models via `hf-hub` per-example — **no centralized CLI for downloads or inspection**. Each example binary (`cargo run --example <name>`) handles its own download. Overlap is incidental, via library composition.

**mistral.rs** — [github.com/EricLBuehler/mistral.rs](https://github.com/EricLBuehler/mistral.rs)
v0.8.0 (2026-04-02), ~7.1k stars. The `mistralrs` CLI handles auto-download (`mistralrs run -m user/model`), `tune` (auto-config), `quantize` (UQFF creation), `doctor` (CUDA/Metal/HF connectivity diagnostics), `serve --ui`. Downloads happen as a side-effect of running models — no standalone download/inspect/diff/cache subcommands. Same author as `safetensors_explorer`.

**apr-cli** — [docs.rs/apr-cli](https://docs.rs/crate/apr-cli/latest) | part of [paiml/aprender](https://github.com/paiml/aprender)
v0.31.2 (2026-04-19), Rust monorepo (~80 workspace crates, ~76 % Rust, 79 CLI commands). Inference orchestration tool centered on a custom "APR" model format with its own Q4K quantization. Relevant overlap: `apr serve plan` accepts a HuggingFace model ID (`hf://org/model`) and emits a `READY` / `WARNINGS` / `BLOCKED` verdict on whether the model fits the user's GPU before downloading, fetching only metadata (~2 KiB `config.json`). Supports SafeTensors and GGUF for inference (loaded locally) and ships a `quantize` subcommand that streams SafeTensors → APR Q4K. **Closest existing analog in Rust to hf-fm's planned `inspect --check-gpu` (v0.10.1).** Positioning is different: apr-cli is heavyweight inference machinery with its own custom format ecosystem; `--check-gpu` would stay a thin flag inside the existing HF-focused `inspect` path. Mechanism is also different — apr-cli reads `config.json` for architectural estimates, whereas `--check-gpu` would use the safetensors header bytes for exact tensor footprints (relevant when the cached files are quantized variants).

**burn** — [github.com/tracel-ai/burn](https://github.com/tracel-ai/burn)
No CLI for HF model management. Library-first; weights loaded via in-code APIs (PyTorch/Safetensors/ONNX import). Out of scope.

**llm crate** — Effectively superseded; replaced by `mistral.rs` / `candle` in current ecosystem.

### D. Adjacent: cache management

**llmserve** — [crates.io/crates/llmserve](https://crates.io/crates/llmserve) | [lib.rs](https://lib.rs/crates/llmserve)
v0.0.7 (2026-04-12), MIT. TUI for orchestrating *inference servers* (llama-server, KoboldCpp, LocalAI, MLX, Ollama, vLLM, LM Studio). Discovers models in the LM Studio cache, HF hub cache (`~/.cache/huggingface/hub/`), and llama.cpp cache. Three-panel layout, vim keys, themes. Overlap: cache *discovery*. Not gc/verify/diff.

**No dedicated Rust tool** for `~/.cache/huggingface/hub/` lifecycle (gc with `--older-than`/`--max-size`/`--except`, verify against LFS metadata, du with tree view, partial-download cleanup) was found in the survey. hf-fm's `cache` subcommand group appears unique on the Rust side; the only direct analog in the broader ecosystem is the Python `hf cache` group introduced upstream.

### Notable observations

- The `hf-hub` crate has ~70 reverse deps, dominated by libraries (`tokenizers`, `fastembed`, `lancedb`, `candle-*`) — most are library-on-library composition, not user-facing tools.
- Two of EricLBuehler's projects (`mistral.rs`, `safetensors_explorer`) cover adjacent territory but never merged the inspect tooling into the inference CLI.
- `rust-hf-downloader` is the closest direct competitor on download/search, but ships no inspect/diff/cache subcommands and is binary-only (no library reuse for downstream Rust apps like `candle-mi` / `anamnesis`).

---

## Comparison matrix

Cells: ✓ = present · ◐ = partial · — = absent.

Capability headers (the 12 hf-fm areas above, plus a forward-looking 13th surfaced by the survey):

| # | Header | Capability |
|---|---|---|
| 1 | MultiDL | multi-connection chunked download with resume + filters/presets |
| 2 | DLFile | single-file download with glob expansion |
| 3 | Inspect | safetensors header inspection (tree, dtypes, sharded aggregation, JSON) |
| 4 | Diff | tensor-layout diff between two models |
| 5 | Du | cache disk-usage drill-down + tree |
| 6 | CacheMgmt | cache delete / clean-partial / gc / verify (SHA256) / path |
| 7 | Search | Hub search with library/pipeline/tag filters |
| 8 | Info | model card / README display |
| 9 | Status | per-repo file-level cache state |
| 10 | ListFiles | remote file listing with sizes + SHA256 |
| 11 | Discover | `model_type` grouping / Hub family discovery |
| 12 | Lib | embeddable Rust library API |
| 13 | VRAMFit | "does this model fit on my GPU?" verdict from metadata only (no weight download) — planned for hf-fm v0.10.1 |

| Tool | MultiDL | DLFile | Inspect | Diff | Du | CacheMgmt | Search | Info | Status | ListFiles | Discover | Lib | VRAMFit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **hf-fetch-model v0.10.x** *(reference)* | ✓ | ✓ | ✓ tree, dtypes, sharded, JSON | ✓ | ✓ tree | ✓ delete/clean-partial/gc/verify/path | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — planned v0.10.1 |
| hf-hub 0.5.0 | ◐ resume, single-conn | ◐ no glob | — | — | — | — | — | ◐ metadata only | — | ◐ list-tree | — | ✓ | — |
| rust-hf-downloader 1.4.0 | ✓ adaptive chunks | ◐ no glob | — | — | — | — | ✓ TUI search | — | — | — | — | — bin only | — |
| rustyface 0.1.3 | ◐ concurrent, no chunks | — repo-only | — | — | — | — | — | — | — | — | — | — bin only | — |
| facecrab 0.1.6 | — single-conn | ✓ resolve-by-id | — | — | — | ◐ registry.toml | — | — | — | — | — | ✓ | — |
| ogenius 0.1.5 | — | ✓ via facecrab | — | — | — | — | — | — | — | — | — | ◐ via facecrab | — |
| hugging-face-client | ◐ basic | ◐ basic | — | — | — | — | — | — | — | ◐ | — | ✓ | — |
| safetensors_explorer 0.2.0 | — | — | ✓ TUI tree, sharded, fuzzy | — | — | — | — | — | — | — | — | — bin only | — |
| safetensors-cli 0.1.0 | — | — | ◐ minimal | — | — | — | — | — | — | — | — | — bin only | — |
| safetensors 0.7.0 | — | — | — no CLI | — | — | — | — | — | — | — | — | ✓ format only | — |
| gguf-rs 0.2.5 | — | — | ◐ GGUF only (info/tensors/meta/validate) | — | — | — | — | — | — | — | — | ✓ | — |
| inspector-gguf 0.3.1 | — | — | ◐ GGUF only, GUI+CLI | — | — | — | — | — | — | — | — | ✓ | — |
| candle 0.x | — via hf-hub | — via hf-hub | — | — | — | — | — | — | — | — | — | ✓ framework | — |
| mistral.rs 0.8.0 | ◐ via hf-hub | ◐ via hf-hub | — | — | — | ◐ doctor only | — | — | — | — | — | ✓ | — |
| apr-cli 0.31.2 | ◐ for inference | — | ◐ APR-format-centric | — | — | — | — | — | — | — | — | ✓ via aprender | ✓ `serve plan --gpu`, `config.json`-based |
| burn | — | — | — | — | — | — | — | — | — | — | — | ✓ framework | — |
| llmserve 0.0.7 | — | — | — | — | ◐ discovery | ◐ multi-cache discovery | — | — | ◐ status | — | ◐ across caches | — bin only | — |

---

## Gaps unique to hf-fm in the surveyed Rust ecosystem

- **Cache `gc` / `verify` / `clean-partial`** — no other Rust tool offers these subcommands.
- **Hub-side `inspect` over HTTP Range** (no full download) combined with sharded `model.safetensors.index.json` aggregation — `safetensors_explorer` reads local files only.
- **`diff` between two safetensors layouts** — no Rust analog found.
- **`du --tree` with box-drawing connectors over the HF cache layout** — only `llmserve` does coarser cache discovery; no drill-down by index.
- **Library + CLI parity** — `rust-hf-downloader` is bin-only, `safetensors_explorer` is bin-only, while hf-fm exposes the same orchestration to embedders (`candle-mi`, `anamnesis`).

The closest competitor footprint is "hf-hub (lib) + rust-hf-downloader (download CLI) + safetensors_explorer (inspect TUI)" — three separate tools whose union still does not cover diff, gc, verify, status, or remote sharded inspection.

### VRAMFit (forward-looking — v0.10.1)

`apr-cli` is the only existing Rust tool that ships a "does this model fit on my GPU?" verdict (`apr serve plan --gpu`). The proposed `hf-fm inspect --check-gpu` overlaps in intent but differs structurally:

| | apr-cli | hf-fm `--check-gpu` (planned) |
|---|---|---|
| Ecosystem | APR-format inference orchestration (80 workspace crates, 79 CLI commands, custom Q4K format) | thin flag inside the existing HF-focused `inspect` |
| Metadata source | `config.json` — architectural estimate | safetensors header — exact tensor footprint, accurate for quantized variants |
| Live VRAM source | undocumented in the README | `hypomnesis 0.1.0` (NVML / DXGI / `nvidia-smi`) |
| Composition | part of the inference pipeline (`apr serve run` follows) | composes with `--filter`, `--cached`, sharded aggregation |
| Primary intent | "should I serve this model?" | "should I download this model?" |

So the niche isn't *unfilled* — apr-cli got there first — but the use-cases barely intersect, and the mechanism differs (architectural estimate vs exact tensor footprint).

---

## Strategic read

The `hf-hub` reverse-dependency graph (~70 crates, mostly libraries — `tokenizers`, `fastembed`, `lancedb`, `candle-*`) confirms there's a healthy ecosystem of *library composition* over the official client, but **almost no user-facing tool development** above it. hf-fm appears to be the only project in that gap.

One interesting adjacency: **EricLBuehler** maintains both `mistral.rs` and `safetensors_explorer` but never merged the inspect tooling into the inference CLI. That's a structural choice — kept inspect as a standalone TUI rather than bolting it onto `mistralrs`. hf-fm goes the other way: one CLI, all capabilities.

---

## Sources

- [hf-hub GitHub](https://github.com/huggingface/hf-hub) / [crates.io](https://crates.io/crates/hf-hub) / [releases](https://github.com/huggingface/hf-hub/releases)
- [rust-hf-downloader on lib.rs](https://lib.rs/crates/rust-hf-downloader) / [docs.rs](https://docs.rs/crate/rust-hf-downloader/latest)
- [rustyface on lib.rs](https://lib.rs/crates/rustyface)
- [facecrab on lib.rs](https://lib.rs/crates/facecrab)
- [ogenius / rusty-genius](https://github.com/tmzt/rusty-genius) / [crates.io](https://crates.io/crates/ogenius/0.1.5)
- [safetensors_explorer GitHub](https://github.com/EricLBuehler/safetensors_explorer) / [lib.rs](https://lib.rs/crates/safetensors_explorer)
- [safetensors-cli GitHub](https://github.com/gzsombor/safetensors-cli) / [lib.rs](https://lib.rs/crates/safetensors-cli)
- [safetensors canonical](https://github.com/safetensors/safetensors)
- [gguf-rs GitHub](https://github.com/ThreatFlux/gguf) / [crates.io](https://crates.io/crates/gguf-rs)
- [inspector-gguf docs.rs](https://docs.rs/inspector-gguf)
- [candle GitHub](https://github.com/huggingface/candle)
- [mistral.rs GitHub](https://github.com/EricLBuehler/mistral.rs) / [docs](https://ericlbuehler.github.io/mistral.rs/INSTALLATION.html)
- [apr-cli on docs.rs](https://docs.rs/crate/apr-cli/latest) / [aprender monorepo](https://github.com/paiml/aprender)
- [vRAMIO on GitHub](https://github.com/ksingh-scogo/vramio) / [Medium intro](https://ksingh7.medium.com/free-tool-to-check-vram-requirements-for-any-huggingface-model-vramio-1eb55d55c7d7) (Python web service, included for VRAMFit landscape context)
- [burn GitHub](https://github.com/tracel-ai/burn)
- [llmserve on lib.rs](https://lib.rs/crates/llmserve)
- [hf-hub reverse deps](https://crates.io/crates/hf-hub/reverse_dependencies)
