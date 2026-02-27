# hf-fetch-model: Fast HuggingFace Model Downloads for Rust

> An embeddable Rust library for downloading HuggingFace models with maximum throughput

**Date:** February 27, 2026
**Status:** Phase 1 complete
**Context:** During the development of plip-rs and candle-mi, model downloads were a recurring bottleneck. No existing Rust crate provides a fast, ergonomic, embeddable library for downloading HuggingFace model repositories. hf-fetch-model fills this gap, and candle-mi will use it as its download backend.

---

## 1. The Gap

### 1.1 What Exists Today

| Crate / Tool | Type | Parallel Chunks | Download Repo | File Filtering | Progress | Embeddable |
|---|---|---|---|---|---|---|
| **hf-hub** v0.5.0 | Library | `.high()` mode | No (file-by-file) | No | No | Yes |
| **rust-hf-downloader** v1.4.0 | Binary (TUI/CLI) | Yes | Yes | Yes (TUI) | Yes (TUI) | **No** |
| **hf_transfer** | Python+Rust (PyO3) | Yes | No (single file) | No | No | **No** (Python only) |
| **xet-core / hf_xet** | Storage backend | Yes (dedup) | N/A | N/A | N/A | N/A |

**hf-hub** is the official HuggingFace Rust client. Its API is file-by-file only:

```rust
let repo = api.model("google/gemma-2-2b-it".to_string());
let path = repo.get("model-00001-of-00002.safetensors").unwrap();
```

The caller must know exact filenames, orchestrate multi-file downloads, and implement progress reporting. The `.high()` builder method enables parallel chunked downloads for individual files, but there is no repo-level orchestration.

**rust-hf-downloader** has the features we need (parallel downloads, progress, filtering) but is a binary-only application — its internals are private and it cannot be used as a library dependency.

**hf_transfer** is a Python package with a Rust core (PyO3). It accelerates single-file downloads but does not provide a Rust API.

**xet-core / hf_xet** is HuggingFace's next-generation storage backend with chunk-based deduplication. It operates at the infrastructure layer, not the user-facing download layer.

### 1.2 What Is Missing

No existing Rust crate provides all of the following as a **library API**:

1. **Repo-level download** — "give me this model, download all files"
2. **Maximum throughput by default** — parallel chunked downloads enabled out of the box
3. **Progress reporting** — per-file and overall, via callbacks for library consumers
4. **File filtering** — download only `*.safetensors`, or exclude `*.bin`, etc.
5. **Checksum verification** — validate downloaded files against HuggingFace metadata
6. **Embeddable** — usable as a Cargo dependency by crates like candle-mi

### 1.3 xet-core and the LFS-to-Xet Migration

[Xet](https://huggingface.co/docs/hub/en/xet/index) is HuggingFace's replacement for Git LFS as the Hub's storage backend. Unlike Git LFS (which stores entire file revisions), Xet uses [content-defined chunking](https://huggingface.co/docs/xet/en/deduplication) to deduplicate data at the chunk level — yielding ~50% storage savings and 2–3x faster transfers in [benchmarks](https://huggingface.co/blog/from-chunks-to-blocks).

**Migration timeline:**
- **Jan 2025:** Xet deployed, handling ~6% of Hub downloads
- **May 2025:** became the default for all new users and organizations
- **Jul 2025:** rolling migration of existing repos from LFS to Xet
- **No LFS deprecation date announced** — backward compatibility is maintained

[xet-core](https://github.com/huggingface/xet-core) is the Rust workspace implementing the chunking/deduplication client. It is consumed internally by `huggingface_hub` (Python) via `hf_xet` (PyO3 bindings). It is explicitly **not meant to be used directly** — it is infrastructure plumbing.

**Implications for hf-fetch-model:** The migration is transparent. The `hf-hub` Rust crate continues to work through HuggingFace's [Git LFS Bridge](https://huggingface.co/blog/migrating-the-hub-to-xet), which reconstructs files from Xet storage on-demand and returns presigned S3 URLs mimicking the LFS protocol. If/when `hf-hub` gains native Xet support, hf-fetch-model inherits it for free — this validates the "wrap `hf-hub`" strategy rather than threatening it.

### 1.4 Value Proposition

hf-fetch-model wraps hf-hub (the official client, with cache compatibility and auth support) and adds the repo-level orchestration layer that is missing. The hard problem — parallel chunked HTTP — is already solved by hf-hub's `.high()` mode. hf-fetch-model provides the ergonomic layer above it.

---

## 2. Architecture

```
┌─────────────────────────────────────┐
│           candle-mi                 │
│   download_model() convenience fn  │
└──────────────┬──────────────────────┘
               │ optional dep (feature = "fast-download")
┌──────────────▼──────────────────────┐
│        hf-fetch-model               │
│  • repo file listing               │
│  • file filtering (glob patterns)  │
│  • parallel file orchestration     │
│  • progress callbacks              │
│  • checksum verification           │
│  • resume / retry                  │
└──────────────┬──────────────────────┘
               │ dep
┌──────────────▼──────────────────────┐
│     hf-hub (tokio, .high())        │
│  • HTTP chunked parallel download  │
│  • HF cache layout compatibility   │
│  • auth token handling             │
└─────────────────────────────────────┘
```

### 2.1 Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Build from scratch vs. wrap hf-hub | **Wrap hf-hub** | Reuse official HTTP layer, cache layout, and auth; avoid reimplementing chunked HTTP |
| Async vs. sync API | **Async-first (tokio)**; optional sync wrapper (Phase 1) | `.high()` requires tokio; sync wrapper via `tokio::runtime::Runtime::block_on` for simple use cases |
| Separate crate vs. in candle-mi | **Separate crate** | Useful beyond MI; keeps candle-mi focused on interpretability |
| Progress reporting | **`ProgressEvent` struct + closure callback** on `FetchConfig`; optional `indicatif` feature | Library consumers choose their UI; CLI gets progress bars for free |
| Error handling | **thiserror** enum | Consistent with candle-mi conventions; no panics in library code |

### 2.2 Target API (Sketch)

```rust
use hf_fetch_model::{FetchConfig, ProgressEvent};

// Minimal usage — download a model with defaults (Phase 0)
// Grit Rule 6: async functions take owned types
let path = hf_fetch_model::download("google/gemma-2-2b-it".to_owned()).await?;

// Full control (Phase 1+)
let config = FetchConfig::builder()
    .revision("main")
    .filter("*.safetensors")
    .token_from_env()            // reads HF_TOKEN
    .concurrency(4)              // parallel file downloads
    .on_progress(|e: &ProgressEvent| {
        println!("{}: {:.1}%", e.filename, e.percent);
    })
    .build()?;

let path = hf_fetch_model::download_with_config("google/gemma-2-2b-it".to_owned(), &config).await?;
```

---

## 3. Coding Conventions (Grit)

hf-fetch-model follows [Grit — Strict Rust for AI-Assisted Development](https://github.com/PCfVW/Amphigraphic-Strict/tree/master/Grit), the same foundation used by candle-mi. Grit is a strict subset of Rust that makes implicit behaviors explicit, targeting patterns where AI-generated code commonly fails.

### 3.1 Grit Rules (all 10 apply)

| Rule | Name | Enforcement | Key constraint |
|------|------|-------------|----------------|
| 1 | Explicit lifetimes | `#![deny(elided_lifetimes_in_paths)]` | All public function signatures must have explicit lifetimes |
| 2 | Explicit conversions | `#![warn(clippy::as_conversions)]` | No implicit Deref coercion; use `.as_str()`, `.as_bytes()`, `.to_owned()` |
| 3 | No panic in libraries | `#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]` | Use `Result` + `?`; never `.unwrap()`, `.expect()`, or `panic!()` |
| 4 | No type erasure | Code review | No `Box<dyn Any>`; prefer generics. `// TRAIT_OBJECT:` annotation when dynamic dispatch is genuinely needed |
| 5 | Unsafe isolation | `#![forbid(unsafe_code)]` | No unsafe code. hf-fetch-model wraps safe APIs only |
| 6 | Owned types in async | Code review | Async functions take owned types (`String`, not `&str`); eliminates async lifetime complexity |
| 7 | Exhaustive matching | `#![deny(clippy::wildcard_enum_match_arm)]` | All enum variants matched explicitly; no `_` catch-all without documentation |
| 8 | Standard error pattern | `thiserror` | `FetchError` uses `thiserror`; `#[from]` for wrapped errors; `# Errors` doc section on all fallible functions |
| 9 | Prefer iterators | `#![warn(clippy::explicit_iter_loop, clippy::manual_filter_map)]` | Iterator chains over imperative loops for transformations |
| 10 | Single async runtime | Code review | Tokio only; no async-std or smol |

### 3.2 Additional Rules (from candle-mi extensions)

| Rule | Name | Enforcement | Key constraint |
|------|------|-------------|----------------|
| 11 | `#[non_exhaustive]` | Code review | Public enums that may gain variants (`FetchError`, `Filter`) must be `#[non_exhaustive]` |
| 17 | `#[must_use]` | `#![warn(clippy::must_use_candidate)]` | All public functions/methods returning a value with no side effects must be `#[must_use]` |

**Rules from candle-mi that do NOT apply** (tensor/MI-specific):
- Rule 12 (Shape documentation) — no tensors
- Rule 16 (Hook purity contract) — no hooks
- PROMOTE / CONTIGUOUS annotations — tensor-specific

### 3.3 Annotation Patterns

Every annotation is mandatory when the corresponding situation applies.

| Annotation | When required | Example |
|------------|---------------|---------|
| `// TRAIT_OBJECT: <reason>` | Every `Box<dyn Trait>` or `&dyn Trait` usage | `// TRAIT_OBJECT: heterogeneous progress handlers` |
| `// EXHAUSTIVE: <reason>` | On `#[allow(clippy::exhaustive_enums)]` | `// EXHAUSTIVE: internal dispatch enum; crate owns all variants` |
| `// EXPLICIT: <reason>` | Intentional no-op match arm, or imperative loop over iterator chain | `// EXPLICIT: no action needed for this variant` |
| `// BORROW: <what>` | Explicit `.as_str()`, `.as_bytes()`, `.to_owned()` conversions | `// BORROW: explicit .as_str() instead of Deref coercion` |
| `// SAFETY: <invariants>` | Every `unsafe` block (not expected: `#![forbid(unsafe_code)]`) | N/A |

### 3.4 Cargo.toml Lint Configuration

```toml
[lints.rust]
unsafe_code = "forbid"
elided_lifetimes_in_paths = "deny"

[lints.clippy]
# Rule 3: No panics
unwrap_used = "deny"
expect_used = "deny"
panic = "deny"
indexing_slicing = "deny"

# Rule 7: Exhaustive matching
wildcard_enum_match_arm = "deny"

# Rule 2: Explicit conversions
as_conversions = "warn"

# Rule 9: Prefer iterators
explicit_iter_loop = "warn"
manual_filter_map = "warn"
manual_find_map = "warn"
needless_range_loop = "warn"

# Rule 17: #[must_use]
must_use_candidate = "warn"

# Additional strictness
pedantic = { level = "warn", priority = -1 }
missing_errors_doc = "warn"
missing_panics_doc = "warn"

# Allow noisy pedantic lints
module_name_repetitions = "allow"
too_many_lines = "allow"
```

### 3.5 CI Enforcement

Every push and PR triggers CI checks that must all pass before merging. This is set up in Phase 0 alongside the first code.

**CI pipeline (`ci.yml`) runs:**

1. `cargo fmt --check` — formatting must match `rustfmt` defaults; no local style drift
2. `cargo clippy --all-targets -- -D warnings` — all Grit lints from §3.4 are enforced; warnings are errors
3. `cargo test` — all tests must pass

These three checks gate every merge from Phase 0 onward — not deferred to Phase 4.

### 3.6 SPDX Headers

Every `.rs` source file must begin with an SPDX license identifier:

```rust
// SPDX-License-Identifier: MIT OR Apache-2.0
```

---

## 4. Roadmap

### Phase 0 — Minimal Viable Download → `v0.1.0`

**Goal:** A working library crate that downloads an entire model repo faster than naive hf-hub usage.

**Deliverables:**
- [x] Initialize crate: `Cargo.toml` (with Grit lint configuration from §3.4), `src/lib.rs`, `#![forbid(unsafe_code)]`, SPDX headers
- [x] Set up CI: `ci.yml` enforcing `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test` on every push/PR (see §3.5)
- [x] Set up `publish.yml` with `workflow_dispatch` for manual re-runs from the GitHub Actions dashboard
- [x] Depend on `hf-hub` v0.5 with `tokio` feature, using `ApiBuilder::new().high()`
- [x] Implement repo file listing via HuggingFace API (`repo::list_repo_files()` using `info().siblings`)
- [x] Implement `download(repo_id: String) -> Result<PathBuf>`: download all files, return cache directory
- [x] Respect HF cache layout (`~/.cache/huggingface/hub/`) for compatibility with Python tooling
- [x] Auth: `HF_TOKEN` environment variable, delegated to hf-hub's `ApiBuilder`
- [x] Error type: `FetchError` enum with `thiserror` v2 (`#[non_exhaustive]`, variants: `Api`, `Io`, `RepoNotFound`, `Auth`)
- [x] Basic integration test: download `julien-c/dummy-unknown`, verify cache path exists

**Exit criteria:** `hf_fetch_model::download("some/small-model".to_owned()).await?` returns a valid cache path with all files present. ✅ Met.

### Phase 1 — Progress & Filtering → `v0.2.0`

**Goal:** Users can see download progress and select which files to download.

**Deliverables:**
- [x] `FetchConfig` builder with `revision`, `token`, `filter` (glob), `exclude` (glob), `concurrency`
- [x] `ProgressEvent` struct and `on_progress` closure callback on `FetchConfig`
- [x] `ProgressEvent` fields: `filename`, `bytes_downloaded`, `bytes_total`, `percent`, `files_remaining`
- [x] Optional `indicatif` feature gate: `IndicatifProgress` multi-progress bar (per-file + overall)
- [x] `download_with_config()` public API
- [x] Common filter presets: `Filter::safetensors()`, `Filter::gguf()`, `Filter::config_only()`
- [x] Optional sync wrapper: `download_blocking()` and `download_with_config_blocking()` for non-async callers

**Exit criteria:** Downloading a multi-shard model shows per-file progress bars and respects `.filter("*.safetensors")`. ✅ Met.

### Phase 2 — Reliability → `v0.3.0`

**Goal:** Downloads succeed on flaky connections and files are verified.

**Deliverables:**
- [ ] App-level resume on restart: skip already-completed files, resume partially downloaded files (hf-hub handles HTTP-level Range resume within a single session; this adds cross-session resume)
- [ ] Retry with exponential backoff + jitter (base 300ms, cap 10s, max 3 retries)
- [ ] SHA256 checksum verification against HuggingFace repo metadata
- [ ] Timeout configuration (per-file and overall)
- [ ] Structured error reporting: which files failed, why, whether retryable

**Exit criteria:** Killing the process mid-download and restarting completes without re-downloading finished files. Corrupted files are detected and re-fetched.

### Phase 3 — candle-mi Integration (no hf-fetch-model release)

**Goal:** candle-mi users can download models with a single function call.

**Deliverables:**
- [ ] Add `hf-fetch-model` as optional dependency of candle-mi: `features = ["fast-download"]`
- [ ] `candle_mi::download_model(repo_id)` convenience function
- [ ] Wire progress reporting to `tracing` (candle-mi's logging layer)
- [ ] Update candle-mi examples to use `download_model()` where appropriate
- [ ] Documentation: how to use fast downloads in candle-mi

**Exit criteria:** `candle_mi::download_model("google/gemma-2-2b-it".to_owned()).await?` downloads and returns the path, with progress visible via `tracing`.

### Phase 4 — CLI & Publish → `v0.4.0`

**Goal:** Standalone CLI tool and crates.io publication.

**Deliverables:**
- [ ] CLI binaries (thin wrapper): `hf-fetch-model google/gemma-2-2b-it --filter "*.safetensors"` (also available as `hf-fm`)
- [ ] `--revision`, `--token`, `--filter`, `--exclude`, `--output-dir`, `--concurrency` flags
- [ ] Benchmarks: compare download speed vs. plain hf-hub, document results in README
- [ ] Pre-publish audit: verify all Grit rules (§3) are satisfied, CI green, `cargo doc` clean
- [ ] README with usage examples, API docs, architecture diagram
- [ ] Publish to crates.io

**Exit criteria:** `cargo install hf-fetch-model && hf-fetch-model google/gemma-2-2b-it` works (and `hf-fm` as shorthand). Benchmarks show measurable speedup over default hf-hub on multi-shard models.

---

## 5. Project Structure

```
hf-fetch-model/
├── Cargo.toml                  # Single crate; features, Grit lints (§3.4)                [Phase 0] ✓
├── Cargo.lock                  # Pinned dependency versions                               [Phase 0] ✓
├── CONVENTIONS.md              # Grit rules and annotation patterns (extracted from §3)   [Phase 0] ✓
├── LICENSE-MIT                 # Dual license (MIT / Apache-2.0)                          [Phase 0] ✓
├── LICENSE-APACHE              #                                                          [Phase 0] ✓
├── README.md                   # Title, badges, one-liner                                 [Phase 0] ✓
├── CHANGELOG.md                # Keep a Changelog format                                  [Phase 0] ✓
├── hf-fetch-model-roadmap.md   # This file                                                [Phase 0] ✓
├── .github/
│   └── workflows/
│       ├── ci.yml              # cargo fmt, clippy, test on push/PR                       [Phase 0] ✓
│       └── publish.yml         # crates.io publish on tag + workflow_dispatch              [Phase 0] ✓
├── src/
│   ├── lib.rs                  # Public API: download(), download_with_config(), sync      [Phase 1] ✓
│   ├── error.rs                # FetchError enum (5 variants, thiserror)                  [Phase 1] ✓
│   ├── repo.rs                 # Repo file listing via HF API                             [Phase 0] ✓
│   ├── download.rs             # Orchestration: filtering, progress, hf-hub .high()       [Phase 1] ✓
│   ├── config.rs               # FetchConfig builder, Filter presets, glob matching       [Phase 1] ✓
│   ├── progress.rs             # ProgressEvent struct, indicatif implementation           [Phase 1] ✓
│   ├── checksum.rs             # SHA256 verification against HF metadata                  [Phase 2]
│   └── retry.rs                # Exponential backoff + jitter logic                       [Phase 2]
├── src/bin/
│   └── main.rs                 # CLI binary (clap-based)                                  [Phase 4]
│                               # Installed as both `hf-fetch-model` and `hf-fm`
├── tests/
│   ├── integration.rs          # Download julien-c/dummy-unknown, verify cache path       [Phase 0] ✓
│   └── filter.rs               # Glob filtering, progress callback, presets tests         [Phase 1] ✓
├── examples/
│   ├── basic.rs                # Minimal download example                                 [Phase 4]
│   └── progress.rs             # Download with indicatif progress bars                    [Phase 4]
└── benches/
    └── throughput.rs            # Benchmark vs. plain hf-hub sequential download           [Phase 4]
```

`cargo install hf-fetch-model` will install two binaries from the same source (added in Phase 4):

```toml
# Cargo.toml (Phase 4 additions)
[[bin]]
name = "hf-fetch-model"        # explicit, discoverable
path = "src/bin/main.rs"

[[bin]]
name = "hf-fm"                 # short alias for daily use
path = "src/bin/main.rs"
```

### Module Responsibilities

| Module | Phase | Status | Responsibility |
|---|---|---|---|
| `lib.rs` | 0–1 | ✓ Phase 1 | `download()`, `download_with_config()`, `download_blocking()`, `download_with_config_blocking()`; re-exports |
| `error.rs` | 0–2 | ✓ Phase 1 | `FetchError` enum (`Api`, `Io`, `RepoNotFound`, `Auth`, `InvalidPattern`); Phase 2: adds checksum, timeout variants |
| `repo.rs` | 0 | ✓ | `list_repo_files()` via `info().siblings`; parse metadata (sizes, SHAs) |
| `download.rs` | 0–1 | ✓ Phase 1 | `download_all_files()`: orchestrate file downloads with filtering and progress reporting |
| `config.rs` | 1 | ✓ | `FetchConfig` builder: revision, filters, token, `on_progress` callback, concurrency; `Filter` presets |
| `progress.rs` | 1 | ✓ | `ProgressEvent` struct; optional `IndicatifProgress` behind `indicatif` feature gate |
| `checksum.rs` | 2 | — | Stream SHA256 during download; verify against repo metadata |
| `retry.rs` | 2 | — | Exponential backoff + jitter; retry policy configuration |
| `bin/main.rs` | 4 | — | CLI: `hf-fetch-model <repo> [--filter] [--exclude] [--revision] [--token] [--output-dir] [--concurrency]` (also installed as `hf-fm`) |

---

## 6. Dependencies

| Crate | Version | Purpose | Phase | Status |
|---|---|---|---|---|
| `hf-hub` (tokio feature) | 0.5 | HTTP downloads, cache, auth | 0 | ✓ |
| `tokio` | 1 | Async runtime | 0 | ✓ |
| `thiserror` | 2 | Error types | 0 | ✓ |
| `globset` | 0.4 | File filtering | 1 | ✓ |
| `indicatif` (optional) | 0.17 | Progress bars | 1 | ✓ |
| `sha2` | — | Checksum verification | 2 | — |
| `tracing` | — | Structured logging (candle-mi integration) | 3 | — |
| `clap` | — | CLI argument parsing | 4 | — |

---

## 7. What hf-fetch-model Does NOT Do

- **Upload** — Out of scope. Download only.
- **Replace hf-hub** — hf-fetch-model depends on hf-hub; it is a layer above, not a replacement.
- **Compete with xet-core** — xet-core is a storage backend; hf-fetch-model is a user-facing download orchestrator. They operate at different layers (see §1.3).
- **Model inference** — No tensor loading, no forward pass. That is candle-mi's job.
- **Reimplementing chunked HTTP** — hf-hub's `.high()` already does this well.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| hf-hub API instability (maintainers warn of changes) | Pin to known-good version; isolate hf-hub behind internal adapter trait |
| HuggingFace moves entirely to xet backend | hf-fetch-model wraps hf-hub, which will adapt via the LFS Bridge (see §1.3); monitor xet-core for a Rust library API |
| `.high()` performance insufficient | Benchmark early (Phase 0); if needed, fall back to direct reqwest with Range headers |
| HF API rate limiting on file listing | Cache repo metadata; respect rate limit headers |
