# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **v0.10.3 Phase C `Format:` / `Size:` lines now fire on cached BnB / GPTQ / AWQ safetensors** — the `anamnesis` dependency in [Cargo.toml](Cargo.toml) was missing the `bnb`, `gptq`, and `awq` features, so anamnesis's tensor-name classifier (`classify_tensor` in `anamnesis::parse::safetensors`) routed `.qweight`, `.scales`, `.quant_map`, `.absmax`, `.SCB` etc. to `TensorRole::Passthrough`, `detect_scheme` returned `Unquantized`, and the Phase C renderer skipped the new lines. Surfaced empirically: `inspect medmekk/Llama-3.2-1B-Instruct-bnb-nf4-double-quant model.safetensors --cached` showed no Format/Size lines and `--json` returned `quant_info: None`, while the FP8 path (dtype-driven, no feature gate) worked correctly. Fix is purely additive on hf-fm's side — the three feature flags are pure code-gates in `anamnesis` (no extra transitive deps) — and unlocks the four schemes the changelog listed under v0.10.3 Phase C (`Bnb4`, `BnbInt8`, `Gptq`, `Awq`). Surfaced while validating v0.10.3 against the anamnesis Phase 5 Step 1c double-quant fixture set; the BnB DQ candidate-hunt workflow that motivated Phase C is now answerable in one inspect command (Format line + `nested_absmax` filter) instead of requiring tensor-table grepping.

## [0.10.3] — Cached-file format coverage

### Added

- **`hf-fm inspect <repo> file.safetensors --cached` now surfaces quantization detection** — Phase C of the [v0.10.3 plan](docs/roadmaps/cache-management-roadmap.md). Quantized safetensors files gain two new lines below the existing `Header:` line: `Format: <scheme>` (one of `FineGrainedFp8` / `PerChannelFp8` / `PerTensorFp8` / `Gptq` / `Awq` / `Bnb4` / `BnbInt8` per anamnesis's `QuantScheme`) and `Size: <stored> stored -> <dequantised> (BF16)` (the dequantised estimate is anamnesis's bf16-equivalent size: `byte_len × 4` for BnB-NF4/FP4 packed nibbles, `num_elements × 2` for FP8/GPTQ/AWQ/BnB-INT8 weight tensors, plus passthrough tensors copied as-is). Unquantized safetensors files render no new lines — the absence communicates "this is full precision". Pulled from anamnesis's `InspectInfo::from(&SafetensorsHeader)` (no extra I/O; the parse already happened on commit `26ef266`'s reader call). New library API: [`inspect::QuantInfo`](src/inspect.rs) (a hf-fm-owned struct holding `scheme: String, stored_bytes: u64, dequantized_bytes: u64`; the `scheme` field stores `QuantScheme`'s `Display` output rather than the enum directly so downstream library consumers aren't coupled to anamnesis's `#[non_exhaustive]` enum). Surfaced on the cached path only; the remote safetensors path's `quant_info` is `None` until v0.11.1 migrates that path to anamnesis. JSON output is additive — `quant_info` field gated by `#[serde(skip_serializing_if = "Option::is_none")]` so non-quantized + non-safetensors output stays byte-identical.
- **`hf-fm inspect <repo> file.pth --cached`** — Phase B commit 6 of the [v0.10.3 plan](docs/roadmaps/cache-management-roadmap.md). New `.pth` dispatch in the cached-inspect path via `anamnesis::parse_pth(path)` + the metadata-only `ParsedPth::tensor_info()` (new in anamnesis `0.5.0` — replaces the earlier `.tensors()` which would materialise each tensor's data via `Cow<'a, [u8]>`; the new method walks the parsed metadata only, no further I/O beyond the initial mmap). The new public helper [`inspect::inspect_pth_cached`](src/inspect.rs) mirrors `inspect_npz_cached`'s shape exactly. Same synthesised-offsets pattern as NPZ: anamnesis exposes per-tensor `byte_len`, hf-fm builds cumulative `(start, end)` pairs so `byte_len()` (= `end - start`) renders the correct size in the tensor table. `metadata: None`, `header_size: 0` (PTH has no metadata block analogous to safetensors's `__metadata__` or GGUF's KV table; the rare `big_endian` flag is not surfaced — can be added as a synthetic `pth.big_endian` key in a future patch if real users request it). Remote PTH inspect (no `--cached`) errors with the same "remote PTH inspect not yet supported (planned for v0.11): pass --cached after downloading..." wording. The `pth` feature is added to `anamnesis`'s feature list (was already enabled at registry-resolve time by commit a6ad095's bump; this commit toggles it on for hf-fm's build). New integration test `inspect_cached_pth_renders` in `tests/cli.rs` + new `find_cached_pth_repo()` helper.
- **`hf-fm inspect <repo> file.npz --cached`** — Phase B commit 5 of the [v0.10.3 plan](docs/roadmaps/cache-management-roadmap.md). New `.npz` dispatch in the cached-inspect path via `anamnesis::inspect_npz(path)`, which reads only the ZIP central directory + per-entry NPY headers (no tensor data is touched). The new public helper [`inspect::inspect_npz_cached`](src/inspect.rs) mirrors `inspect_gguf_cached`'s shape exactly: takes `(repo_id, filename, revision)`, returns the format-agnostic `SafetensorsHeaderInfo`. Per-tensor `(name, shape, dtype)` flow through directly; `data_offsets` is synthesised as cumulative `(start, end)` pairs from per-tensor `byte_len` (anamnesis doesn't surface real on-disk offsets for NPZ — they live inside ZIP entries — so the synthetic offsets preserve `byte_len()` correctness without claiming on-disk truth). `metadata: None` (NPZ has no metadata block analogous to safetensors' `__metadata__` or GGUF's KV table). Remote NPZ inspect (no `--cached`) errors with the same wording pattern v0.10.2 introduced for GGUF: `remote NPZ inspect not yet supported (planned for v0.11): pass --cached after downloading <file> with hf-fm download`. New integration test `inspect_cached_npz_renders` in `tests/cli.rs` + new `find_cached_npz_repo()` helper that mirrors `find_cached_gguf_repo` exactly so future NPZ tests share the same skip-pattern.
- **`hf-fm inspect <repo> file.gguf --cached --tree`** — extends `--tree` to GGUF. The tree pipeline (`build_tree` → `collapse_ranges` → `try_collapse_range` → `render_tree`) is format-agnostic by construction: numeric-sibling collapse parses segment names via `.parse::<usize>()` and verifies structural equality across siblings, so GGUF's `blk.<N>.<part>` naming collapses to `blk.[0..N]` exactly the way safetensors' `model.layers.<N>.<part>` collapses to `model.layers.[0..N]`. No production code change was needed; the new `inspect_cached_gguf_tree_renders` integration test and the new `find_cached_gguf_repo()` helper in `tests/cli.rs` confirm the end-to-end render works on any cached `.gguf`. Closes the v0.10.2 deferred-GGUF-rendering item (Phase A commit 2 of the [v0.10.3 plan](docs/roadmaps/cache-management-roadmap.md)).
- **`hf-fm inspect <repo> file.gguf --cached --dtypes`** — extends `--dtypes` to GGUF. The aggregator (`compute_dtype_groups`) already buckets by `t.dtype` string and sums per-tensor `t.byte_len()` directly — no `dtype_bytes()` lookup is involved — so GGUF quantization dtypes (`Q4_K_M`, `Q2_K`, `IQ4_NL`, `F32`, …) bucket transparently with the byte counts anamnesis populates at parse time. No production code change was needed; the new `inspect_cached_gguf_dtypes_renders` integration test in `tests/cli.rs` (reusing `find_cached_gguf_repo()` from commit 2) confirms the histogram renders end-to-end with the expected `Dtype` / `Tensors` / `Params` / `Size` columns and at least one GGUF-quantization row. Closes the v0.10.2 deferred-GGUF-rendering item (Phase A commit 3 of the [v0.10.3 plan](docs/roadmaps/cache-management-roadmap.md)).
- **`hf-fm list-families --show quant`** — opt-in column read from each cached repo's `config.json` `quantization_config.quant_method` field (the transformers-standard location). Falls back to `gguf` when any cached file in the newest snapshot directory has a `.gguf` extension. Closes [Gap 3 of the anamnesis Phase 5 dogfooding report](docs/dogfooding-feedbacks/hf-fm-dogfooding-anamnesis-phase5.md) — the "what BnB models do I have cached" question no longer requires scanning repo names. New library API: [`cache::FamilyEntry`](src/cache.rs); [`cache::list_cached_families`](src/cache.rs) return type evolves from `BTreeMap<String, Vec<String>>` to `BTreeMap<String, Vec<FamilyEntry>>` (the only existing caller is `run_discover`, which uses `.into_keys()` and is unaffected).
- **`hf-fm list-families --tag <TAG>`** — filter cached repos by a HuggingFace tag (case-insensitive). Tags are fetched at query time via [`discover::fetch_model_card`](src/discover.rs), fanned out through a bounded `tokio::sync::Semaphore` (8 permits) — same pattern as `search --show size` from [`27199bf`](https://github.com/PCfVW/hf-fetch-model/commit/27199bf). Per-repo fetch failures silently drop the row from the filter result (the listing is not aborted). Empty families are pruned. Closes [Gap 8](docs/dogfooding-feedbacks/hf-fm-dogfooding-anamnesis-phase5.md). New library API: [`discover::fetch_tags_concurrent`](src/discover.rs). No persistence change — `Snapshot` schema stays at v1 (tags are not cached locally; the user's design preference was query-time fetch over schema bump).
- **`hf-fm cache path <REPO_ID> --revision <REV>`** — the long-promised revision flag (the help text at [src/bin/main.rs:534-544](src/bin/main.rs#L534-L544) previously carried a "planned for a future `--revision` flag" note). When set, `cache path` resolves the snapshot directory for the named ref instead of `main`. The error wording also adapts: "repo X is cached but has no ref for `v2`" instead of `"main"`. Closes [Gap 9](docs/dogfooding-feedbacks/hf-fm-dogfooding-anamnesis-phase5.md) — the last "acknowledged but unimplemented" item from the report.
- **`hf-fm search --show <COLUMNS>`** — opt-in result enrichment. `--show tags` appends the full tag list inline (free; already in the HF API payload — no extra round-trip). `--show size` adds a total-repo-size column (formatted via the existing [`format::format_size`](src/format.rs#L25-L54) helper) populated by one extra `/api/models/{repo_id}?blobs=true` request per result, fanned out through a bounded `tokio::sync::Semaphore` (8 permits) to stay friendly to the HF Hub on `--limit 100`-style invocations. Per-repo size lookups that fail render as `—`; the search itself is **not** aborted on a single failure. Closes [Gap 1 of the anamnesis Phase 5 dogfooding report](docs/dogfooding-feedbacks/hf-fm-dogfooding-anamnesis-phase5.md) — the "filter 13 search results to actually-`bitsandbytes`-tagged + under 1 GiB" workflow that previously took 9–11 commands now takes one: `hf-fm search fp4 --tag bitsandbytes --show tags,size`. New library API: [`hf_fetch_model::discover::fetch_repo_total_size`](src/discover.rs) and [`hf_fetch_model::discover::fetch_repo_sizes_concurrent`](src/discover.rs).
- **`hf-fm status --preset <P>` + `.hf-fm-snapshot.json` sidecar** — closes [Gap 5 of the anamnesis Phase 5 dogfooding report](docs/dogfooding-feedbacks/hf-fm-dogfooding-anamnesis-phase5.md). `download` now writes a tiny JSON sidecar at the repository's cache root (`{cache_root}/models--{org}--{name}/.hf-fm-snapshot.json`) recording the `--preset` used at download time (plus `--filter` and `--exclude` glob lists, reserved for a future patch — the schema is `version: 1` so consumers can negotiate). `status` reads the sidecar and reports files that don't match the recorded preset's glob list as `excluded` instead of `MISSING`, distinguishing intentional skips from incomplete downloads. `status --preset <P>` overrides the sidecar and **works retroactively** on caches that pre-date this feature (no sidecar required). New `FileStatus::Excluded { expected_size }` variant on the already-`#[non_exhaustive]` enum. New library API: [`hf_fetch_model::cache::Snapshot`](src/cache.rs), [`read_snapshot`](src/cache.rs), [`write_snapshot`](src/cache.rs), [`SNAPSHOT_FILENAME`](src/cache.rs), [`SNAPSHOT_VERSION`](src/cache.rs). Single source of truth for preset → glob mapping: new [`hf_fetch_model::config::preset_globs`](src/config.rs) consumed by both the `download` path (via [`Filter::safetensors`](src/config.rs) et al., refactored to share the same constants) and the `status` path. Sidecar write is best-effort — failures emit a `warning:` line on stderr but do not fail the download.
- **`hf-fm discover --tag <TAG>`** — filters the discovery results to model families whose representative top-by-downloads model carries the given HuggingFace tag (e.g. `bitsandbytes`, `gguf`, `conversational`). Mirrors the existing `hf-fm search --tag` flag and composes with `--limit`. Closes [Gap 7 of the anamnesis Phase 5 dogfooding report](docs/dogfooding-feedbacks/hf-fm-dogfooding-anamnesis-phase5.md): the dogfooder's "find me a cross-architecture BnB-quantized model not in my cache" question was previously impossible to express in one command because `discover` returned only top-by-downloads families with no tag axis (so `bert`, `clip`, `vit`, `whisper` filled the top of the list even when looking for LLM quantization variants). Implementation in [src/discover.rs](src/discover.rs): `discover_new_families` gains an `Option<&str>` tag parameter, the inline `format!()` URL build is refactored to `reqwest::Client::get(HF_API_BASE).query(&query_params)` for symmetry with `search_models`, and a client-side `tags.iter().any(eq_ignore_ascii_case)` check verifies the tag is present on each returned model (defensive — the HF API may ignore the `filter` parameter when combined with other params). `discover` also gains a worked-examples `after_help` block (it had none before — only `search` did). Output header switches between `"New families with tag {t:?} not in local cache (top models by downloads):"` and the unfiltered wording based on whether `--tag` is set.

### Changed

- **`inspect::SafetensorsHeaderInfo` is now `#[non_exhaustive]` with a `quant_info` field + a public `new()` constructor** — Phase C side-effect. The new `quant_info: Option<QuantInfo>` field is technically a breaking change for external code that constructs `SafetensorsHeaderInfo` via struct-literal syntax. Marking the struct `#[non_exhaustive]` signals to downstream consumers (`candle-mi`, `anamnesis`) that the type may grow over time and forces them through a constructor path (or pattern-match with `..`). The codebase already uses this convention on `FileStatus` and `GateStatus`; `SafetensorsHeaderInfo` has been growing through v0.10.x (the field shape went from 4 to 5 in this commit alone) and will likely grow further in v0.11.x (when the format-agnostic rename lands). A new `pub fn new(tensors, metadata, header_size, file_size, quant_info) -> Self` constructor is the canonical way to build instances from outside the lib crate — the `hf-fm` binary crate's filter-render path is updated to use it. Existing field reads via `.tensors` / `.metadata` / `.header_size` / `.file_size` are unaffected — `#[non_exhaustive]` only restricts construction and exhaustive pattern matching.
- **`FetchError::UnsupportedInspectFormat` Display message now names all four supported formats** — Phase B commit 7 of the [v0.10.3 plan](docs/roadmaps/cache-management-roadmap.md). Wording lifts from `"hf-fm inspect supports .safetensors or .gguf (got .{extension} for {filename})"` to `"hf-fm inspect supports .safetensors, .gguf, .npz, or .pth (got .{extension} for {filename})"`. The cached-only constraint on `.gguf` / `.npz` / `.pth` lives in a separate dispatch-time error (`"remote {FORMAT} inspect not yet supported (planned for v0.11): pass --cached after downloading..."`), so the unsupported-extension message stays focused on the extension dimension. New unit test `unsupported_format_error_lists_all_four_formats` in `src/error.rs`'s `tests` module asserts every format appears in the Display output plus the unrecognised extension and filename are still surfaced.
- **`anamnesis` runtime dependency bumped `0.4.5` → `0.5.0`** — kicks off v0.10.3 Phase B. The two API entry points used by hf-fm so far are unchanged across the bump (`anamnesis::parse_gguf(path)` and `anamnesis::parse_safetensors_header_from_reader<R: Read>`), so the previous commit `26ef266` continues to compile cleanly. The bump also enables the `npz` feature in preparation for the next commit (`.npz` cached inspect). New 0.5.0 surface area used downstream this slice: `anamnesis::inspect_npz` (commit B; NPZ metadata) and `ParsedPth::tensor_info() -> Vec<PthTensorInfo>` (commit C; PTH metadata-only enumeration, *new* in 0.5.0 — the previously-only `.tensors()` materialised data via `Cow`, the new method gives a clean per-tensor `(name, shape, dtype, byte_len)` view with no I/O beyond the initial parse). Five new transitive deps land via the `npz` feature: `arbitrary`, `crossbeam-utils`, `derive_arbitrary`, `zip 2.4`, `zopfli`.
- **`hf-fm inspect --cached` on `.safetensors` files now uses `anamnesis::parse_safetensors_header_from_reader`** — Phase B commit 4 of the [v0.10.3 plan](docs/roadmaps/cache-management-roadmap.md). The cache-hit safetensors path replaces hf-fm's in-tree `serde_json` header parser with anamnesis's reader-based variant: hf-fm hands the open `std::fs::File` to anamnesis, which reads the 8-byte u64 length prefix + the JSON header bytes itself (bypassing `safetensors::SafeTensors::read_metadata`'s data-section requirement, capping the declared header length at 100 MiB internally). Same in/out behaviour — same tensors, same `__metadata__`, same `header_size`/`file_size` fields, and an explicit `sort_by_key(|t| t.data_offsets.0)` after the call preserves hf-fm's v0.10.2 file-order sort (anamnesis returns tensors sorted alphabetically by name; the inspect table has always been file-ordered so users can spot first/last tensors per shard at a glance). The remote path (`inspect_safetensors` → `fetch_header_bytes` → `parse_header_json`) stays bespoke until v0.11.1. Net diff in [src/inspect.rs:278-340](src/inspect.rs#L278-L340): the manual 8-byte-prefix + bounds-check + buffer-allocate logic disappears (anamnesis handles all three); `parse_header_json` stays as the remote path's parser. No public API change.
- **`hf-fm inspect <repo> file.gguf --cached` summary now uses GGUF-flavoured wording** — three cosmetic fixes against the safetensors-flavoured renderer that v0.10.2 commit 4b documented as "known cosmetic issues": (a) the misleading `Header: 0 B (JSON), <total> total` line is replaced for GGUF with `Size: <total>` — semantically accurate (GGUF has no length-prefixed JSON header, so the v0.10.2 wording printed a meaningless `0 B (JSON)` for every GGUF invocation), and `Size:` rather than `File:` avoids a label collision with the preceding `File: <filename>` line; (b) the one-line comma-separated `Metadata:` rendering switches to a tabular block (one `key=value` per indented line, keys sorted alphabetically so prefix groups — `general.*`, `<arch>.*`, `tokenizer.*`, `gguf.*` — cluster naturally) when the metadata has more than 6 keys; (c) multi-line values like `tokenizer.chat_template`'s Jinja templates get their own indented continuation lines under a `key=` header instead of wrapping awkwardly inline. Safetensors output is unchanged below the 6-key threshold and the `Header:` line keeps its `(JSON)` suffix. The new private renderers `format_header_line` and `format_metadata_lines` carry 9 new unit tests; the existing `inspect_cached_metadata_present` integration test is loosened to accept both inline and tabular renders. Closes the v0.10.2 deferred-GGUF-rendering item (Phase A commit 1 of the [v0.10.3 plan](docs/roadmaps/cache-management-roadmap.md)).
- **`hf-fm info` flags a HuggingFace default-template README** — adds a one-line `Note:` warning above the README dump when the body looks like an unmodified HF template. Heuristic: literal `# Model Card for Model ID` line in the first 20 body lines, or more than 30% of those lines being HTML comments (`<!--` after whitespace trim). Pure integer arithmetic, no float casts. Closes [Gap 4](docs/dogfooding-feedbacks/hf-fm-dogfooding-anamnesis-phase5.md) — the default 40-line dump on templated cards no longer reads as "this model has no documentation" vs "this user kept the boilerplate". Four new unit tests in `tests::looks_like_default_template_*` cover the positive case (verbatim HF default), the negative case (a content-rich README excerpt), the blank-input edge case, and the high-comment-density branch.
- **`hf-fm inspect --help` now cross-links `hf-fm diff`** — addresses the "diff was an afterthought" observation from the dogfooding report's "Things I almost missed" section. The mutual cross-link in the other direction is already in place via `diff`'s help. The new line appears at the very end of `inspect`'s `after_help`: `See also: hf-fm diff <A> <B>    # compare two repos' tensor layouts`.
- **`hf-fm search <repo> --exact` now prints `See also: hf-fm info <repo>`** — addresses the "search --exact and info both surface the same Tags: row" observation from the same section. Users who reach `search --exact` and want the longer-form view (license, full README) now find it explicitly. Runtime println (not `after_help`) — it has access to the resolved `model_id` and therefore points at the concrete repo, not just the subcommand.
- **`hf-fm inspect` count line is now a labelled two-line form** — closes [Gap 6 of the anamnesis Phase 5 dogfooding report](docs/dogfooding-feedbacks/hf-fm-dogfooding-anamnesis-phase5.md). The previous `6/112/898 tensors shown, 147.5K/45.4M/382.5M params (filter: "gate_proj", limit: 6)` triple-number form forced the reader to recover the `shown / matching-filter / total` mapping from the trailing qualifier. The three multi-number branches at [src/bin/main.rs:4986-5020](src/bin/main.rs#L4986-L5020) now produce explicit labels: `Showing 6 of 112 tensors matching filter "gate_proj" (898 tensors total, limit: 6).` on one line, followed by `Param counts: 147.5K shown, 45.4M matching filter, 382.5M total.` on the next. Same information; zero new code paths; the `(false, false)` no-filter no-truncation branch (`5 tensors, 147.5K params`) is unchanged since it has nothing to clarify. `--json` mode is unaffected (the count line is human-readable-only and never appears in JSON output). README, the [inspect-before-downloading tutorial](docs/tutorials/inspect-before-downloading.md), and the `inspect_cached_filter` integration test are updated to match. The archived candle issue snapshot at [docs/issues/candle-3448-p1.md](docs/issues/candle-3448-p1.md) is deliberately left with the old wording — it is a historical record of what was posted upstream and should not be retroactively rewritten.
- **`hf-fm search --help` and `hf-fm discover --help` now include worked examples** — closes [Gap 2 of the anamnesis Phase 5 dogfooding report](docs/dogfooding-feedbacks/hf-fm-dogfooding-anamnesis-phase5.md). The `--tag` flag has existed on `search` since the introduction of the discovery API but the help text only described what it does, not that it *composes* with the positional query. The dogfooder guessed correctly without testing, but the gap is real: `search "fp4" --tag bitsandbytes` is the textbook first filter to reach for and was invisible. The new `Examples:` block on `Search`'s `after_help` ([src/bin/main.rs](src/bin/main.rs)) demonstrates the composition explicitly. A parallel `Examples:` block lands on `Discover`'s `after_help` (showing `--tag` composing with `--limit`). [docs/cli-reference.md](docs/cli-reference.md) gains a paired tag-composition example in the search examples block (next to the existing `--tag gguf` example) and a `discover --tag bitsandbytes` example in the discover block.
- **`docs/cli-reference.md` updated for v0.10.2's GGUF inspect dispatch** — three corrective edits caught after the release. (a) Subcommands table's `inspect` row now reads "Inspect `.safetensors` file headers (remote or cached) or `.gguf` files (cached only, since v0.10.2)" instead of safetensors-only. (b) `## Inspect examples` section gains a one-sentence preamble flagging multi-format coverage. (c) A `hf-fm inspect <repo> <file>.gguf --cached` example added to the Inspect-examples block (using `bartowski/Mistral-7B-Instruct-v0.3-GGUF` as the worked example — same repo the v0.10.2 smoke test used). The corresponding README and FAQ entries were already correct as of [`a7e635e`](https://github.com/PCfVW/hf-fetch-model/commit/a7e635e); the cli-reference omissions slipped through the doc-rollout commit and only surfaced in the post-v0.10.2 audit.

## [0.10.2] — Ecosystem-adoption release

### Added

- **`hf-fm inspect <repo> file.gguf --cached`** — new dispatch path for inspecting GGUF tensor files from the local HuggingFace cache, delegating to [`anamnesis::parse_gguf`](https://docs.rs/anamnesis/) for the on-disk metadata parse. Tensor names, GGUF-native shape order, dtype names (e.g. `Q2_K`, `IQ1_S`, `IQ2_XXS`, `Q4_K`, `F32`), and per-tensor byte counts surface through the existing render path. GGUF metadata is surfaced through the `Metadata:` block: scalar values (strings, booleans, integers, floats) pass through verbatim, with synthetic keys `gguf.version` and `gguf.alignment` appended so the format-level info is visible alongside `general.architecture`, `general.name`, `llama.context_length`, etc.; large array values (e.g. `tokenizer.ggml.tokens` with 50K+ entries) are intentionally skipped to keep the line legible. Remote GGUF inspect (no `--cached`) returns a clear error pointing at v0.11 with a recovery hint (`Pass --cached after downloading the file with hf-fm download`). New library API: [`hf_fetch_model::inspect::inspect_gguf_cached`](src/inspect.rs). The previous `FetchError::UnsupportedInspectFormat` Display message updated from "supports .safetensors only" to "supports .safetensors or .gguf". The unused `has_safetensors_extension` helper in `src/bin/main.rs` is removed; the dispatch now classifies the extension inline (`.safetensors` / `.gguf` / unsupported). **Known cosmetic issues**, all v0.10.3 cleanup territory (none affect correctness): (a) the `Header:` line still reads `0 B (JSON), <total> total` for GGUF — the "(JSON)" suffix is safetensors-flavoured; (b) the one-line comma-separated `Metadata:` rendering is unwieldy when GGUF surfaces 30+ keys; (c) `tokenizer.chat_template` values are multi-line Jinja templates that don't render cleanly inline. `--tree` and `--dtypes` for GGUF are out of scope for this commit and arrive in commit 4c of v0.10.2.
- **`hf-fm diff <REPO_A> <REPO_B> --dtypes`** — new flag renders side-by-side per-dtype histograms instead of the per-tensor body. Six columns (Dtype, A Tensors, A Size, B Tensors, B Size, Δ Size) with dynamic widths (each column floors at its header width and grows to fit the widest data cell, mirroring `inspect --dtypes`'s convention). Em-dashes for dtypes only present in one side. Rows sorted by `max(A bytes, B bytes)` descending so the largest contributors surface first. Footer shows totals per side plus signed Δ tensors and Δ size. Conflicts at parse time with `--summary` (incoherent intents). Composes with `--filter` (histograms aggregate over filtered tensors only). Composes with `--json`: `DiffResult` gains a `dtype_histograms: { a, b }` field alongside the existing per-tensor diff (additive — JSON consumers that don't pass `--dtypes` see no change). Design target of 75 chars on a default terminal is documented in the new `DIFF_DTYPES_DESIGN_WIDTH` constant and guarded by `diff_dtypes_canonical_case_fits_design_width` so future column-format changes that blow past the budget on a 3-dtype scaled-sibling shape (BF16 / U8 / F32, GiB-range sizes — the canonical [candle #3530](docs/issues/candle-3530-p1.md) case) are caught at test time. **Direct enabler for the candle #3530 p2 reply** — once the OP shares the crashing 80B repo and the working 35B sibling, this flag lands the side-by-side dtype evidence in a single screenshot.
- **`byte_count` field on `DiffTensorSide` (JSON-only)** — every per-tensor entry in `diff --json`'s `only_a` / `only_b` / `differ` arrays now carries a `byte_count` field alongside `dtype` and `shape`. Populated from `TensorInfo::byte_len()`. Enables downstream `jq` recipes to sum bytes per name-pattern bucket — the JSON-first hook for the layer-index collapse analysis planned in commit 3 of v0.10.2 (FAQ entry). Additive to the schema; existing consumers see new fields, nothing breaks.
- **FAQ entry: *"How do I compare two HuggingFace models structurally?"*** — new entry under [Discovery](docs/FAQ.md#discovery--finding-what-to-inspect-or-download) walks through `diff` → `diff --dtypes` → `diff --json | jq`, with a worked example on the public scaled-sibling pair `openai/gpt-oss-20b` vs `openai/gpt-oss-120b` (the validation pair from v0.10.2 development) and a `jq` recipe that abstracts numeric segments in tensor names (`gsub("[0-9]+"; "{N}")`) and groups by the resulting pattern, summing the new `byte_count` field per group. The JSON-first approach lets users iterate on the collapse heuristic against their own pair before any of it becomes a built-in flag (a `--collapse` flag is deferred to v0.11, gated on real-world cases per the [cache-management roadmap](docs/roadmaps/cache-management-roadmap.md)). README's [Inspect & compare](README.md#inspect--compare) section gains a matching `diff --dtypes` example and a cross-link to the FAQ entry; [docs/cli-reference.md](docs/cli-reference.md) gets the new flag in the Diff flags table and a new example in the Diff examples block.

### Changed

- **New runtime dependency: `anamnesis = "0.4.5"` with the `gguf` feature enabled** — first adoption of [anamnesis](https://crates.io/crates/anamnesis), the framework-agnostic tensor-format-parsing crate. This is the foundation for the GGUF inspect cached dispatch path landing in subsequent v0.10.2 commits — `inspect <repo> file.gguf --cached` (delegates to `anamnesis::parse_gguf(path).inspect()`) plus `--tree` and `--dtypes` rendering for GGUF tensors. No user-visible behavior change from this commit alone; the dep is wired into `Cargo.toml` and `Cargo.lock` only. Adds five transitive deps to the build graph: `crunchy`, `float8`, `half`, `memmap2`, `safetensors` (the latter is anamnesis's parser source; hf-fm's in-tree safetensors header parser stays in place for now and is scheduled for retirement in v0.10.3 once the dispatch coverage is uniform). The roadmap line previously named `anamnesis = "0.4.3"` is bumped to `0.4.5` in the same commit — two patch releases happened upstream between the plan being written and the dep landing, both backward-compatible.
- **`hypomnesis` bumped 0.2 → 0.2.1; adopted `name_or_unknown()` and the `test-helpers` dev-feature builder** — closes the [hypomnesis-adoption.md](docs/dogfooding-feedbacks/hypomnesis-adoption.md) dogfooding loop. The inline `dev.name.as_deref().unwrap_or("unknown GPU")` at [src/gpu_check.rs:200](src/gpu_check.rs#L200) collapses to a single `dev.name_or_unknown()` call (exact same string surfaced, no behavior change for users). `hypomnesis = { version = "0.2.1", features = ["test-helpers"] }` is added under `[dev-dependencies]` so `GpuDeviceInfo::builder()` is available to tests without leaking into the runtime binary; the deferred-tests comment block at [src/gpu_check.rs:475-482](src/gpu_check.rs#L475-L482) is replaced by two real unit tests (`gpu_check_json_fit_path`, `gpu_check_json_miss_path`) exercising the `fits: true / headroom_bytes` and `fits: false / short_bytes` branches of `gpu_check_json` against synthetic `GpuDeviceInfo` fixtures. Pulled forward from the v0.10.4 line of the [cache-management roadmap](docs/roadmaps/cache-management-roadmap.md) into v0.10.2 so the dogfooding loop closes in the same patch that opens the anamnesis adoption loop.
- **README and FAQ install pages now document the `--force` requirement when upgrading** — without it, `cargo install hf-fetch-model --features cli` short-circuits whenever any prior version of the binary is in `~/.cargo/bin/` (cargo exits `0` with a low-priority "already installed" notice on stderr, leaving the binary on `PATH` unchanged). The README's [Install](README.md#install) section gains an [Upgrading from a previous version](README.md#upgrading-from-a-previous-version) subsection covering `cargo install … --force` plus a `hf-fm --version` verification step; the FAQ's [Installation and authentication](docs/FAQ.md#installation-and-authentication) section gains a matching Q — [*How do I upgrade hf-fm? Why does `cargo install` silently keep the old version?*](docs/FAQ.md#how-do-i-upgrade-hf-fm-why-does-cargo-install-silently-keep-the-old-version). Companion note at [`docs/dogfooding-feedbacks/cargo-install-silent-skip.md`](docs/dogfooding-feedbacks/cargo-install-silent-skip.md) captures the failure mode in full (reproduction, mis-diagnosis cost, and proposed `hf-fm --check-update` flag + `cargo-binstall` support as future patch-release candidates).

## [0.10.1] — inspect --check-gpu (hypomnesis adoption)

### Added

- **`hf-fm inspect <repo> [FILE] --check-gpu [N]`** — new flag adds a one-line GPU-fit verdict to the `inspect` output: model weight size compared against free VRAM on device `N` (default 0). Works on both the per-file path (`inspect <repo> file.safetensors --check-gpu`) and the whole-repo path (`inspect <repo> --check-gpu`, forces shard aggregation so the verdict reflects the total weight bytes across every shard). The verdict block lists the device name, total / free / used VRAM, and a ✓ / ✗ result with the headroom or shortfall — followed by a one-paragraph reminder that the figure is weights-only and large-context inference typically needs ~1.3–1.5× more for KV cache + activations. `--filter` and `--limit` continue to scope only the printed tensor table; the verdict always uses the unfiltered model totals. Composes with `--json`: a `gpu_check` object is added to the existing schema for the per-file path, the `--tree --json` schema, and the `--dtypes --json` schema; the repo-level plain `--json` schema switches from a `Vec<(filename, header)>` array to `{"files": [...], "gpu_check": {...}}` when `--check-gpu` is passed (the array schema is preserved when `--check-gpu` is absent). On systems with no NVIDIA GPU detected, an out-of-range device index, or where neither NVML nor DXGI is usable, the verdict line reports the failure verbatim and the command still exits 0 — `--check-gpu` is informational, never a gate. Backed by a new `hypomnesis = "0.2"` dependency (the framework-agnostic VRAM-measurement crate); hf-fm v0.10.1 is hypomnesis's *first external consumer*, validating the public `device_info` surface ahead of the candle-mi migration. Closes the v0.10.1 line of the [cache-management roadmap](docs/roadmaps/cache-management-roadmap.md).

- **`docs/dogfooding-feedbacks/hypomnesis-adoption.md`** — companion feedback note capturing concrete observations from the v0.10.1 adoption: testability gap of the `#[non_exhaustive] GpuDeviceInfo` (external crates cannot construct a test fixture, blocking unit tests of the fit / miss render paths), the missing `format_total` / `format_used` parity with the `report` feature's `format_free`, the suggestion of a `GpuDeviceInfo::name_or_unknown()` convenience, and notes on which error variants actually surface for `device_info` callers. Establishes the pattern: every external-crate adoption gets its own feedback doc so the upstream maintainer learns from real downstream wear-and-tear.

- **v0.10.4 line on the [cache-management roadmap](docs/roadmaps/cache-management-roadmap.md)** — captures the two follow-ups already identified for `--check-gpu`: `--check-gpu all` (multi-GPU verdict, iterates over `hypomnesis::device_count`) and `--check-gpu --context N` (KV-cache budgeting from the model's architectural config — replaces the v0.10.1 weights-only disclaimer with a real fit verdict against weights + KV + activations).

### Changed

- **`format_size` extracted to a dedicated [`src/format.rs`](src/format.rs) module** — was a private helper in `src/bin/main.rs` since v0.9.4; v0.10.1 promotes it to a binary-internal module via `#[path = "../format.rs"] mod format;` because the new `src/gpu_check.rs` verdict renderer needs the same formatter. No public API change (the module lives outside `src/lib.rs`'s module tree); call sites in `main.rs` are unchanged thanks to a top-level `use format::format_size;`. Five new unit tests cover the four bucket transitions (B / KiB / MiB / GiB / TiB).

## [0.10.0] — Cache maturity & first docs

### Fixed

- **`hf-fm inspect <repo> --dtypes` / `--tree` / `--limit` now work on sharded repos** — previously, when the user asked for tensor-level aggregation on a multi-shard repo (one with `model.safetensors.index.json`), the shard-index fast path silently swallowed the flag and printed only the per-shard tensor counts (e.g. `4 shards, 291 tensors`) — same output as the bare `inspect <repo>`. The `dtypes`, `limit`, and `tree` flags weren't even plumbed through to `run_inspect_repo`'s signature, so they had no chance of taking effect. Now: when any of the three flags is set, the fast path is bypassed; every shard's header is read (the existing `inspect_repo_safetensors`/`_cached` path, ~tens of KiB per shard via HTTP Range), tensors are flattened across shards, then the same renderers used by single-file `inspect` (`print_dtype_summary`, `print_tree_summary`, plus their `--json` variants) take over. Tree mode shows one consolidated tree with `[0..N]` collapsing across the whole model — confirms architecture, layer count, and tied-vs-untied `lm_head` at a glance. `--limit` on multi-shard adds a `Shard` column so the user can see which file each tensor came from. `--filter` continues to work as before, including its existing fast-path-internal filtering when no aggregation flag is set. Discovered while inspecting `zed-industries/zeta-2` (4 shards, 291 tensors, 8.25B BF16 params).

- **`hf-fm <repo>` no longer reports `Cached at:` when filtered files are missing on disk** — `download_all_files_map` previously called `try_resolve_repo_from_cache` *before* the network listing, which scanned the snapshot directory and applied the user's include/exclude filter to whatever happened to be on disk. A snapshot containing only `config.json` + `tokenizer.json` (both matching `--preset safetensors`'s `*.json` clause) was therefore reported as fully cached even with `model.safetensors` absent. The fast-path now runs *after* the remote file listing and verifies every filtered remote file resolves to a real path under the snapshot dir; any single missing file declines the fast-path, falling back to the regular dispatch pipeline. Cost: one cheap HTTP listing per `hf-fm <repo>` invocation when the cache happens to be complete (vs zero before). Discovered while dogfooding v0.9.8 against the gemma-4-E2B-it download.

### Added

- **First tutorial: ["Inspect before you download"](docs/tutorials/inspect-before-downloading.md)** — opens a new `docs/tutorials/` directory and lands the v0.10.0 docs effort with a 6-minute walkthrough on `inspect --list` / `--dtypes` / `--tree` / `--filter` / `--limit`, using `zed-industries/zeta-2` as the running example. The tutorial pins every command to a specific commit SHA (`1529be60…`) so embedded outputs stay reproducible across re-reads, and closes with the practical pivot from "15.37 GiB BF16 won't fit on a 5060 Ti" to "bartowski's Q4_K_M GGUF (4.87 GiB) fits with ~9 GiB headroom" — using `search --tag gguf` and `list-files` to find the community quant. A closing line tees up `inspect --check-gpu` as the v0.10.1 capability that will turn the manual fit math into a one-line verdict. Cross-linked from the README documentation table and from the FAQ's "How do I see a model's tensor names without downloading it?" answer. Establishes the pattern for tutorials #2 (broader workflow) and the case-studies series queued for the rest of v0.10.0.
- **Streaming progress for `cache verify`** — long verifications (multi-GiB safetensors) used to print everything in a single batch at the end; the user had no signal that hf-fm was alive, just a multi-minute pause. Now: an `indicatif` spinner with the classic ASCII rotating bar (`| / - \`) ticks at 100 ms intervals while each file is hashed (`[3/52] Verifying W_dec_0.safetensors (10.97 GiB)`), and result lines (`✓ <name> SHA256 OK`, `— <name> no LFS hash`, `! <name> MISSING`, `✗ <name> SHA256 MISMATCH`) are streamed via `bar.suspend(|| println!(...))` as each file completes — so the user gets progressive output and a steady liveness signal regardless of file size or count. Files are now processed in alphabetical order so the streamed output order matches the returned `Vec<FileVerification>` and stays stable across runs. New public API: `hf_fetch_model::cache::verify_cache_with_progress` plus a `VerifyEvent` enum (`Started` / `FileStart` / `FileComplete`) — the existing `verify_cache` is now a thin no-op-callback wrapper. Library consumers building GUIs / TUIs can drive their own progress display; the CLI uses it to power the spinner.

- **`hf-fm cache verify <REPO_ID|N>`** — new subcommand to re-check the on-disk integrity of an already-downloaded repo. Fetches the expected SHA256 digests from the HuggingFace API and recomputes each cached file's digest locally, reporting one of four states per file: `SHA256 OK` (digest matches), `SHA256 MISMATCH` (digest differs — corrupted file, expected and actual digests both printed for forensics), `no LFS hash` (small git-stored files such as `config.json` carry no SHA256 in the API response, so verification is skipped), or `MISSING` (the file is listed remotely but not present in the snapshot directory). Exit-code semantics: any mismatch returns non-zero so the command composes cleanly into CI / cron-style integrity checks; `skipped` and `missing` alone are non-failures since a partial cache is a legitimate state. Accepts both repo IDs (`google/gemma-2-2b-it`) and `du`-style numeric indices (`hf-fm cache verify 2`), and a `--revision` flag to pin verification to a specific branch/tag — the same revision plumbing as `inspect`/`status`. New public library API: `hf_fetch_model::cache::verify_cache` plus `FileVerification` and `VerifyStatus` types, mirroring the shape of the existing `repo_status` API. Closes the third and final cache-management feature in v0.10.0 (alongside `cache gc` and `du --tree`); the v0.10.0 vision — see, act, verify — is now complete.
- **FAQ entry: "Why didn't my pipeline catch a download failure?"** — `hf-fm download-file ... 2>&1 | tail -20` masks hf-fm's exit code with `tail`'s, hiding failures behind a successful pipeline result. The new FAQ entry under *Errors and unexpected output* explains the mechanic and gives copy-paste recipes for recovering the real exit code via `${PIPESTATUS[0]}` (bash/zsh) and `$LASTEXITCODE` after the producer (PowerShell). Not specific to hf-fm — but worth naming because long downloads invite the `| tail` reflex.
- **`hf-fm du --tree`** — new hierarchical view of the local cache: each cached repo renders as a branch (size + file count, optional `--age` column, partial-download marker) and its files render as leaves (sorted by size descending). Reuses the box-drawing connectors (`├──`, `└──`, `│   `) and dynamic-column alignment established by `inspect --tree` in v0.9.6. File-leaf name and size columns are computed *globally* across every file in every repo so leaf size columns line up vertically across the whole tree, with the name column capped at 60 characters — outliers like the deeply-nested `gemma_2b_blocks.0.…/sae_weights.safetensors` SAE paths overflow locally without dragging short-name repos to that width. Composes with `--age` to add a last-modified column on each repo branch; conflicts (clap parse-time) with the positional repo argument because the per-repo view is already covered by `du <REPO_ID>`. One command-and-flag combo to learn for both flat and tree-shaped cache visibility — no separate `cache list` subcommand. First v0.10.0 cache-management feature to land.
- **`hf-fm cache gc`** — new garbage-collect subcommand for cached models, by age (`--older-than DAYS`) and/or size budget (`--max-size SIZE`, binary units only — `KiB`/`MiB`/`GiB`/`TiB`). When both flags are given, age eviction runs first; if the cache is still over budget, oldest non-protected repos are evicted next, oldest first with `repo_id` ascending as a deterministic tiebreaker. `--except REPO_ID` (repeatable) protects specific repos from eviction. `--dry-run` previews without deleting; `--list-kept` shows every kept repo for transparency on small caches (default: hidden so the preview stays terse on 50+ repo caches). Repos with active partial downloads (mtime within the last hour) are skipped to avoid racing with `hf-fm download` — run `cache clean-partial` first to clear stale partials. Decimal-prefixed size suffixes (`KB`, `MB`, `GB`, `TB`) are rejected with a pointer to the binary spelling, since `hf-fm` displays binary throughout and silent reinterpretation would mislead. Per-repo deletion failures are collected so a single permission error doesn't abort the run; the command exits non-zero if any deletion failed. Deletion logic now lives in a shared `delete_repo_dir` helper that also refuses to follow symlinks (a hardening that benefits `cache delete` too — previously a symlinked `models--…` dir could redirect `remove_dir_all` to its target on Unix). Closes the action gap in v0.10.0's cache-management story: `du` for seeing, `cache` for acting.

## [0.9.8] — Download durability

### Added

- **`--timeout-per-file-secs <N>` and `--timeout-total-secs <N>` CLI flags** — plumb the existing `FetchConfigBuilder::timeout_per_file` / `timeout_total` builder methods through to the CLI. Default behaviour is unchanged when the flags are omitted (300 s per file, no total limit), so users on fast connections see no difference. Slow-connection users on multi-GiB files can now extend the budget — e.g. `--timeout-per-file-secs 1800` for a file in the 5–15 GiB range. Wired into the default download command, `download-file`, and the `download-file` glob path; a shared `apply_timeout_overrides` helper keeps the per-call-site plumbing DRY.
- **Resume after interrupted chunked downloads** — new `chunked_state` module records per-chunk completion offsets in a small `{etag}.chunked.part.state` JSON sidecar next to the existing `.chunked.part` partial. `prepare_or_resume_temp_file` reuses an existing pair when its `(schema_version, etag, total_size, connections)` quadruple matches the current download — bytes already downloaded are kept and each chunk sends `Range: bytes=<start+completed>-<end>` to skip them. On any invariant mismatch (upstream etag changed, different `--connections-per-file`), both files are removed and a fresh state is written. The sidecar is updated atomically (write-tmp + rename) every 16 MiB of per-chunk progress and removed on successful finalize. Verified end-to-end on the real Gemma 4 multimodal repo (9.54 GiB) in the slow-connection regime where the v0.9.7 binary was unable to complete.
- **`PartialFile::sidecar_paths()`** — new public method on `cache::PartialFile` returning the `.chunked.part.state` and `.chunked.part.state.tmp` siblings of a partial, so `cache clean-partial` can sweep the whole bundle and not leave kilobyte-sized orphans behind.

### Changed

- **`.chunked.part` files now persist across transient interruptions** (timeout-induced future drop, Ctrl-C, panic, retryable chunk error). Previously every interruption wiped the partial via the v0.9.5 `TempFileGuard` Drop, forcing a full restart on the next invocation; combined with the 300 s/file ceiling that left slow-connection users on large files in an unrecoverable loop. The guard now defaults to **keep on drop** and exposes a `mark_corrupt()` opt-in for the rare case where post-guard bytes are known to be unusable. The v0.9.5 audit-fix intent — clean removal of orphan partials — is preserved through `cache clean-partial`, which handles the whole partial-download bundle including the new sidecars.
- **`download_chunked` doc-comment refreshed** to describe the resume contract and clarify which corruption cases are handled where (etag/total-size/schema-version mismatch in `prepare_or_resume_temp_file`; future post-guard checks via `TempFileGuard::mark_corrupt`).

## [0.9.7] — Inspect discoverability & newbie-friendly UX

### Added

- **`--preset npz`** — new filter preset for NumPy-based weight repositories such as Google's GemmaScope transcoders (hundreds of `.npz` files + `config.yaml`). Matches `*.npz`, `*.npy`, `config.yaml`, `*.json`, `*.txt`. Available on the default download command, `--dry-run`, and `list-files`. Library users can call `Filter::npz()` directly.
- **`inspect --list`** — new discovery flag that prints a numbered `.safetensors` file table (filename + size) for the target repo and exits without reading any headers. Output starts with `Repo:` and `Rev: <commit-sha>` so the user sees which snapshot they are looking at. Each row is a 1-based index that can be used as the `filename` argument on a follow-up run (e.g. `hf-fm inspect <repo> 3`). Files are sorted alphabetically, so shard orderings (`model-00001-of-00016.safetensors`, …) are natural.
- **`inspect` accepts a numeric index as `filename`** — `hf-fm inspect <repo> 3` resolves index 3 against the alphabetically-sorted safetensors list, prints `Resolving index 3 → <name> (repo rev: <short-sha>)` to stderr, and then runs the normal inspect flow on the resolved name. Literal filenames continue to work exactly as before. Out-of-range indices produce a clear error pointing to `--list`.
- **`inspect --list` composes with `--cached` and `--revision`** — `--cached` lists safetensors in the local snapshot (immune to remote changes); `--revision <sha>` locks both `--list` and the follow-up `inspect <n>` run to the same commit for end-to-end reproducibility. When no `--revision` is given, the tip line in the `--list` footer shows the full commit SHA users can pass to lock the view — the full SHA is the only form the HF API contract formally guarantees, so the tip matches the `Rev:` header rather than recommending a 12-char prefix.
- **New public API in `hf_fetch_model::repo`: `list_repo_files_with_commit`** — returns `(Vec<RepoFile>, Option<String>)` where the second element is the resolved commit SHA. Callers who only need the files should keep using `list_repo_files_with_metadata` (now a thin wrapper).
- **New public API in `hf_fetch_model::inspect`: `list_cached_safetensors`** — cheap name-and-size enumeration of `.safetensors` files in a cached snapshot, paired with the snapshot's commit SHA. Does not parse headers.
- **`docs/FAQ.md`** — first entry in the v0.10.0 docs effort ahead of schedule. Fifteen entries across five sections (About / Installation & auth / Discovery / Cache / Errors), conversational tone, clickable table of contents, style conventions embedded as HTML comments so future growth stays consistent. Linked from the `README.md` documentation index.

### Fixed

- **`inspect` on unsupported file types now emits a clear error** — previously, passing a non-safetensors file (e.g. an `.npz`) produced `safetensors header error … failed to parse header JSON: expected value at line 1 column 1`, which misled users into thinking the download was corrupt. The new error names the mismatch explicitly: `hf-fm inspect supports .safetensors only (got .npz for <path>)`. Driven by a new `FetchError::UnsupportedInspectFormat` variant.
- **Redundant HTTP client rebuild in the single-file download path** — `download_file_by_name` was silently shadowing the `reqwest::Client` built at the top of the function with a second, structurally identical one before the dispatch call. The TLS session and connection pool established during the metadata fetch were thrown away, forcing a fresh handshake. The second `build_client` is gone; the first client flows through to chunked downloads and the 416 fallback.
- **`--list` tip no longer recommends a short commit SHA** — the footer tip in `inspect --list` used to print `Pass --revision <12-char-prefix>`, while the `Rev:` header printed the full 40-char SHA. Short-SHA resolution is accepted by the HF Hub in practice but not formally part of the API contract. The tip now prints the full SHA, matching the header — one atomic string to copy-paste, unambiguous against any API endpoint.

### Changed

- **`--help` now sorts subcommands alphabetically** — `hf-fm --help` and `hf-fm cache --help` list commands in alphabetical order at runtime via `display_order`, so new commands land in the right place automatically regardless of enum declaration order.
- **`list-families` wraps repo lists onto multiple lines** — each repo now prints on its own line, indented under the family column. The single run-on line for large families (e.g., `llama`) is gone, and the separator rule is sized to the widest single repo name rather than the widest joined list. The output now begins with a `Cache: <absolute path>` header so it is immediately clear which cache directory is being listed.
- **`indicatif` bumped `0.17` → `0.18`** — no source changes required; every API we call (`MultiProgress`, `ProgressBar`, `ProgressStyle`) is unchanged across the bump. The 0.18 break was internal (`console` crate 0.15 → 0.16). Concrete build-tree win: `hf-hub` already pulled in indicatif 0.18, so we were compiling indicatif twice; now we compile it once.
- **`sha2` bumped `0.10` → `0.11`** — one three-line source change in [`src/checksum.rs`](src/checksum.rs) to replace `format!("{digest:x}")` with a manual lowercase-hex loop, because `sha2` 0.11 returns `hybrid_array::Array<u8, _>` from `finalize()`, which (unlike the old `generic_array::GenericArray`) does not implement `fmt::LowerHex`. Verified against the existing known-value test (SHA256 of `"hello\n"` matches the canonical `5891b5…6be03` digest). Follow-on benefit: `generic-array`, `block-buffer 0.10`, `crypto-common 0.1`, and `typenum 1.19` all drop out of the build tree, and the recurring `cargo install` notice `"Adding generic-array v0.14.7 (available: v0.14.9)"` is silenced (a transitive `crypto-common 0.1.7` had pinned `generic-array = "=0.14.7"` — nothing in that chain could budge until `sha2` moved to the `digest` 0.11 wave).
- **Public sync functions in `hf_fetch_model::inspect` now document blocking I/O** — `inspect_safetensors_local`, `inspect_safetensors_cached`, `inspect_repo_safetensors_cached`, and the new `list_cached_safetensors` each gain a `# Blocking I/O` doc section warning async library consumers to wrap calls in [`tokio::task::spawn_blocking`]. Particularly relevant on network-mounted caches (NFS/CIFS) where a `stat`-per-file walk across a large sharded repo can take seconds. A future minor release may add companion `_async` wrappers so the `spawn_blocking` dance becomes transparent.
- **Regression tests for `cache_layout` path builders** — two small unit tests inside `src/cache_layout.rs` cover `blob_path` (basic `{repo_dir}/blobs/{etag}` shape) and `temp_blob_path` (the non-obvious edge case: etags containing periods must survive the `.chunked.part` suffix intact, which is why the function uses string concatenation rather than `Path::with_extension`). Rounds out the existing `cache_layout_matches_hf_hub` integration test, which validated `repo_folder_name` / `snapshot_dir` / `ref_path` against real `hf-hub` output but left the blob-path functions untested.

## [0.9.6] — Inspect discoverability

### Added

- **`inspect --dtypes` flag** — shows a per-dtype summary (tensor count, parameter count, byte size) instead of listing individual tensors. Composes with `--filter` to show dtype breakdown for a subset of tensors.
- **`inspect --dtypes --json` composition** — emits the dtype summary as JSON (`{ dtypes: [...], total_tensors, total_params }`) instead of silently falling back to the full-header JSON. Useful for scripts that aggregate dtype usage across a cache.
- **`inspect --tree` flag** — hierarchical tensor-name view grouped by dotted namespace prefix. Single-child chains collapse into dotted paths (e.g., `model.embed_audio.embedding_projection.weight` shown on one line); contiguous numeric sibling groups with structurally-identical sub-trees collapse to `layers.[0..27]   (×28)` with the template shown once. Solves *structural* discovery for newbies inspecting large models — pairs naturally with `--dtypes` (aggregation discovery). Composes with `--filter` (apply before tree building) and `--json` (separate schema with tagged-enum nodes: `leaf` / `branch` / `ranged`). Conflicts with `--dtypes` and `--limit`.
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
