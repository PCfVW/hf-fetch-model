# Frequently Asked Questions

<!-- Last updated: 2026-06-03, hf-fm v0.10.4 (--check-gpu --context KV budgeting) -->

<!--
STYLE CONVENTIONS for editing this FAQ — keep growth consistent.

1. Tone: conversational, matching the project's README voice. Address the
   reader as "you". Prefer short paragraphs over bullet points.
2. Question format: "### How do I …?" or "### What is …?" as the heading.
   Use natural-language questions — GitHub's anchor generator produces
   usable slugs from them (e.g. "#how-do-i-install-it"). Keep them in
   the contents list too.
3. Answer length: 2–4 sentences, plus at most one small code block with
   a concrete command. Anything longer is a tutorial, not an FAQ entry —
   link out instead.
4. Shell context: when showing env vars, give both variants side by
   side — `HF_TOKEN=… hf-fm …` for bash/zsh and
   `$env:HF_TOKEN="…"; hf-fm …` for PowerShell — so Windows users
   are not left guessing. (See CLAUDE.md for the project's shell policy.)
5. "MSRV" should be spelled out the first time as "Minimum Rust Version
   (MSRV)"; the acronym is OK on reuse.
6. Freshness marker: update the "Last updated" date and version at the
   top whenever any answer text changes — not for typo fixes or new
   entries that don't touch existing answers.
7. Scope: answer questions about features that actually ship today.
   Do not pre-document unshipped work (cache gc, GGUF inspect, etc.) —
   those get dedicated docs when they land.
8. Grouping: if a section grows past ~5 entries, consider splitting it.
   If an entry grows past ~6 sentences, consider promoting it to
   docs/tutorials/ or docs/case-studies/ and linking from here.
-->

A living list of the questions we and our early users have actually run into. If your question is not here, please open an issue on [GitHub](https://github.com/PCfVW/hf-fetch-model/issues) — we add entries as real questions arrive.

## Contents

- [About hf-fetch-model](#about-hf-fetch-model)
  - [What is hf-fm? How does it differ from `huggingface-cli`?](#what-is-hf-fm-how-does-it-differ-from-huggingface-cli)
  - [How does it differ from `safetensors_explorer`?](#how-does-it-differ-from-safetensors_explorer)
  - [Why the two binary names, `hf-fetch-model` and `hf-fm`?](#why-the-two-binary-names-hf-fetch-model-and-hf-fm)
  - [Is it stable? What does a `0.9.x` version number mean?](#is-it-stable-what-does-a-09x-version-number-mean)
- [Installation and authentication](#installation-and-authentication)
  - [How do I install it? What is the Minimum Rust Version?](#how-do-i-install-it-what-is-the-minimum-rust-version)
  - [How do I upgrade hf-fm? Why does `cargo install` silently keep the old version?](#how-do-i-upgrade-hf-fm-why-does-cargo-install-silently-keep-the-old-version)
  - [What is the `cli` feature and do I need it?](#what-is-the-cli-feature-and-do-i-need-it)
  - [How do I pass a HuggingFace token? Why does a gated model fail?](#how-do-i-pass-a-huggingface-token-why-does-a-gated-model-fail)
- [Discovery — finding what to inspect or download](#discovery--finding-what-to-inspect-or-download)
  - [A repo has many `.safetensors` files — how do I pick one to inspect?](#a-repo-has-many-safetensors-files--how-do-i-pick-one-to-inspect)
  - [A repo has many `.safetensors` files — can I pick one interactively?](#a-repo-has-many-safetensors-files--can-i-pick-one-interactively)
  - [How do I see a model's tensor names without downloading it?](#how-do-i-see-a-models-tensor-names-without-downloading-it)
  - [How do I compare two HuggingFace models structurally?](#how-do-i-compare-two-huggingface-models-structurally)
  - [How do I know if a model fits on my GPU?](#how-do-i-know-if-a-model-fits-on-my-gpu)
  - [How do I list only the weight files in a repo, not the tokenizer and README?](#how-do-i-list-only-the-weight-files-in-a-repo-not-the-tokenizer-and-readme)
  - [How do I see what is already cached locally?](#how-do-i-see-what-is-already-cached-locally)
- [Cache location and management](#cache-location-and-management)
  - [Where does hf-fm store downloaded files? Is the layout compatible with Python `huggingface_hub`?](#where-does-hf-fm-store-downloaded-files-is-the-layout-compatible-with-python-huggingface_hub)
  - [My disk is getting full — which models are taking the most space?](#my-disk-is-getting-full--which-models-are-taking-the-most-space)
  - [What is a `.chunked.part` file, and is it safe to delete?](#what-is-a-chunkedpart-file-and-is-it-safe-to-delete)
- [Downloading large files on slow connections](#downloading-large-files-on-slow-connections)
  - [Why does my download keep timing out, and how do I extend the budget?](#why-does-my-download-keep-timing-out-and-how-do-i-extend-the-budget)
  - [My download was interrupted — do I have to start over?](#my-download-was-interrupted--do-i-have-to-start-over)
- [Errors and unexpected output](#errors-and-unexpected-output)
  - [Which file formats can `inspect` read?](#which-file-formats-can-inspect-read)
  - [I got a `checksum mismatch` error — what do I do?](#i-got-a-checksum-mismatch-error--what-do-i-do)
  - [Why does `inspect` say `Source: remote (2 HTTP requests)`?](#why-does-inspect-say-source-remote-2-http-requests)
  - [Why didn't my pipeline catch a download failure?](#why-didnt-my-pipeline-catch-a-download-failure)

---

## About hf-fetch-model

### What is hf-fm? How does it differ from `huggingface-cli`?

`hf-fetch-model` is a Rust CLI and library for downloading, inspecting, and comparing HuggingFace models. It overlaps with Python's `huggingface-cli` on the basics (fetch a repo, list files, manage the cache) but diverges in three places:

1. it downloads large files with many parallel HTTP connections,
2. it can read a `.safetensors` file's tensor-name/dtype/shape metadata **without downloading the weights** by doing a narrow HTTP Range request against the header,
3. it ships as a standalone binary with no Python dependency. It writes to the same cache directory as Python `huggingface_hub`, so the two tools coexist happily.

### How does it differ from `safetensors_explorer`?

[`safetensors_explorer`](https://github.com/EricLBuehler/safetensors_explorer) by Eric Buehler is a Rust TUI for exploring safetensors and GGUF files — full-screen interactive browsing, fuzzy search over tensor names, directory-wide merging. It and `hf-fm inspect` are complementary rather than competing:

1. `safetensors_explorer` is an interactive **TUI** that shines at exploring a model locally with the keyboard; `hf-fm inspect` is a **CLI** that produces printable output (pipeable into other tools, pasteable into bug reports),
2. `safetensors_explorer` reads **local files only**; `hf-fm inspect` additionally reads the tensor metadata of a **remote** model via HTTP Range, before anything is downloaded,
3. `safetensors_explorer` currently covers **safetensors and GGUF**; hf-fm covers **safetensors** (remote or cached) plus **GGUF / NumPy `.npz` / PyTorch `.pth`** for cached files (since v0.10.2–v0.10.3), with remote inspect for those three on the roadmap (v0.11).

Reach for `safetensors_explorer` when you want to sit at a TUI and explore a model locally; reach for `hf-fm inspect` when you want to preview a remote model before downloading, or when you want text output you can pipe and paste.

### Why the two binary names, `hf-fetch-model` and `hf-fm`?

They are the same program. `hf-fetch-model` is the long form that appears on crates.io and shows up in `cargo install` output; `hf-fm` is the short alias you actually type. Both binaries are installed by the same `cargo install` command — pick whichever reads better in your shell history.

### Is it stable? What does a `0.9.x` version number mean?

hf-fm is production-grade on the download and inspect paths we use daily, but the `0.x` version number means we follow Cargo's SemVer interpretation: any minor bump (`0.9 → 0.10`) is allowed to break public API. Patch bumps (`0.9.6 → 0.9.7`) do not break the API — so if you are pinning a dependency, a `~0.9` or `=0.9.7` constraint is safe. The `CHANGELOG.md` always lists breaking changes under each release.

---

## Installation and authentication

### How do I install it? What is the Minimum Rust Version?

From crates.io, in one command:

```
cargo install hf-fetch-model --features cli
```

That installs both `hf-fetch-model` and `hf-fm` into `~/.cargo/bin` (or `%USERPROFILE%\.cargo\bin\` on Windows). The Minimum Rust Version (MSRV) is **1.88**, declared via the `rust-version` field in `Cargo.toml` — Cargo warns if your active toolchain is older. If `cargo install` fails complaining about the Rust version, run `rustup update stable`.

### How do I upgrade hf-fm? Why does `cargo install` silently keep the old version?

Pass `--force` to `cargo install` and then re-check the version that actually landed:

```
cargo install hf-fetch-model --features cli --force
hf-fm --version
```

Without `--force`, `cargo install` short-circuits whenever **any** version of the binary is already in `~/.cargo/bin/` — even when crates.io has a newer release. The "already installed" notice is logged to stderr at low priority and is easy to miss in a busy terminal: the install command exits `0`, looks like it succeeded, but the binary on `PATH` is unchanged. `--force` bypasses the short-circuit and always builds the latest. The companion note at [`docs/dogfooding-feedbacks/cargo-install-silent-skip.md`](dogfooding-feedbacks/cargo-install-silent-skip.md) captures the failure mode in full — including a reproduction recipe and the proposed `hf-fm --check-update` flag tracked as a future patch-release candidate.

### What is the `cli` feature and do I need it?

hf-fm ships as a **library crate by default** — no command-line binaries unless you ask for them. The `cli` feature pulls in `clap`, `tracing-subscriber`, and the progress bar dependencies needed to build the `hf-fetch-model` and `hf-fm` executables. You need it when running `cargo install`; you do **not** need it when adding hf-fetch-model as a dependency to your own Rust project — in that case, use the default library-only build and call `FetchConfig::builder()` directly.

### How do I pass a HuggingFace token? Why does a gated model fail?

Either pass `--token <value>` on the command line, or set the `HF_TOKEN` environment variable once and forget about it — every subcommand reads it automatically. Gated models (like Meta's Llama releases or Google's Gemma family) require both a valid token and that you have accepted the licence on the model's HuggingFace page; without either, the first HTTP request returns a 401 or 403 and hf-fm surfaces it as a `FetchError::Api` or `FetchError::RepoNotFound`.

```
# bash / zsh
HF_TOKEN=hf_xxx hf-fm meta-llama/Llama-3.2-1B

# PowerShell
$env:HF_TOKEN="hf_xxx"; hf-fm meta-llama/Llama-3.2-1B
```

---

## Discovery — finding what to inspect or download

### A repo has many `.safetensors` files — how do I pick one to inspect?

Use `inspect --list` to see them all with sizes, then call `inspect <repo> <n>` with the index you want:

```
hf-fm inspect Qwen/Qwen2.5-Coder-7B-Instruct --list
# → 1  model-00001-of-00004.safetensors  4.54 GiB
#   2  model-00002-of-00004.safetensors  4.59 GiB
#   …
hf-fm inspect Qwen/Qwen2.5-Coder-7B-Instruct 2 --tree
```

Indices are stable for as long as the repository does not change remotely — for scripted reproducibility, pass the `--revision <sha>` that `--list` prints in its header to both commands. Since v0.10.5, the listing covers every tensor format `inspect` can read (`.safetensors` / `.gguf` / `.npz` / `.pth`), so a GGUF-only repo lists and index-resolves the same way. Prefer one command over two? The next question covers the interactive `--pick` flag, which shares the same numbered universe.

### A repo has many `.safetensors` files — can I pick one interactively?

Yes — `--pick` (since v0.10.5) collapses the two-step `--list` → `inspect <repo> <n>` workflow into one command:

```
hf-fm inspect little-lake-studios/demoncore-flux demonCORE --pick --dtypes
# Multiple tensor files match "demonCORE" in little-lake-studios/demoncore-flux:
#   1  transformer/demonCORENSFW_fluxV11.safetensors     15.40 GiB
#   2  transformer/demonCORESFWNSFW_fluxV12.safetensors  15.40 GiB
#   3  transformer/demonCORESFWNSFW_fluxV13.safetensors  15.40 GiB
# Pick [1..3]: 3
# Resolving to transformer/demonCORESFWNSFW_fluxV13.safetensors
```

Under `--pick`, the positional argument is a **case-insensitive substring** filter, never a numeric index: when exactly one file matches, hf-fm skips the prompt and resolves directly (printing `Resolving to <name>` on stderr); with no positional at all, the picker offers every supported tensor file. It composes with every rendering flag (`--tree`, `--dtypes`, `--filter`, `--limit`, `--check-gpu`, `--json`) — the table and prompt go to **stderr**, so `--pick --json > out.json` still writes clean JSON to the file. Pressing Enter on an empty line (or Ctrl-D / Ctrl-Z) cancels with a non-zero exit code.

`--pick` requires an interactive terminal (stdin and stderr attached). In scripts and CI, use `--list` + the numeric index instead — the two workflows coexist and share the same alphabetically-sorted file universe, so `#3` means the same file in both.

### How do I see a model's tensor names without downloading it?

Run `hf-fm inspect <repo>` with no filename and it will inspect every `.safetensors` file in the repository. For one specific file, add the filename (or an index from `--list`). Internally, hf-fm fetches only the JSON header via an HTTP Range request — for a typical 2 GiB safetensors file, you transfer maybe 70 KiB of metadata. Add `--tree` for the hierarchical view that groups numeric layers (`layers.[0..27]   (×28)`), or `--dtypes` for a dtype-and-parameter summary. For a complete walkthrough on a real 4-shard model, see [Inspect before you download](tutorials/inspect-before-downloading.md).

### How do I compare two HuggingFace models structurally?

`hf-fm diff <REPO_A> <REPO_B>` classifies every tensor across both repos into four buckets — *only-in-A*, *only-in-B*, *dtype/shape differences*, and *matching* — by reading each side's safetensors headers via HTTP Range (no weight data downloaded). For scaled-sibling pairs (one model is a bigger sibling of the other in the same family) the per-tensor output is dominated by the extra-layer wall, so reach for `--dtypes` instead:

```
$ hf-fm diff openai/gpt-oss-20b openai/gpt-oss-120b --dtypes

  A: openai/gpt-oss-20b
  B: openai/gpt-oss-120b

  Dtype  A Tensors     A Size  B Tensors      B Size      Δ Size
  U8           192  18.91 GiB        288  113.46 GiB  +94.55 GiB
  BF16         630   6.72 GiB        942    8.07 GiB   +1.35 GiB
  ──────────────────────────────────────────────────────────────
  A: 822 tensors, 25.63 GiB | B: 1230 tensors, 121.54 GiB | Δ: +408 tensors, +95.90 GiB
```

This is a side-by-side per-dtype histogram with a signed Δ Size column. Same dtype mix on both sides plus proportional scaling = scaled siblings (same architecture, different size). A dtype present in only one side, or a wildly disproportionate Δ Size ratio across dtypes, would point at an architectural variant rather than a clean scale-up. Composes with `--filter` (histograms aggregate over filtered tensors only) and `--json`. The complementary text mode `diff` (without `--dtypes`) remains useful for short tensor-level inspections — see `hf-fm diff --help`.

For deeper structural analysis on the per-tensor list (when `--dtypes` says "same dtype mix but I want to see *which* tensors differ"), the JSON output now ships a `byte_count` field on every entry. Pipe it through `jq` to collapse `only_a` / `only_b` entries by name pattern — useful when one side has 200 extra layers and you want the *kinds* of new tensors, not the raw 200-row list:

```bash
hf-fm diff org/model-A org/model-B --json \
  | jq -r '
      .only_b
      | group_by(.name | gsub("[0-9]+"; "{N}"))
      | map({
          pattern: (.[0].name | gsub("[0-9]+"; "{N}")),
          tensors: length,
          bytes: (map(.b.byte_count) | add),
        })
      | sort_by(-.bytes)
      | .[] | "\(.pattern)  \(.tensors)  \(.bytes) bytes"
    '
```

For a scaled-sibling pair this collapses `model.layers.0.self_attn.q_proj.weight`, `model.layers.1.self_attn.q_proj.weight`, … into a single `model.layers.{N}.self_attn.q_proj.weight` line with a count and a summed-byte total. The JSON-first approach lets you iterate on the collapse heuristic (regex, segment, expert-routing-aware) against your own pair before any of it becomes a built-in flag. The same recipe with `.only_a` swapped in does the symmetric job.

### How do I know if a model fits on my GPU?

Pass `--check-gpu` for a weights-vs-VRAM verdict, and add `--context N` for the full picture including the KV cache:

```
hf-fm inspect meta-llama/Llama-3.2-3B --cached --check-gpu --context 32768
```

`--check-gpu` alone reads the device's total / free / used VRAM via [`hypomnesis`](https://crates.io/crates/hypomnesis) (NVML on Linux/Windows, DXGI on Windows; `nvidia-smi` fallback), sums the model's weight bytes across every shard, and prints a one-line `✓ X.YZ GiB headroom` / `✗ short by X.YZ GiB` verdict. Default device is `0`; pass `--check-gpu 1` on a multi-GPU box. Works on the cached and network paths identically. On a system with no NVIDIA GPU the verdict reports `unavailable — <reason>` and the command still exits 0 — `--check-gpu` is informational, never a gate.

**`--context N` (v0.10.4) makes it real.** Weights are the easy part; on a consumer card the KV cache is what decides whether a long context fits. `--context N` reads the model's `config.json` and computes the KV bytes at sequence length `N`, then measures the fit against `weights + KV`:

```
  Model weights:  5.98 GiB  (BF16, 3.21B params)
  KV cache @ ctx=32768:  3.50 GiB  (BF16)
  Total:          9.48 GiB  (weights + KV)
  Fit:            ✓ 4.20 GiB headroom (weights + KV; runtime extra)
```

The estimate is **parameter-driven, not a per-model lookup table** — it applies the universal formula `2 × layers × kv_heads × head_dim × N × dtype_bytes` to the model's actual architecture integers, so a model hf-fm has never seen computes correctly. It is architecture-aware where the simple formula breaks:

- **GQA** (Llama-3, Mistral): uses `num_key_value_heads`, not the query-head count (often a 4×+ difference).
- **Sliding window** (Mistral, Phi): KV caps at the window. Gemma-2 / Gemma-3 mix local and global layers, which are *blended* (global layers at full context + local layers at the window).
- **Hybrid Mamba/attention** (Granite-4, Nemotron-H, Bamba, Qwen3-Next): KV applies only to the few attention layers, and a separate `Recurrent state` line reports the fixed Mamba2 state (which does **not** grow with context). This is why a hybrid fits far more context than a same-size transformer.
- **MLA** (DeepSeek): the naive formula overestimates ~10×, so it is **skipped** with a note rather than printing a wrong number.

`--context` requires `--check-gpu`, and composes with `--json` (the `gpu_check` object gains a `kv_cache` sub-object and `model.total_bytes`).

**Known limitations** — all flagged in the output, none silent:

- **MLA is skipped, not estimated.** DeepSeek-V2/V3 print `KV cache: skipped (MLA / latent attention — naive estimate unreliable)` and fall back to the weights-only verdict.
- **Mixed sliding-window is approximate** (within a few percent): the Gemma blend models the *count* of local vs global layers, not their exact positions in the stack.
- **KV dtype is assumed equal to the activation dtype.** The KV element size comes from the config's `torch_dtype` (bf16 / fp16 = 2 bytes), independent of weight quantization. If you run an FP8 / Q4 KV cache to fit more context, the real figure is smaller — so treat the reported number as a safe upper bound.
- **Non-Mamba2 recurrent state is excluded.** For Qwen3-Next (Gated DeltaNet) and Jamba (Mamba1) the *attention* KV is correct, but the recurrent state is labeled `excluded (small, constant)` rather than computed — it is tens of MiB and constant in context, so it never flips a consumer-GPU verdict.

### How do I list only the weight files in a repo, not the tokenizer and README?

Use `list-files` with a `--preset` matching the weight format:

```
hf-fm list-files google/gemma-2-2b-it --preset safetensors
hf-fm list-files google/gemma-scope-2b-pt-transcoders --preset npz
```

The presets bundle the weight extension plus the common config files (`*.json`, `*.txt`, and for `npz` the `config.yaml` GemmaScope uses). Available presets are `safetensors`, `gguf`, `npz`, `pth`, and `config-only` — `hf-fm list-files --help` shows the full list.

### How do I see what is already cached locally?

Four views, in order of detail:

- `hf-fm list-families` — grouped by model architecture (`gemma2`, `llama`, `qwen2`, …), one repo per line under each family. Shows which cache directory you are looking at on the first line.
- `hf-fm du` — one row per repo with disk usage and a partial-download marker, sorted largest first. Type `hf-fm du <N>` to drill into the Nth repo and see its files.
- `hf-fm du --tree` — hierarchical view of every cached repo and its files in a single box-drawing tree, with sizes right-aligned across all rows. Composes with `--age` to add a last-modified column on each repo branch.
- `hf-fm status <repo>` — per-file completeness status for one repo.

---

## Cache location and management

### Where does hf-fm store downloaded files? Is the layout compatible with Python `huggingface_hub`?

Yes, fully compatible. hf-fm writes to `~/.cache/huggingface/hub/` by default (or `$HF_HOME/hub/` when the environment variable is set), using the same `models--<org>--<name>/` directory layout, the same `blobs/` and `snapshots/` subfolders, and the same `refs/` ref files that Python's `huggingface_hub` produces. You can download a model with Python and inspect it with hf-fm, or vice versa, without either tool noticing the other. On Windows the path is `%USERPROFILE%\.cache\huggingface\hub\`.

### My disk is getting full — which models are taking the most space?

Run `hf-fm du` for a size-sorted summary. Each row has a `#` index you can pass back to the command to see that repo's files one level deeper (`hf-fm du 3`), and `du --age` adds a "last modified" column that is handy for spotting models you downloaded once months ago and never touched again. When you have identified what to remove, use `hf-fm cache delete <repo-id>` (repo ID or the numeric index from `du`); it prompts for confirmation unless you pass `--yes`.

### What is a `.chunked.part` file, and is it safe to delete?

`.chunked.part` files are temporary staging files for multi-connection downloads — hf-fm writes downloaded bytes there before renaming them into the final blob. From v0.9.8 onwards they also persist across interruptions so the next invocation can resume from them, paired with a small `.chunked.part.state` JSON sidecar that tracks per-chunk progress. They are safe to delete when you have abandoned a download for good — run `hf-fm cache clean-partial` for a prompted cleanup (add `--dry-run` to preview, `--yes` to skip confirmation); the sweep removes the partial and its sidecar together. A repo with a stale partial also shows up as a `●` marker in `hf-fm du`.

---

## Downloading large files on slow connections

### Why does my download keep timing out, and how do I extend the budget?

By default hf-fm gives each file a 300-second budget — fine for typical multi-GiB safetensors at typical home-broadband speeds, but it can run out before a 10 GiB file finishes if your effective throughput is below ~35 MiB/s. Pass `--timeout-per-file-secs <N>` to extend it; `1800` (30 minutes) is a sensible value for files in the 5–15 GiB range on slower links. There is also a `--timeout-total-secs` flag that bounds the entire batch when you are downloading many files at once.

```
hf-fm google/gemma-4-E2B-it --preset safetensors --timeout-per-file-secs 1800
```

### My download was interrupted — do I have to start over?

No. As of v0.9.8, hf-fm preserves the partial `.chunked.part` file plus a small `.chunked.part.state` sidecar that records per-chunk progress; the next time you run the same command, each parallel chunk picks up from where it stopped. This works across timeouts, Ctrl-C, and crashes — as long as the file on the remote did not change (the etag is verified before resuming). If the etag does not match, hf-fm starts fresh and tells you so.

---

## Errors and unexpected output

### Which file formats can `inspect` read?

Four tensor formats: `.safetensors` (remote via HTTP Range, or cached) and `.gguf` / NumPy `.npz` / PyTorch `.pth` (cached only — pass `--cached` after downloading; remote inspect for these three is on the roadmap for v0.11). Two errors point at the edges of that support: an unsupported extension (`.bin`, etc.) gives `hf-fm inspect supports .safetensors, .gguf, .npz, or .pth (got .bin for …)`, and inspecting a `.gguf` / `.npz` / `.pth` *without* `--cached` gives `remote <FORMAT> inspect not yet supported (planned for v0.11): pass --cached after downloading`. On a repo you have not fetched yet, `hf-fm list-files <repo>` shows what is available first.

### I got a `checksum mismatch` error — what do I do?

A `checksum mismatch` means the file's computed SHA256 does not match the hash HuggingFace's API reported. Most of the time this is caused by a truncated download — delete the local file (or run `hf-fm cache delete <repo>` for the whole repo) and retry; the multi-connection download path will re-fetch it cleanly. If the mismatch repeats on a fresh download, that is genuinely unusual — open an issue on [GitHub](https://github.com/PCfVW/hf-fetch-model/issues) with the repo ID, filename, and the error message.

### Why does `inspect` say `Source: remote (2 HTTP requests)`?

Reading a safetensors header remotely takes two ranged HTTP requests: the first fetches the 8-byte little-endian `u64` at the start of the file that encodes the header's length, and the second fetches exactly that many bytes of JSON. That is the entire network cost of an `inspect` run — the multi-gigabyte weight data is never touched. When the file is already in your local cache, the line reads `Source: cached` instead and there are no HTTP requests at all.

### Why didn't my pipeline catch a download failure?

If you wrap a hf-fm command in a shell pipe like `hf-fm download-file ... 2>&1 | tail -20`, the pipeline's exit code is the **last** command's, not hf-fm's. So a hf-fm timeout or network error can be hidden behind a successful `tail`, leaving you thinking the download worked when it didn't. This is true of every CLI tool, not just hf-fm — but it bites here because long downloads invite the impulse to wrap them in `| tail` to keep the terminal tidy. hf-fm does print `error: …` lines to stderr on failure, but if you fold stderr into stdout via `2>&1` and then truncate, the failure signal lives in the tail of the output rather than the exit code.

To check hf-fm's real exit code through a pipe:

```
# bash / zsh — PIPESTATUS holds each pipeline stage's exit code
hf-fm download-file ... 2>&1 | tail -20
echo "hf-fm exit: ${PIPESTATUS[0]}"

# PowerShell — capture and inspect $LASTEXITCODE after the producer
hf-fm download-file ... 2>&1 | Select-Object -Last 20
echo "hf-fm exit: $LASTEXITCODE"
```

Or skip the pipe and check `$?` (bash/zsh) / `$LASTEXITCODE` (PowerShell) directly after the bare command — simplest when you do not actually need to truncate the output.
