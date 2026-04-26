# Frequently Asked Questions

<!-- Last updated: 2026-04-26, hf-fm v0.9.8 -->

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
  - [What is the `cli` feature and do I need it?](#what-is-the-cli-feature-and-do-i-need-it)
  - [How do I pass a HuggingFace token? Why does a gated model fail?](#how-do-i-pass-a-huggingface-token-why-does-a-gated-model-fail)
- [Discovery — finding what to inspect or download](#discovery--finding-what-to-inspect-or-download)
  - [A repo has many `.safetensors` files — how do I pick one to inspect?](#a-repo-has-many-safetensors-files--how-do-i-pick-one-to-inspect)
  - [How do I see a model's tensor names without downloading it?](#how-do-i-see-a-models-tensor-names-without-downloading-it)
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
  - [What does `hf-fm inspect supports .safetensors only (got .npz for …)` mean?](#what-does-hf-fm-inspect-supports-safetensors-only-got-npz-for--mean)
  - [I got a `checksum mismatch` error — what do I do?](#i-got-a-checksum-mismatch-error--what-do-i-do)
  - [Why does `inspect` say `Source: remote (2 HTTP requests)`?](#why-does-inspect-say-source-remote-2-http-requests)

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
3. `safetensors_explorer` currently covers **safetensors and GGUF**; hf-fm covers safetensors today, with GGUF on the roadmap.

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

Indices are stable for as long as the repository does not change remotely — for scripted reproducibility, pass the `--revision <sha>` that `--list` prints in its header to both commands.

### How do I see a model's tensor names without downloading it?

Run `hf-fm inspect <repo>` with no filename and it will inspect every `.safetensors` file in the repository. For one specific file, add the filename (or an index from `--list`). Internally, hf-fm fetches only the JSON header via an HTTP Range request — for a typical 2 GiB safetensors file, you transfer maybe 70 KiB of metadata. Add `--tree` for the hierarchical view that groups numeric layers (`layers.[0..27]   (×28)`), or `--dtypes` for a dtype-and-parameter summary.

### How do I list only the weight files in a repo, not the tokenizer and README?

Use `list-files` with a `--preset` matching the weight format:

```
hf-fm list-files google/gemma-2-2b-it --preset safetensors
hf-fm list-files google/gemma-scope-2b-pt-transcoders --preset npz
```

The presets bundle the weight extension plus the common config files (`*.json`, `*.txt`, and for `npz` the `config.yaml` GemmaScope uses). Available presets are `safetensors`, `gguf`, `npz`, `pth`, and `config-only` — `hf-fm list-files --help` shows the full list.

### How do I see what is already cached locally?

Three views, in order of detail:

- `hf-fm list-families` — grouped by model architecture (`gemma2`, `llama`, `qwen2`, …), one repo per line under each family. Shows which cache directory you are looking at on the first line.
- `hf-fm du` — one row per repo with disk usage and a partial-download marker, sorted largest first. Type `hf-fm du <N>` to drill into the Nth repo and see its files.
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

### What does `hf-fm inspect supports .safetensors only (got .npz for …)` mean?

Exactly what it says: `inspect` currently parses safetensors headers only, and you passed a file with a different extension (likely `.npz`, `.gguf`, `.bin`, or `.pth`). This is not a corrupted file — hf-fm just does not know how to read that format yet. If you are on a NumPy-based repo like GemmaScope, use `hf-fm list-files <repo> --preset npz` to see what is available; GGUF inspection is on the roadmap for a future release.

### I got a `checksum mismatch` error — what do I do?

A `checksum mismatch` means the file's computed SHA256 does not match the hash HuggingFace's API reported. Most of the time this is caused by a truncated download — delete the local file (or run `hf-fm cache delete <repo>` for the whole repo) and retry; the multi-connection download path will re-fetch it cleanly. If the mismatch repeats on a fresh download, that is genuinely unusual — open an issue on [GitHub](https://github.com/PCfVW/hf-fetch-model/issues) with the repo ID, filename, and the error message.

### Why does `inspect` say `Source: remote (2 HTTP requests)`?

Reading a safetensors header remotely takes two ranged HTTP requests: the first fetches the 8-byte little-endian `u64` at the start of the file that encodes the header's length, and the second fetches exactly that many bytes of JSON. That is the entire network cost of an `inspect` run — the multi-gigabyte weight data is never touched. When the file is already in your local cache, the line reads `Source: cached` instead and there are no HTTP requests at all.
