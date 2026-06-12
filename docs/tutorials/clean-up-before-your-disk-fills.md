# Clean up before your disk fills

*See what the HuggingFace cache holds, decide what to keep, and reclaim the rest — with a dry-run before anything is deleted.*

*~1,490 words · about 6 min read*

<!-- Last updated: 2026-06-12, hf-fm v0.10.5 -->

<!--
STYLE CONVENTIONS for editing this tutorial — keep growth consistent.

1. Tone: match the FAQ and the inspect tutorial. Conversational, address
   the reader as "you", short paragraphs over bullet lists where prose
   works.
2. Reproducibility: unlike the inspect tutorial, there is no revision to
   pin — the cache is machine-local. Output blocks come from two capture
   sessions on the same machine: the du / gc / verify / path blocks are
   from 2026-06-11 (592 GiB, 66 repos); the partial-download blocks
   (status PARTIAL, du ●, clean-partial, resume) are from a 2026-06-12
   follow-up that staged an interrupted download (note 3), so they show
   one extra repo. The reader's numbers WILL differ; the column shapes and
   legends must not. The two sessions are deliberately not reconciled into
   one frozen total — the narrative moves forward in time as the reader
   downloads and interrupts a model.
3. Safety: every destructive command appears with --dry-run, or with its
   confirmation prompt visible and answered `n`. Never paste an output
   that shows an actual deletion the reader did not see previewed first.
   The partial-download captures (status PARTIAL rows, du ● marker,
   clean-partial dry run, the resume run) were produced by deliberately
   interrupting a `--preset safetensors` download of the ungated mirror
   NousResearch/Meta-Llama-3.1-8B (chosen so no license is needed to
   reproduce) and re-capturing as it resumed. The clean-partial dry run is
   framed as a counterfactual ("had you abandoned it") because in the real
   session the download was resumed, not cleaned — keep that framing so the
   narrative timeline stays honest.
4. Output blocks: paste exact output, do not paraphrase. Trim only when a
   block runs longer than ~20 lines and the trimmed lines are
   representative repetition (note the trim with `…`).
5. Length budget: under 300 lines total, including embedded outputs.
   Update the word count + reading-time line at the top whenever the
   prose changes non-trivially (250 wpm).
6. Word count = total words in this file excluding code blocks and
   HTML comments. Reading time = word count / 250, rounded to the
   nearest minute, minimum 1.
-->

The second tutorial in the v0.10 docs effort. If a step is confusing or an output looks different from yours, please open an issue on [GitHub](https://github.com/PCfVW/hf-fetch-model/issues) — the tutorial improves as real questions land.

## Contents

- [Why the cache fills silently](#why-the-cache-fills-silently)
- [The 30-second answer](#the-30-second-answer)
- [Seeing: the `du` family](#seeing-the-du-family)
- [Diagnosing one repo: `status`](#diagnosing-one-repo-status)
- [Acting: `delete`, `clean-partial`, `gc`](#acting-delete-clean-partial-gc)
- [Trust, but verify: `cache verify`](#trust-but-verify-cache-verify)
- [Scripting: `cache path`](#scripting-cache-path)
- [What you've learned](#what-youve-learned)

## Why the cache fills silently

Every `hf-fm <repo>` — and every Python `huggingface_hub` download, since both tools share the same cache — adds a `models--org--name/` directory under `~/.cache/huggingface/hub/` that persists until you delete it. There is no expiry, no size limit, no garbage collection. You discover the problem the day the disk is full, facing dozens of directories whose `blobs/` + `refs/` + `snapshots/` internals were never meant to be cleaned by hand: delete the wrong piece and you orphan refs or break the snapshot symlinks for a model you still use.

hf-fm splits the job into two verbs, and that split is the whole mental model of this tutorial: **`du` for seeing, `cache` for acting** — the same separation Docker draws between `docker system df` and `docker system prune`. Everything in the `du` family is read-only; everything under `cache` that deletes asks first, or accepts `--dry-run`.

## The 30-second answer

Two commands. The first shows where the bytes are; the second previews what an age-based sweep would reclaim. Nothing is deleted.

```sh
hf-fm du
```

```
Cache: C:\Users\Eric JACOPIN\.cache\huggingface\hub

    #        SIZE  REPO                                                        FILES
    1  159.05 GiB  mntss/clt-gemma-2-2b-2.5M                                      53
    2   40.62 GiB  bluelightai/clt-qwen3-1.7b-base-20k                            88
    3   26.51 GiB  mntss/clt-gemma-2-2b-426k                                      53
    4   23.68 GiB  bluelightai/clt-qwen3-0.6b-base-20k                            88
    5   19.52 GiB  google/gemma-2-2b                                              37
    …
   65     2.3 KiB  chanind/sae-gemma-2-2b-standard                                 1
   66       567 B  EleutherAI/pythia-70m                                           1
  ────────────────────────────────────────────────────────────────────────────────────────
  592.39 GiB  total (66 repos, 859 files)
```

```sh
hf-fm cache gc --older-than 90 --dry-run
```

```
Cache: C:\Users\Eric JACOPIN\.cache\huggingface\hub

Will remove:
  Qwen/Qwen2.5-Coder-7B-Instruct         14.19 GiB  4 months ago
  google/codegemma-7b-it                 15.92 GiB  4 months ago
  codellama/CodeLlama-7b-hf              12.55 GiB  4 months ago
  …
  mntss/clt-gemma-2-2b-2.5M             159.05 GiB  3 months ago
  allenai/OLMo-1B-hf                      4.39 GiB  3 months ago

Cache: 592.39 GiB → 292.52 GiB (free 299.87 GiB)
```

That preview says: everything untouched for 90 days, removed in one stroke, frees 299.87 GiB — half the cache. When the list looks right, re-run without `--dry-run` and answer the prompt. The rest of the tutorial is what to check before you trust that list.

## Seeing: the `du` family

`du` is one command with progressive flags. The `#` column is not decoration: every index it prints is accepted wherever a repo ID is — `du 23`, `cache delete 23`, `cache verify 23` — so you never type `models--microsoft--Phi-3.5-mini-instruct` or even `microsoft/Phi-3.5-mini-instruct` by hand. A `●` marker after a row flags a repo with an interrupted download. Start one, stop it, and that repo's row picks up the marker:

```
   50    1.09 GiB  NousResearch/Meta-Llama-3.1-8B                                  3  ●
  ● = partial downloads
```

(That partial is a download we interrupt on purpose in the next section, which is why this repo is absent from the full listings above — they predate it. Its size reads as only 1.09 GiB and 3 files because `du` counts *finalized* files; the in-flight shards still live in `blobs/` as `.chunked.part` and surface as the `●`, not as counted size. `status`, below, is what shows their true progress.)

`--age` adds the question GC will ask — *when did I last touch this?*

```sh
hf-fm du --age
```

```
    #        SIZE  REPO                                                        FILES  AGE
    1  159.05 GiB  mntss/clt-gemma-2-2b-2.5M                                      53  3 months ago
    2   40.62 GiB  bluelightai/clt-qwen3-1.7b-base-20k                            88  19 days ago
    3   26.51 GiB  mntss/clt-gemma-2-2b-426k                                      53  3 months ago
    4   23.68 GiB  bluelightai/clt-qwen3-0.6b-base-20k                            88  15 days ago
    5   19.52 GiB  google/gemma-2-2b                                              37  15 days ago
    …
```

Row 1 is the classic case this tutorial exists for: a 159 GiB sparse-crosscoder set used heavily three months ago and never since. Row 2 looks similar in size but was touched 19 days ago — an age-based sweep keeps it.

Drill into one repo by its index:

```sh
hf-fm du 23
```

```
  microsoft/Phi-3.5-mini-instruct:

    #        SIZE  FILE
    1    4.63 GiB  model-00001-of-00002.safetensors
    2    2.49 GiB  model-00002-of-00002.safetensors
    3    1.76 MiB  tokenizer.json
    …
   20       195 B  generation_config.json
  ─────────────────────────────────────────────────
    7.12 GiB  total (20 files)
```

And `du --tree` renders the whole cache as one structural view — repos as branches, files as leaves sorted by size — which is where pathological layouts jump out (here, per-layer decoder shards stepping down from 10.97 GiB):

```sh
hf-fm du --tree
```

```
  ├── mntss/clt-gemma-2-2b-2.5M    .  .  .  .  .  .  .  .  .  .  .  .  .159.05 GiB  (53 files)
  │   ├── W_dec_0.safetensors                                            10.97 GiB
  │   ├── W_dec_1.safetensors                                            10.55 GiB
  │   ├── W_dec_2.safetensors                                            10.13 GiB
  …
```

## Diagnosing one repo: `status`

`du` tells you a repo's weight; `status` tells you its health — it fetches the remote file list and classifies every file:

```sh
hf-fm status microsoft/Phi-3.5-mini-instruct
```

```
microsoft/Phi-3.5-mini-instruct (main @ 2fe192450127e6a83f7441aef6e3ca586c338b77)
Cache: C:\Users\Eric JACOPIN\.cache\huggingface\hub\models--microsoft--Phi-3.5-mini-instruct

  .gitattributes                      1.5 KiB  complete
  CODE_OF_CONDUCT.md                    453 B  complete
  …
  model-00001-of-00002.safetensors   4.63 GiB  complete
  model-00002-of-00002.safetensors   2.49 GiB  complete
  …
  tokenizer_config.json               3.9 KiB  complete

20/20 complete, 0 partial, 0 missing
```

Four states can appear, and the all-green run above shows only one. Start a download and interrupt it — close the laptop, drop the connection, hit Ctrl-C — then ask `status` what survived. Here is an ungated Llama-3.1-8B mirror, fetched with `--preset safetensors` and interrupted partway through its four shards:

```sh
hf-fm status NousResearch/Meta-Llama-3.1-8B --preset safetensors
```

```
NousResearch/Meta-Llama-3.1-8B (main @ 1f47e50cdbe801ad8a5174156ec3a0655108fb9f)
Cache: C:\Users\Eric JACOPIN\.cache\huggingface\hub\models--NousResearch--Meta-Llama-3.1-8B

  .gitattributes                      1.5 KiB  excluded
  LICENSE                             7.4 KiB  excluded
  config.json                           826 B  complete
  generation_config.json                185 B  complete
  model-00001-of-00004.safetensors   1.38 GiB / 4.63 GiB    PARTIAL
  model-00002-of-00004.safetensors   1.27 GiB / 4.66 GiB    PARTIAL
  model-00003-of-00004.safetensors   1.36 GiB / 4.58 GiB    PARTIAL
  model-00004-of-00004.safetensors   1.09 GiB  complete
  model.safetensors.index.json       23.4 KiB  MISSING
  original/consolidated.00.pth      14.96 GiB  excluded
  …
  tokenizer.json                     8.66 MiB  MISSING

3/17 complete, 3 partial, 5 missing, 6 excluded
```

All four at once. `complete` and `MISSING` mean what they say. Each `PARTIAL` row shows that file's *own* downloaded bytes against its full size — `1.38 GiB / 4.63 GiB`, a different figure per row, not one repo-level number stamped on every line (a v0.10.5 fix: the count is read from the resume sidecar, so it is true progress, not the preallocated blob's reserved size). A mid-download `status` is therefore a live per-file progress report. `excluded` marks files the `--preset` deliberately skipped — here the `original/consolidated.00.pth` PyTorch checkpoint and the docs — telling "I chose not to fetch this" apart from "this download is incomplete". (Drop `--preset` and those same files report `MISSING`; the preset is what lets `status` know the difference.)

One thing to unlearn: a `.chunked.part` file is **not** garbage. It is resume state — run the same download again and it continues from those exact bytes instead of restarting (v0.9.8):

```sh
hf-fm NousResearch/Meta-Llama-3.1-8B --preset safetensors
```

```
  Disk: 13.88 GiB to fetch, 2088.86 GiB available (2074.98 GiB after download)
[hf-fm] model-00003-of-00004.safetensors: 1.36 GiB/4.58 GiB (29%)
[hf-fm] model-00001-of-00004.safetensors: 1.38 GiB/4.63 GiB (29%)
[hf-fm] model-00002-of-00004.safetensors: 1.27 GiB/4.66 GiB (27%)
  …
Downloaded to: …/models--NousResearch--Meta-Llama-3.1-8B/snapshots/1f47e50c…
  14.97 GiB in 205.7s (74.5 MiB/s)
```

The shards pick up at 27–29% — exactly where `status` left them above — not 0%. The interrupted bytes counted. (The final `14.97 GiB` line is hf-fm reporting the file's full size, not the bytes fetched this run; the ~4 GiB already on disk is why it finished as fast as it did.) Clean partials only when you have decided *not* to resume.

## Acting: `delete`, `clean-partial`, `gc`

Three commands, in increasing blast radius. All preview before acting.

**One repo:** `cache delete` takes a repo ID or a `du` index, shows what it is about to remove, and asks:

```sh
hf-fm cache delete julien-c/dummy-unknown
```

```
  julien-c/dummy-unknown  (276.5 KiB, 8 files)
  Delete? [y/N] n
  Aborted.
```

It removes the repo's entire `models--…` directory — blobs, refs, snapshots — so nothing is left half-alive. Pass `--yes` to skip the prompt in scripts.

**Leftover partials:** suppose you had *abandoned* that interrupted Llama download instead of resuming it. `cache clean-partial` sweeps the leftover `.chunked.part` files (and their resume sidecars) across the whole cache, with the same `--dry-run` / prompt / `--yes` controls. Run the dry run first:

```sh
hf-fm cache clean-partial --dry-run
```

```
Cache: C:\Users\Eric JACOPIN\.cache\huggingface\hub

Would remove 3 files (13.87 GiB):
  NousResearch/Meta-Llama-3.1-8B: c28b25e7…chunked.part  (4.66 GiB)
  NousResearch/Meta-Llama-3.1-8B: d8e9504d…chunked.part  (4.58 GiB)
  NousResearch/Meta-Llama-3.1-8B: f8b9704a…chunked.part  (4.63 GiB)
```

The reclaimed figure is real disk: a chunked download preallocates each temp blob at its *full* size, so a half-finished 4.66 GiB shard occupies 4.66 GiB on disk even though `status` (reading the sidecar) correctly reports only the bytes truly transferred. Pass a repo ID to scope the sweep to one model (`cache clean-partial NousResearch/Meta-Llama-3.1-8B`); omit it to clean everything. Either way, heed the lesson above — a partial is resume state, so clean only the downloads you have given up on. An empty cache simply reports `No partial downloads found.`

**The sweep:** `cache gc` evicts by either of the two budgets you actually think in. `--older-than <DAYS>` answers *"what haven't I used lately?"* (the 30-second answer above). `--max-size <SIZE>` answers *"how much space can I spare?"* — it removes oldest-first until the cache fits the target. Both combine, and `--except` protects repos you want kept regardless of age:

```sh
hf-fm cache gc --max-size 400GiB --except bluelightai/clt-qwen3-1.7b-base-20k --dry-run
```

```
Will remove:
  Qwen/Qwen2.5-Coder-7B-Instruct   14.19 GiB  4 months ago
  google/codegemma-7b-it           15.92 GiB  4 months ago
  …
  mntss/clt-gemma-2-2b-2.5M       159.05 GiB  3 months ago

Protected by --except:
  bluelightai/clt-qwen3-1.7b-base-20k
```

Two caveats worth knowing before you trust GC. "Last accessed" is approximated by the newest modification time among the repo's snapshot files — the HF cache layout does not record true access times, so a repo you *read* daily but never re-download looks old; protect it with `--except`. And repos with a partial download modified within the last hour are skipped automatically, so GC never races an `hf-fm` download running in another shell.

## Trust, but verify: `cache verify`

After a crash, a flaky disk, or before archiving a cache you plan to rely on offline, re-check the bytes against HuggingFace's LFS metadata (needs network for the expected hashes — the files themselves are read locally):

```sh
hf-fm cache verify bartowski/Qwen2.5-0.5B-Instruct-GGUF
```

```
bartowski/Qwen2.5-0.5B-Instruct-GGUF (main @ 41ba88dbac95fed2528c92514c131d73eb5a174b)
Cache: C:\Users\Eric JACOPIN\.cache\huggingface\hub

  — .gitattributes                         3.1 KiB  no LFS hash
  ✓ Qwen2.5-0.5B-Instruct-IQ2_M.gguf    313.37 MiB  SHA256 OK
  ! Qwen2.5-0.5B-Instruct-IQ3_M.gguf    326.87 MiB  MISSING
  …

27 files: 1 SHA256 OK, 0 mismatch, 3 skipped, 23 missing
```

Read the legend, not the row count: `✓` is a re-hashed match, `✗ SHA256 MISMATCH` is real corruption (delete the repo and re-download; `verify` also exits non-zero), `—` marks small non-LFS files that have no upstream hash to check, and `!` `MISSING` simply means never downloaded — entirely normal in a quant repo like this one, where you fetched one variant out of 27 files. The one number that should always be 0 is `mismatch`.

## Scripting: `cache path`

Where visibility meets the rest of your toolchain: `cache path` prints a repo's snapshot directory and exits non-zero if the repo is not cached.

```sh
hf-fm cache path google/gemma-2-2b-it
```

```
C:\Users\Eric JACOPIN\.cache\huggingface\hub\models--google--gemma-2-2b-it\snapshots\299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8
```

```sh
cd $(hf-fm cache path google/gemma-2-2b-it)        # bash / zsh
cd (hf-fm cache path google/gemma-2-2b-it)         # PowerShell
```

## What you've learned

| Question | Command |
|----------|---------|
| Where are the bytes? | `du`, `du --age`, `du --tree` |
| What's inside repo #N? | `du <N>` |
| Is this repo healthy? | `status <repo>`, `cache verify <repo\|N>` |
| Remove one repo | `cache delete <repo\|N>` |
| Remove interrupted-download leftovers | `cache clean-partial` |
| Reclaim space in bulk | `cache gc --older-than D` / `--max-size S` (+ `--except`, always `--dry-run` first) |
| Hand the path to another tool | `cache path <repo>` |

One model in one sentence: `du` sees, `cache` acts — and every acting command previews or prompts before a byte is deleted.

For the other half of cache discipline — deciding what is worth downloading *before* it lands on your disk — see the companion tutorial, [Inspect before you download](inspect-before-downloading.md). The [FAQ's cache section](../FAQ.md#cache-location-and-management) and the [CLI reference](../cli-reference.md#cache-commands) cover every flag mentioned here.
