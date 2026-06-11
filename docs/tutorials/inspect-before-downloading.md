# Inspect before you download

*Read tensor metadata over HTTP Range — no weight data downloaded — and decide whether the model is worth the bandwidth.*

*~1,720 words · about 7 min read*

<!-- Last updated: 2026-06-11, hf-fm v0.10.5 -->

<!--
STYLE CONVENTIONS for editing this tutorial — keep growth consistent.

1. Tone: match the FAQ. Conversational, address the reader as "you", short
   paragraphs over bullet lists where prose works.
2. Pinning: every command in the tutorial uses
   `--revision 1529be60e35a55682cb6029e3840b53d9a57e6d0` against
   `zed-industries/zeta-2`. The captured outputs are anchored to that SHA.
   If you re-run without `--revision`, expect drift if Zed publishes new
   commits to `main`.
3. Output blocks: paste exact output, do not paraphrase. Trim only when a
   block runs longer than ~20 lines and the trimmed lines are
   representative repetition (note the trim with `…`).
4. §6's verdict paragraph points at `--check-gpu --context N` (v0.10.4,
   KV-cache budgeting) and links the FAQ for the formula + limitations;
   keep that pointer and do not re-expand the old manual "30–50% on top"
   math. The output block stays on `--check-gpu --dtypes` (no `--context`)
   so its weights-only note is a faithful capture.
5. Length budget: under 300 lines total, including embedded outputs.
   Update the word count + reading-time line at the top whenever the
   prose changes non-trivially (250 wpm).
6. Word count = total words in this file excluding code blocks and
   HTML comments. Reading time = word count / 250, rounded to the nearest
   minute, minimum 1.
7. --check-gpu output (added v0.10.1): the GPU 0 line is the maintainer's
   RTX 5060 Ti — `free` / `used` figures will drift run-to-run as the
   desktop's other processes change. Re-capture if rerunning the tutorial
   end-to-end; the verdict (✗ short by ~1–2 GiB) should remain stable
   regardless of the moment.
-->

A new tutorial in the v0.10.0 docs effort. If you find a step confusing or an output that looks different from yours, please open an issue on [GitHub](https://github.com/PCfVW/hf-fetch-model/issues) — the tutorial improves as real questions land.

## Contents

- [Why inspect first?](#why-inspect-first)
- [The 30-second answer](#the-30-second-answer)
- [Discovery: `--list`](#discovery---list)
- [Sizing: `--dtypes`](#sizing---dtypes)
- [Architecture: `--tree`](#architecture---tree)
- [Targeting: `--filter` and `--limit`](#targeting---filter-and---limit)
- [The verdict — and what to do when it doesn't fit](#the-verdict--and-what-to-do-when-it-doesnt-fit)
- [What you've learned](#what-youve-learned)

## Why inspect first?

A modern code-completion model can be 15 GiB on disk. On a 50 Mbps connection that's 40 minutes. If at the end of those 40 minutes you discover the model has a vocab too big to fit your tokenizer, an architecture that doesn't match the inference framework you planned to use, or a weight precision your GPU can't run — you have just spent 40 minutes to learn nothing useful.

`hf-fm inspect` reads a model's tensor metadata over HTTP Range requests. The header bytes are typically tens of kilobytes; the weight bytes are not fetched. You will know more about the model than the model card tells you, and you will have downloaded under a megabyte to do it.

The running example throughout is [`zed-industries/zeta-2`](https://huggingface.co/zed-industries/zeta-2) — Zed Industries' code-completion model, a real 4-shard Llama-class repo at the time of writing.

## The 30-second answer

Here is the entire payoff in one command:

```sh
hf-fm inspect zed-industries/zeta-2 \
  --revision 1529be60e35a55682cb6029e3840b53d9a57e6d0 \
  --tree
```

Output:

```
  Repo:   zed-industries/zeta-2
  Source: aggregated across 4 shards

  ├── lm_head.weight  BF16  [155136, 4096]  1.18 GiB
  └── model.
      ├── embed_tokens.weight  BF16  [155136, 4096]  1.18 GiB
      ├── layers.[0..31].   (×32)
      │   ├── input_layernorm.weight           BF16  [4096]  8.0 KiB
      │   ├── mlp.
      │   │   ├── down_proj.weight  BF16  [4096, 14336]  112.00 MiB
      │   │   ├── gate_proj.weight  BF16  [14336, 4096]  112.00 MiB
      │   │   └── up_proj.weight    BF16  [14336, 4096]  112.00 MiB
      │   ├── post_attention_layernorm.weight  BF16  [4096]  8.0 KiB
      │   └── self_attn.
      │       ├── k_proj.weight  BF16  [1024, 4096]  8.00 MiB
      │       ├── o_proj.weight  BF16  [4096, 4096]  32.00 MiB
      │       ├── q_proj.weight  BF16  [4096, 4096]  32.00 MiB
      │       └── v_proj.weight  BF16  [1024, 4096]  8.00 MiB
      └── norm.weight          BF16  [4096]  8.0 KiB
  291 tensors, 8.25B params
```

In one screen you can see the model has 32 transformer layers, an unusually large 155,136-token vocabulary, GQA at a 32:8 ratio (visible from `q_proj` being 4096-wide and `k_proj` being 1024-wide), an untied `lm_head` (separate 1.18 GiB tensor), and 8.25 billion parameters in BF16.

The rest of this tutorial unpacks how each piece of that picture is produced. Seven commands, no weight bytes.

## Discovery: `--list`

Start here when you walk up to a repo you don't know:

```sh
hf-fm inspect zed-industries/zeta-2 --list
```

```
Repo: zed-industries/zeta-2
Rev:  1529be60e35a55682cb6029e3840b53d9a57e6d0 (main)

#  File                                  Size
-  --------------------------------  --------
1  model-00001-of-00004.safetensors  4.62 GiB
2  model-00002-of-00004.safetensors  4.58 GiB
3  model-00003-of-00004.safetensors  4.66 GiB
4  model-00004-of-00004.safetensors  1.51 GiB

4 files, 15.37 GiB total

Tip: run `hf-fm inspect zed-industries/zeta-2 <n>` to inspect file #n.
     Pass `--revision 1529be60e35a55682cb6029e3840b53d9a57e6d0` on both sides to lock against this view.
```

Three things to read off this output. The repo has four `.safetensors` shards totaling 15.37 GiB. The `Rev:` line is the commit SHA that `main` resolved to when you ran the command. And the footer offers shorter forms — `inspect <repo> 1` would inspect the first shard by index instead of by name.

### Sidebar: pin your revision

Every command in this tutorial passes `--revision 1529be60e35a55682cb6029e3840b53d9a57e6d0`. That commit hash is what was published to `main` when the tutorial was written. By pinning, the embedded outputs match exactly what you see locally — even if Zed Industries pushes new commits to `main` between now and your read.

For your own investigations, the `Rev:` line above is where to find the SHA to pin against. Drop it onto every follow-up `inspect`, `diff`, or `download-file` command and your notes stay reproducible. A short prefix usually works in practice; the full 40-character SHA is what the HF API formally guarantees, so the tutorial uses the full one.

## Sizing: `--dtypes`

Now the first big question: how much does this model weigh, in what precision?

```sh
hf-fm inspect zed-industries/zeta-2 \
  --revision 1529be60e35a55682cb6029e3840b53d9a57e6d0 \
  --dtypes
```

```
  Repo:   zed-industries/zeta-2
  Source: aggregated across 4 shards

  Dtype  Tensors       Params       Size
  BF16       291        8.25B  15.37 GiB
  ─────────────────────────────────────────
  291 tensors, 8.25B params
```

One row, three numbers. The model is **8.25 billion parameters**, all in **BF16**, totaling **15.37 GiB on disk**. No mixed precision, no FP8 quantization, no INT4 — just BF16 throughout.

The `Source: aggregated across 4 shards` line is worth noticing. To produce that one-row histogram, hf-fm fetched the header of every shard (~tens of kilobytes each via HTTP Range) and rolled up the dtype counts. Four small range requests instead of the 15.37 GiB download.

If the model had been mixed precision — say FP8 attention with BF16 norms — you would see two rows, with each dtype's tensor count, parameter count, and bytes. That breakdown is what tells you what hardware can run the model and what quantization budget is left to play with.

## Architecture: `--tree`

The dtype histogram tells you *how big*. The tree tells you *what shape*.

```sh
hf-fm inspect zed-industries/zeta-2 \
  --revision 1529be60e35a55682cb6029e3840b53d9a57e6d0 \
  --tree
```

The output is the same one we showed in §2. Take a second look at it now with the rest of this tutorial in mind.

The `layers.[0..31]   (×32)` collapsing is what makes a Llama-class model legible at a glance. Each of the 32 transformer layers has the same nine-tensor topology — input layernorm, MLP triple (down/gate/up), post-attention layernorm, attention quadruple (q/k/v/o) — so hf-fm shows the template once and reports the multiplicity. Without that collapsing the tree would be 288 lines of repetition.

Two architectural details worth catching:

- **GQA at 32:8 ratio.** The query and output projections are square at `[4096, 4096]`, but the key and value projections are narrower at `[1024, 4096]`. That 4× ratio means 32 query heads share 8 KV heads, which is the standard Llama-3 8B / Mistral 7B shape. KV cache is 4× smaller than naive multi-head attention — directly relevant to context length you can support on a given GPU.

- **Untied `lm_head`.** `lm_head.weight` shows up as a separate tensor at `[155136, 4096]`, distinct from `model.embed_tokens.weight`. Some smaller Llama variants tie these two so they share weights; zeta-2 does not. That single decision adds 1.18 GiB to the model and gives the output projection independent capacity.

The 155,136-token vocabulary is also unusual. Llama-3's tokenizer is 128,256 entries; Qwen 2.5's is 151,936. The size and the tokenizer family it suggests (Qwen-derived) tell you something about training data and downstream compatibility before you have read a single line of the model card.

## Targeting: `--filter` and `--limit`

Sometimes you want a slice of the picture, not the whole thing. `--filter` matches against tensor names and composes with every other view; `--limit` truncates the flat tensor list to the first N entries.

For example, "what does the output head cost in this model?":

```sh
hf-fm inspect zed-industries/zeta-2 \
  --revision 1529be60e35a55682cb6029e3840b53d9a57e6d0 \
  --filter lm_head --tree
```

```
  Repo:   zed-industries/zeta-2
  Source: aggregated across 4 shards

  └── lm_head.weight  BF16  [155136, 4096]  1.18 GiB
  Showing 1 of 291 tensors matching filter "lm_head".
  Param counts: 635.4M matching filter, 8.25B total.
```

The footer's `Showing 1 of 291 tensors matching filter "lm_head"` (635.4M matching out of 8.25B total) is the answer: the head alone is 635 million parameters, 1.18 GiB. If you are sketching a fine-tuning budget, that's the cost of replacing the head and freezing the rest.

Or "show me the first five tensors with their source shard":

```sh
hf-fm inspect zed-industries/zeta-2 \
  --revision 1529be60e35a55682cb6029e3840b53d9a57e6d0 \
  --limit 5
```

```
  Repo:   zed-industries/zeta-2
  Source: aggregated across 4 shards

  Tensor                                Dtype    Shape                Size     Params  Shard
  model.embed_tokens.weight             BF16     [155136, 4096]   1.18 GiB     635.4M  model-00001-of-00004.safetensors
  model.layers.0.input_layernorm.weight BF16     [4096]            8.0 KiB       4.1K  model-00001-of-00004.safetensors
  model.layers.0.mlp.down_proj.weight   BF16     [4096, 14336]  112.00 MiB      58.7M  model-00001-of-00004.safetensors
  model.layers.0.mlp.gate_proj.weight   BF16     [14336, 4096]  112.00 MiB      58.7M  model-00001-of-00004.safetensors
  model.layers.0.mlp.up_proj.weight     BF16     [14336, 4096]  112.00 MiB      58.7M  model-00001-of-00004.safetensors
  ─────────────────────────────────────────────────────────────────────────────────────────────
  Showing 5 of 291 tensors (limit: 5).
  Param counts: 811.6M shown, 8.25B total.
```

The `Shard` column is hf-fm reminding you that shard 1 holds the embedding plus the first set of layer weights. That is occasionally useful when debugging: if a download failed for a specific shard, you know which tensors are at risk.

One last `inspect` view before the verdict section. `--check-gpu` puts the model's weight bytes next to the device's free VRAM and tells you whether one fits in the other — the fit math done for you against device 0:

```sh
hf-fm inspect zed-industries/zeta-2 \
  --revision 1529be60e35a55682cb6029e3840b53d9a57e6d0 \
  --check-gpu --dtypes
```

```
  Repo:   zed-industries/zeta-2
  Source: aggregated across 4 shards

  Dtype  Tensors       Params       Size
  BF16       291        8.25B  15.37 GiB
  ─────────────────────────────────────────
  291 tensors, 8.25B params

  Model weights:  15.37 GiB  (BF16, 8.25B params)
  GPU 0:          NVIDIA GeForce RTX 5060 Ti — 15.93 GiB VRAM
                  free: 13.60 GiB, used: 2.33 GiB
  Fit:            ✗ short by 1.77 GiB for the weights alone

  Note: reports weights only. Large-context inference typically needs ~1.3–1.5×
  weight size for KV cache and activations.
```

The verdict is short and clear: the weights are 1.77 GiB larger than what is actually free on this RTX 5060 Ti right now — and that is *before* the KV cache. Add `--context N` (v0.10.4) to fold the KV cache in at a chosen sequence length and get a real `weights + KV` verdict; it reads the architecture from `config.json`, so GQA, sliding-window, and hybrid Mamba models are all handled (see the [FAQ](../FAQ.md#how-do-i-know-if-a-model-fits-on-my-gpu) for the formula and its limitations). The next section is the practical pivot to a quant that fits.

## The verdict — and what to do when it doesn't fit

`--check-gpu` already gave you the answer: zeta-2's 15.37 GiB of BF16 weights won't fit on a 16 GB card, and that's before any inference state. The community pivot is to quantize — search for GGUF variants:

```sh
hf-fm search zeta-2 --tag gguf
```

```
Models matching "zeta-2" (by downloads):

  hf-fm bartowski/zed-industries_zeta-2-GGUF                                         (5,963 downloads)  [transformers, text-generation]
  hf-fm bluevoid-pl/zeta2-GUFF                                                       (3,155 downloads)  [transformers]
  hf-fm zetasepic/Qwen2.5-72B-Instruct-abliterated-GGUF                              (736 downloads)  [text-generation]
  …
  hf-fm mradermacher/Zeta-2-i1-GGUF                                                  (103 downloads)  [transformers]
  hf-fm mradermacher/Zeta-2-GGUF                                                     (92 downloads)  [transformers]
  …
```

Bartowski tops the list with ~6× the runner-up's downloads — a known quantization specialist on HF, the canonical pick when bartowski has done a model.

Note: search by tag returns prefix-matches too — confirm the upstream org before downloading a community quant. The `zetasepic` row above is an abliterated Qwen 72B, unrelated to `zed-industries` — a typical false-positive pattern, and the elided rows between it and `mradermacher` are more of the same.

List the quant ladder bartowski published:

```sh
hf-fm list-files bartowski/zed-industries_zeta-2-GGUF --no-checksum
```

The full output runs ~25 lines; the relevant excerpt for our 16 GB GPU:

```
  zed-industries_zeta-2-Q4_K_M.gguf     4.87 GiB
  zed-industries_zeta-2-Q5_K_M.gguf     5.61 GiB
  zed-industries_zeta-2-Q6_K.gguf       6.54 GiB
  zed-industries_zeta-2-Q8_0.gguf       8.77 GiB
  …
```

Mapping each variant against ~14 GiB of usable free VRAM on a 5060 Ti:

| Variant | Size | Headroom for KV + runtime |
|---------|------|---------------------------|
| **Q4_K_M** | 4.87 GiB | **~9 GiB — sweet spot** |
| Q5_K_M | 5.61 GiB | ~8 GiB |
| Q6_K | 6.54 GiB | ~7 GiB |
| Q8_0 | 8.77 GiB | ~5 GiB |

Q4_K_M is the typical recommendation for llama.cpp users — quality, size and speed all in a comfortable place — and it leaves nine gigabytes for KV cache, activations, and CUDA workspace. That's enough for very long contexts.

The closing command, the only download in this whole tutorial:

```sh
hf-fm download-file bartowski/zed-industries_zeta-2-GGUF "*Q4_K_M*"
```

You went from "15 GiB I can't use" to "5 GiB that fits with room to spare" without downloading the wrong thing first.

## What you've learned

- **`inspect --list`** discovers what's in a repo, in seconds, and shows the commit SHA you should pin to.
- **`inspect --dtypes`** answers how big and in what precision, aggregated across every shard.
- **`inspect --tree`** answers what shape — architecture, layer count, GQA ratio, tied vs untied heads, vocabulary size.
- **`inspect --filter` / `--limit`** answer "what does *this part* cost?" before you commit to anything.
- **`inspect --check-gpu`** turns weight bytes vs. free VRAM into a one-line ✓ / ✗ verdict — add **`--context N`** to fold in the KV cache and judge `weights + KV` at a real context length.
- **`search --tag` + `list-files`** find community quantizations when the headline repo doesn't fit.
- **`--revision <sha>`** pins every command above to a specific commit so notes stay reproducible.

Total bytes downloaded to learn all of that: less than a megabyte. Every command above is also valid input to library callers — see [`examples/candle_inspect.rs`](../../examples/candle_inspect.rs) for the embeddable equivalent.

The companion path — **inspect after you download** — already exists today via `--cached` on `.gguf` files (since v0.10.2) and `.npz` / `.pth` files plus quantization detection on cached safetensors (since v0.10.3).

For details on every flag in `inspect`, see the [CLI reference](../cli-reference.md). For common follow-up questions, the [FAQ](../FAQ.md) covers gating, cache layout, and the quirks of partial downloads.

And once the models you *did* download start crowding your disk, the companion tutorial [Clean up before your disk fills](clean-up-before-your-disk-fills.md) covers the other end of the lifecycle: `du`, `status`, and the `cache` commands.
