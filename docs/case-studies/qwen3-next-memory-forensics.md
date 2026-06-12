# Reconstructing an OOM from a crash log

*How `inspect --dtypes` gave ground-truth on-disk numbers for a model the reporter never named — and why the analysis stayed in a drawer.*

**Source:** candle [#3530](https://github.com/huggingface/candle/issues/3530) · archive: [candle-3530-p1](../issues/candle-3530-p1.md) (posted), [p2](../issues/candle-3530-p2.md) (draft)

---

## The symptom

An experienced ML engineer reported candle using **81 GB of 119 GB** on an
NVIDIA Spark (Blackwell SM121) while loading "about the size of an 80B NVFP4
model," then failing on the KV cache. Their working hypothesis was that candle
was **double-mapping** the weights — `mmap`-then-copy, counted twice. The whole
question turned on one unknown number: *what does the model actually weigh on
disk?* If ~40 GiB, 81 GB used ≈ 2× → double-mapping. If ~80 GiB, no
double-mapping → the pressure is real and lives somewhere else.

## The diagnostic path

`inspect --dtypes` reads safetensors headers over HTTP Range — no weight data
downloaded — and bins every tensor by dtype. That is exactly the ground truth
the hypothesis needed. The catch: **the reporter never shared the repo.** They
replied to our first comment ([p1](../issues/candle-3530-p1.md), which laid out
the two-branch logic and asked for the repo) without naming it.

So the second step ([p2](../issues/candle-3530-p2.md)) was to *fingerprint* the
model from the crash log. Four constraints in their own output narrowed the
field hard:

- **"Hybrid Mamba Allocation: 28 slot(s) … 36 linear-attention layer(s)"** —
  a hybrid attention + state-space architecture; very few public families ship
  this.
- **"NVFP4"** — NVIDIA's FP4 quantization scheme.
- **"80B"** — the parameter scale.
- **Spark / SM121** — the hardware NVIDIA's own NVFP4 builds target.

Those four intersect at a single public candidate:
`nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4`. Running the tool against it:

```
hf-fm inspect nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 --dtypes
```
```
  Dtype    Tensors       Params       Size
  F32       147864       147.9K  577.6 KiB
  F8_E4M3    73920        4.87B   4.53 GiB
  U8         73920       38.93B  36.26 GiB
  BF16        2024        3.46B   6.45 GiB
  ─────────────────────────────────────────
  297728 tensors, 47.26B params
```

The histogram corrected the reporter's "FP4 = ½ × param count" shorthand. The
**U8** row (36.26 GiB) *is* the packed FP4 weights — that part of the heuristic
holds. But NVFP4 carries metadata the shorthand omits: an **F8_E4M3** row of
per-block scaling factors at 4.53 GiB (note the tensor count, 73,920, matches
U8 *exactly* — the 1:1 block-scale structure), and a **BF16** row of norms and
embeddings at 6.45 GiB (kept full-precision because quantizing them wrecks
perplexity for a few percent of bytes). Real on-disk total: **47.24 GiB**, not
~40 — about **+30% on top of the FP4 bulk**, a general NVFP4 fact worth
carrying forward.

With that number, the 81 GB reconciles as a single honest accounting — **no
double-mapping required**:

| Component | Size | Source |
|---|---|---|
| Model weights | 47.24 GiB | `inspect --dtypes` above |
| KV cache pool | 18.00 GiB | their vllm-rs log |
| Hybrid Mamba state | 2.06 GiB | their log |
| Runtime / RSS | ~14 GiB | process, libs, allocator |
| **Total** | **~81 GiB** | matches `used: 81` |

The model is mapped once. That reframes the bug from "is candle mis-counting?"
to the thing the reporter had themselves surfaced: *why does the KV allocator
cap at 18 GB while 38 GB of unified memory sits free on SM121?* The fix lives in
the allocator's UMA policy, not the model loader.

## The outcome — honestly

This is the case the maintainer called "mitigated success." p1 was **posted**;
the reporter engaged but **did not share the repo**, which broke the diagnostic
loop. p2 — the forensic reconstruction above, with the identified model,
corrected on-disk math, and memory reconciliation — was **drafted and never
posted**. It rests on a *guessed input*: if the reporter is running a private
build or a different family, the headline numbers shift (the draft flags this
explicitly, distinguishing "wrong variant" — numbers move a few percent — from
"wrong family" — direction holds, magnitudes shift). Posting a confident
multi-number analysis built on a guessed repo is a different risk than posting a
verified one, and the call was to hold it pending the reporter naming the model.
They never did. So the analysis is correct *conditional on the guess*, complete,
and unpublished — diagnostic leverage that never reached the thread.

There is an honest tension here worth naming: fingerprinting trades certainty
for traction. In a thread that would otherwise stall waiting for a reporter who
has moved on, getting concrete numbers on the table can be the difference
between progress and silence — but only if you're willing to caveat the guess
loudly, and only if someone is still listening.

## The transferable workflow

> To test a memory hypothesis ("is it double-mapped? does it fit?"), get the
> **on-disk ground truth first**: `inspect <repo> --dtypes` bins every tensor by
> dtype over HTTP Range, no download — and for quantized models the dtype
> histogram is itself the story (the FP4 weights, the FP8 block scales, the
> full-precision norms). When the reporter omits the repo but leaves a crash
> log, **fingerprint it**: architecture family + quantization scheme + parameter
> scale + target hardware often intersect at a single public model. Then say,
> loudly, that the numbers are conditional on the guess — and invite the one
> correction that makes them exact: the repo line, or just `config.json`.
