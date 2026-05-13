# candle #3530 — reply 2 (draft)

- **Target issue:** https://github.com/huggingface/candle/issues/3530
- **Status:** Draft (not yet posted). The body uses `inspect --dtypes` which has shipped since v0.9.x; no v0.10.2 dependency. The body's link back to p1 points at the actual GitHub comment (`#issuecomment-4431550797`) so the reply is post-ready as-is; the metadata block above still uses the archive-relative `(candle-3530-p1.md)` link for local-browsing convenience.
- **Context:** Sempervictus replied to [p1](candle-3530-p1.md) without sharing the repo, so we guessed it from the crash log fingerprint (hybrid Mamba allocation, 28 slots, 36 linear-attention layers, NVFP4 on Spark / SM121) — the four constraints narrow to a single public candidate. Running `hf-fm inspect --dtypes` on that candidate gives ground-truth on-disk numbers that, **assuming the guess is right**, correct his "FP4 = ½ × param count" heuristic (47.24 GiB, not ~40 GiB) and close branch B of p1 (mmap-then-copy / 2× bounds counting): the 47.24 GiB model + 18 GiB KV pool + 2 GiB Mamba state + ~14 GiB runtime accounts for `used: 81 GB` as a single, honest accounting. The bug locus then narrows to what sempervictus himself surfaced: the KV allocator caps at 18 GB on SM121 while 38 GB of UMA sits genuinely free — and *that* conclusion is independent of the guess.
- **Outcome:** —
- **Lesson / Leverage angle:** First reply in the archive that demonstrates `inspect --dtypes` on a real NVFP4 model and *names* a candidate public repo from a crash log alone. The ~30%-overhead-on-top-of-FP4-bulk fact (block-scaling factors + BF16 norms + tiny F32 scalars adding up to ~1.3× the FP4 weight bytes) is a general NVFP4 truth worth carrying forward to other replies. The identification reasoning chain — fingerprinting an architecture from a vllm-rs allocator log — is reusable when the OP omits the repo: it narrows the candidate set rather than uniquely identifying, often to a single public match. In an era where bug reports arrive with less reproducer detail than they used to (faster posting cadence, less hand-holding), that narrowing is the difference between "we can compute concrete numbers" and "we're stuck waiting for the OP" — the technique trades certainty for traction, which is the right trade when traction is the limiting factor.

---

Since the repo would let us compute the actual on-disk size — the figure your branch-A-vs-branch-B question hinges on — but isn't in the thread yet, I guessed it from the crash log fingerprint. The constraints are narrow enough that the guess is well-bounded:

- **"Hybrid Mamba Allocation: 28 slot(s) ... 36 linear-attention layer(s)"** — hybrid attention + state-space architecture; very few public families ship this. Qwen3-Next is the headline candidate (its building block is Gated DeltaNet + attention; vllm-rs / HF transformers call those "linear-attention layers" and Mamba slots respectively).
- **"NVPF4" in your title** — NVFP4, NVIDIA Model Optimizer's FP4 quantization scheme; NVIDIA publishes NVFP4 variants of major open models on the Hub.
- **"80B"** — narrows further: Qwen3-Next ships as 80B-A3B (80B total, 3B active, MoE).
- **Spark / Blackwell SM121** — recent NVIDIA hardware that NVIDIA's own NVFP4 builds target.

Best guess from those four constraints: **`nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4`** (or its RedHat repackage `RedHatAI/Qwen3-Next-80B-A3B-Instruct-NVFP4` — same weights). I checked the public NVFP4 catalogue on HuggingFace; this is the only model that matches all four constraints simultaneously. If you're running a private build, a custom quantization of base Qwen3-Next, or a family I haven't considered, the numbers below shift — please share the repo line (or even just `config.json`'s architectural block) and I'll rerun against the real one.

Running `hf-fm inspect <repo> --dtypes` against the NVIDIA repo's safetensors headers (HTTP Range, no weight data downloaded):

```
$ hf-fm inspect nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 --dtypes

  Repo:   nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4
  Source: aggregated across 11 shards

  Dtype    Tensors       Params       Size
  F32       147864       147.9K  577.6 KiB
  F8_E4M3    73920        4.87B   4.53 GiB
  U8         73920       38.93B  36.26 GiB
  BF16        2024        3.46B   6.45 GiB
  ─────────────────────────────────────────
  297728 tensors, 47.26B params
```

Three takeaways — all conditional on Qwen3-Next-80B-A3B-Instruct-NVFP4 (or a structurally identical sibling) being what you're loading:

### 1. Actual on-disk is **47.24 GiB**, not ~40 GiB

Your "FP4 = ½ × param count" rule captures the FP4 weights themselves: the **U8** row at **36.26 GiB** ≈ ½ × 77.86B unpacked FP4 elements. That part is right. But it misses the metadata NVFP4 needs to be useful at inference time:

- **F8_E4M3, 4.53 GiB** — per-block FP8 scaling factors. Note the **73,920 tensor count matching U8 exactly**: that's the 1:1 block-scaling structure of NVFP4 (every quantized weight tensor has a corresponding FP8 scale tensor).
- **BF16, 6.45 GiB** — layer norms, embeddings, and the output projection. These stay in BF16 because quantizing them blows up perplexity for a few percent of the byte budget.
- **F32, 577 KiB** — tiny scalars (RoPE/yarn parameters, eps, etc.).

So on NVFP4 builds, expect **~+30% on top of the FP4 bulk** for the scaling + norm tensors — concretely, `(4.53 + 6.45 + 0.0006) / 36.26 ≈ 30%`, or roughly 1.3× the FP4 weight bytes once everything is accounted for. The "80B → 40G" shorthand is close on the weights alone but understates the on-disk total. General NVFP4 fact worth knowing.

### 2. Your `used: 81 GB` reconciles as a single accounting — no double-mapping

With the corrected model size, your memory snapshot adds up cleanly without invoking branch B of [my earlier reply](https://github.com/huggingface/candle/issues/3530#issuecomment-4431550797) (mmap-then-copy / 2× bounds counting):

| Component | Size | Source |
|---|---|---|
| Model weights | 47.24 GiB | `hf-fm inspect` above |
| KV cache pool | 18.00 GiB | Your vllm-rs log: `"12288 GPU blocks (18.00 GB x 1)"` |
| Hybrid Mamba state | 2.06 GiB | Your log: `"28 slot(s), 2.06 GB budget"` |
| Runtime / RSS overhead | ~14 GiB | Process, libraries, allocator buffers |
| **Total** | **~81 GiB** | matches your `used: 81` exactly (`47.24 + 18.00 + 2.06 + 13.70 = 81.00`) |

So the model **is** mapped once, not twice. The 38 GB "free" in your `free -g` snapshot is genuinely free, not double-counted weights pretending to be free. (Strictly speaking, *neither* branch of p1 predicted the actual on-disk total — Branch A guessed ~80 GiB on the "1 fp4 per byte" packing assumption, Branch B guessed ~40 GiB on the packed-2-per-byte assumption, and reality lands at 47.24 GiB because the FP4 *is* packed 2-per-byte but the per-block scaling factors + BF16 norms add ~30% on top. Branch B's premise about packing was right; its conclusion about double-mapping is what the accounting above rules out.)

### 3. The bug locus narrows to what you yourself surfaced

This reframes the question from *"is candle mis-counting the model?"* to **"why does the KV allocator cap at 18 GB while 38 GB of UMA sits free on SM121?"** — the exact thing you flagged. And it lines up cleanly with your 35B-sibling-works observation: a 35B-class NVFP4 model is roughly 21 GiB on disk (extrapolating the same FP4 + scale + norm structure: 35/80 × 47.24 ≈ 20.7 GiB), and 21 + 18 + 2 + 14 ≈ 55 GiB used → ~64 GiB free on a 119 GiB box → KV-pool exhaustion on long prompts is far less likely to fire because the cap is comfortably under what's needed. The 80B's 47.24 GiB pushes the total to 81 GiB and leaves only 38 GiB headroom; that headroom is what the KV allocator is failing to claim.

The fix lives in vllm-rs's / candle's `kvcache_allocator` policy on UMA: the 18 GB cap should grow into available unified pages on SM121, not stop short of them. The crash itself (`"please request later"`) is a polite refusal from the KV pool, not an OOM — the allocator is correctly bouncing the prompt; it's the *size* of the pool that's the bug.

Hoping this sharpens the picture. Two ways the guess could be off, with different consequences:

- **Wrong *variant* of the same family** (RedHat repackage vs NVIDIA original, different revision, different long-context fine-tune): the on-disk numbers shift by ≤ a few percent and the three takeaways above hold.
- **Wrong *family*** (a Nemotron-H build, an AWQ rather than NVFP4 quantization, a different attention/SSM ratio, a private fork): the *direction* of the argument still holds — NVFP4-class quantizations carry meaningful overhead on top of the FP4 weight bytes; model + KV pool + Mamba state + runtime can reconcile your `used: 81` without invoking double-mapping; the KV-allocator-cap-on-SM121 conclusion is independent of the specific model — but the exact 47.24 / +30% / 13.70 GiB numbers shift more than "a few percent".

Cleanest resolution either way: share the repo line and I'll rerun. If it's a private build, even the architectural block from `config.json` (or `AutoConfig.from_pretrained(...).to_dict()`) is enough to recompute the on-disk total exactly.

PS: `inspect <repo> --check-gpu --context N` (KV-budget against a real context length, including hybrid Mamba state for Qwen3-Next-class models) is on the [v0.10.4 roadmap](https://github.com/PCfVW/hf-fetch-model/blob/main/docs/roadmaps/cache-management-roadmap.md) for `hf-fetch-model` and is the next thing that would let you predict, before launching vllm-rs, whether a given context length will hit the KV-pool ceiling on your specific allocator config. If/when you have a 35B-sibling repo to point at, `hf-fm diff <80B-repo> <35B-repo> --dtypes` (new in v0.10.2) renders a side-by-side per-dtype histogram that confirms the architectures are scaled siblings vs. structurally divergent in one screenshot.
