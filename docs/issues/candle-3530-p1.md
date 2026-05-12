# candle #3530 — reply 1 (posted)

- **Target issue:** https://github.com/huggingface/candle/issues/3530
- **Status:** Posted (May 12, 2026)
- **Context:** First reply on the thread. @sempervictus (RageLtMan, SemperVictus, Boston MA — 203 public repos, 120 followers) reports candle uses 81 GB / 119 GB on NVIDIA Spark loading what they describe as "about the size of an 80B NVPF4 model" and hypothesizes double-mapping (mmap-then-copy → 2× bounds counting). Zero existing comments at draft time. The OP's question is binary: is the actual on-disk model size ~40 GiB (supporting the 2× theory) or ~80 GiB (ruling it out)? `hf-fm inspect --check-gpu --dtypes` reads the safetensors header byte counts — ground truth for that comparison — which is the diagnostic this reply offers. v0.10.1 shipped `--check-gpu` the same day as this draft, making this the first candle-side application of the new flag.
- **Outcome:** —
- **Lesson / Leverage angle:** First-responder framing on the freshest open issue; cleanest fit yet for v0.10.1's `--check-gpu`. The dtype histogram + verdict block is the differentiator over Python alternatives (`huggingface_hub` and `safetensors_explorer` don't ship an equivalent fit-against-actual-GPU view). If the OP shares the repo, we can run the command ourselves and post the result inline as a p2.

---

The double-mapping hypothesis turns on a binary question — the actual on-disk size of the model file — which `hf-fm inspect --check-gpu --dtypes` answers by reading each tensor's safetensors header via HTTP Range (no weight data downloaded):

- If the model packs **1 fp4 per byte** (one fp4 per `u8` cell, 4 bits of padding — common): 80B params ≈ 80 GiB on disk → your 81 GB `used` is approximately correct, no double-mapping, and the title's "model size unused" framing probably points at runtime accounting (KV / activations budget) rather than a bounds-calc bug.
- If the model packs **2 fp4 per byte** (true packed format): 80B params ≈ 40 GiB on disk → your 81 GB `used` is ~2× the model, supporting your mmap-then-copy hypothesis.

For reference, here's what the command's output shape looks like on an MXFP4-class model at a smaller scale (`openai/gpt-oss-20b`, same dtype family):

```
$ hf-fm inspect openai/gpt-oss-20b --check-gpu --dtypes

  Repo:   openai/gpt-oss-20b
  Source: aggregated across 4 shards

  Dtype  Tensors       Params       Size
  BF16       630        3.61B   6.72 GiB
  U8         192       20.30B  18.91 GiB
  ─────────────────────────────────────────
  822 tensors, 23.91B params

  Model weights:  25.63 GiB  (U8 + others, 23.91B params)
  GPU 0:          NVIDIA GeForce RTX 5060 Ti — 15.93 GiB VRAM
                  free: 13.53 GiB, used: 2.40 GiB
  Fit:            ✗ short by 12.10 GiB for the weights alone

  Note: reports weights only. Large-context inference typically needs ~1.3–1.5×
  weight size for KV cache and activations.
```

The `U8` row is the MXFP4 weights — safetensors has no native fp4 dtype, so the format stores them as raw bytes (one fp4 per byte, in this case) with separate scaling factors elsewhere. The `Size` column is the actual disk-byte count: that's the number to compare against your `used: 81` snapshot.

Could you share the specific repo you're loading on Spark? Running the command above against it lands the ground-truth weight bytes, and the gap math is sharp either way.

Hoping this helps.

PS: `hf-fm` is a small Rust CLI for HuggingFace repos (no Python dependency, no weight data fetched). If needed: `cargo install hf-fetch-model --features cli`; but don't hesitate to share the specific repo; I'll check it.
