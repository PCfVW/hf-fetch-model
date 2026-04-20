# candle #3401 — reply 1 (posted)

- **Target issue:** https://github.com/huggingface/candle/issues/3401 — "Adding support for LightOnOCR_2_1B"
- **Status:** Posted (Apr 16, 2026 — afternoon)
- **Context:** First-time contributor (@Abhinav5132) trying to port LightOnOCR-2-1B to candle. Hit the same multimodal-prefix pattern as #3448: weights use `model.language_model.*` but candle's qwen3 expects `model.*`. Zero prior replies on the 5-week-old issue. We used `hf-fm inspect --tree` to reveal the full architecture (language tower + Pixtral-style vision encoder + projection) and pointed to the directly-applicable candle precedents: `qwen3_vl`, `pixtral`, `paligemma`, `paddleocr_vl`.
- **Leverage angle:** personal connection — maintainer has been knowing LightOn people for years.

---

You're right that the weights have a `model.language_model.*` prefix — that's because LightOnOCR-2-1B is a multimodal model, not a pure LLM. The language tower is qwen3-shaped, but it's wrapped alongside a vision encoder and a projection module.

Here's the full structure:

```
$ hf-fm inspect lightonai/LightOnOCR-2-1B model.safetensors --tree

  Repo:     lightonai/LightOnOCR-2-1B
  File:     model.safetensors
  Source:   remote (2 HTTP requests)

  └── model.
      ├── language_model.
      │   ├── embed_tokens.weight  BF16  [151936, 1024]  296.75 MiB
      │   ├── layers.[0..27].   (×28)
      │   │   ├── input_layernorm.weight           BF16  [1024]  2.0 KiB
      │   │   ├── mlp.
      │   │   │   ├── down_proj.weight  BF16  [1024, 3072]  6.00 MiB
      │   │   │   ├── gate_proj.weight  BF16  [3072, 1024]  6.00 MiB
      │   │   │   └── up_proj.weight    BF16  [3072, 1024]  6.00 MiB
      │   │   ├── post_attention_layernorm.weight  BF16  [1024]  2.0 KiB
      │   │   └── self_attn.
      │   │       ├── k_norm.weight  BF16  [128]        256 B
      │   │       ├── k_proj.weight  BF16  [1024, 1024]  2.00 MiB
      │   │       ├── o_proj.weight  BF16  [1024, 2048]  4.00 MiB
      │   │       ├── q_norm.weight  BF16  [128]        256 B
      │   │       ├── q_proj.weight  BF16  [2048, 1024]  4.00 MiB
      │   │       └── v_proj.weight  BF16  [1024, 1024]  2.00 MiB
      │   └── norm.weight  BF16  [1024]  2.0 KiB
      ├── vision_encoder.
      │   ├── ln_pre.weight      BF16  [1024]            2.0 KiB
      │   ├── patch_conv.weight  BF16  [1024, 3, 14, 14] 1.15 MiB
      │   └── transformer.layers.[0..23].   (×24)
      │       ├── attention.{k,o,q,v}_proj.weight       BF16  [1024, 1024]  2.00 MiB each
      │       ├── attention_norm.weight                 BF16  [1024]        2.0 KiB
      │       ├── feed_forward.{down,gate,up}_proj.weight  BF16  (1024↔4096) 8.00 MiB each
      │       └── ffn_norm.weight                       BF16  [1024]        2.0 KiB
      └── vision_projection.
          ├── linear_1.weight                    BF16  [1024, 1024]  2.00 MiB
          ├── linear_2.weight                    BF16  [1024, 1024]  2.00 MiB
          ├── norm.weight                        BF16  [1024]        2.0 KiB
          └── patch_merger.merging_layer.weight  BF16  [1024, 4096]  8.00 MiB

  532 tensors, 1.01B params
```

So you have three things to port — but the good news is **candle already has direct precedents for all three**:

1. **Language tower (qwen3)** — uniform across all 28 layers (the `(×28)` range collapse confirms structural equality). The `q_norm`/`k_norm` inside `self_attn` is the qwen3 tell (vs qwen2). The reference implementation is [`candle-transformers/src/models/qwen3_vl/text.rs`](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/qwen3_vl/text.rs) — it already uses exactly the prefix pattern you need:
   ```rust
   let vb_m = vb.pp("model").pp("language_model");
   ```
   You likely don't need to touch the standalone `qwen3.rs` at all.

2. **Vision encoder** — Pixtral-style (24 layers, `attention_norm`/`ffn_norm` naming, `patch_conv` for patch embedding, `feed_forward.{down,gate,up}_proj` rather than `mlp.*`). candle has [`candle-transformers/src/models/pixtral/`](https://github.com/huggingface/candle/tree/main/candle-transformers/src/models/pixtral) — worth comparing tensor names closely; they may match directly.

3. **Vision projection** — `patch_merger.merging_layer` is specific to LightOnOCR. [`paligemma.rs`](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/paligemma.rs) shows the canonical multimodal-projector pattern. [`paddleocr_vl/`](https://github.com/huggingface/candle/tree/main/candle-transformers/src/models/paddleocr_vl) is the closest conceptual neighbor (also OCR + qwen-family + vision).

The shape to aim for:

```rust
pub struct LightOnOcr {
    language_model: qwen3::Model,            // vb.pp("model").pp("language_model")
    vision_encoder: pixtral::VisionModel,    // vb.pp("model").pp("vision_encoder")
    vision_projection: VisionProjection,     // vb.pp("model").pp("vision_projection")
}
```

This way you don't modify qwen3 — you wrap it, the same way `qwen3_vl` and `paligemma` do. First-time contributions with a clean composition like this tend to land well. 👍

PS: `hf-fm inspect --tree` reads safetensors headers via HTTP Range requests, i.e. no weight data downloaded; if needed: `cargo install hf-fetch-model --features cli` (requires v0.9.6 or later).
