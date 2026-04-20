# candle #3448 — reply 2 (posted, superseded by p3)

- **Target issue:** https://github.com/huggingface/candle/issues/3448
- **Status:** Posted (2026-04-13), superseded by [p3](candle-3448-p3.md) (oversimplified diagnosis).
- **Context:** @AlpineVibrations tried the `model.language_model.` prefix fix from p1 and hit a new error: `shape mismatch for model.language_model.layers.15.mlp.gate_proj.weight, expected: [6144, 1536], got: [12288, 1536]`. We used `hf-fm inspect --filter "layers.15.mlp"` and drew an **incorrect** conclusion: "intermediate_size is 12288, not 6144."
- **Outcome:** No reply in 3 days (as of 2026-04-16). The silence itself was informative — possibly because the diagnosis was too thin to act on, or because nobody could reconcile "intermediate_size is 12288" with the config.json value of 6144. This pushed us to dig deeper, which produced [p3](candle-3448-p3.md).
- **Why it's wrong:** the shape is per-layer, not uniform. Layer 0 is `[6144, 1536]`, layer 15 is `[12288, 1536]`. Gemma 4 applies `intermediate_size * 2` to the last `num_kv_shared_layers` (20 of 35) via the `use_double_wide_mlp` config flag. candle's `Gemma4TextConfig` is missing this flag entirely. The real fix lives in [p3](candle-3448-p3.md).
- **Lesson:** when a model exhibits a shape mismatch, don't assume the single-value config is "just wrong" — verify whether the shape is uniform across all N layers (`hf-fm inspect --tree` would have shown this immediately, since range collapse would have refused to fire). This diagnostic pattern is now part of the `--tree` story.

---

@AlpineVibrations Following up: `gate_proj` and `up_proj` are separate tensors (not merged), both `[12288, 1536]` (vision tower tensors trimmed):

```
$ hf-fm inspect google/gemma-4-E2B-it model.safetensors --filter "layers.15.mlp"

  Tensor                                              Dtype    Shape              Size
  model.language_model.layers.15.mlp.down_proj.weight BF16     [1536, 12288]  36.00 MiB
  model.language_model.layers.15.mlp.gate_proj.weight BF16     [12288, 1536]  36.00 MiB
  model.language_model.layers.15.mlp.up_proj.weight   BF16     [12288, 1536]  36.00 MiB
```

So the `intermediate_size` is 12288, not 6144. The candle gemma4 example likely reads the wrong value from `config.json` or has it hardcoded.
