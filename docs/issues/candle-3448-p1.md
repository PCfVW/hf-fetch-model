# candle #3448 — reply 1 (posted)

- **Target issue:** https://github.com/huggingface/candle/issues/3448
- **Status:** Posted (Apr 11, 2026)
- **Context:** First reply on the thread. @AlpineVibrations and @KarstenB both reported `Error: cannot find tensor model.embed_tokens.weight` when running the candle `gemma4` example against `google/gemma-4-E2B-it`. No one had answered. We used `hf-fm inspect --filter embed` to diagnose.
- **Outcome:** @AlpineVibrations replied "good catch" and tried the fix; it resolved the naming issue and surfaced a shape mismatch on `layers.15.mlp.gate_proj.weight` (handled in p2/p3).

---

The error `cannot find tensor model.embed_tokens.weight` is a naming mismatch. Gemma 4 is multimodal (vision + audio + language), so all tensors are namespaced under their modality. The embedding tensor candle is looking for actually exists as:

```
model.language_model.embed_tokens.weight    BF16  [262144, 1536]  768.00 MiB
```

Here are all the "embed" tensors in the model:

```
$ hf-fm inspect google/gemma-4-E2B-it model.safetensors --filter embed

  Tensor                                              Dtype    Shape                  Size     Params
  model.embed_audio.embedding_projection.weight       BF16     [1536, 1536]       4.50 MiB       2.4M
  model.embed_vision.embedding_projection.weight      BF16     [768, 768]         2.25 MiB       1.2M
  model.language_model.embed_tokens.weight            BF16     [262144, 1536]   768.00 MiB     402.7M
  model.language_model.embed_tokens_per_layer.weight  BF16     [262144, 8960]     4.38 GiB      2.35B
  model.vision_tower.patch_embedder.input_proj.weight BF16     [768, 768]         1.12 MiB     589.8K
  model.vision_tower.patch_embedder.position_embedding_table BF16  [2, 10240, 768]  30.00 MiB   15.7M

  6/2011 tensors, 2.77B/5.12B params (filter: "embed")
```

The candle gemma4 example likely needs to prepend `model.language_model.` when loading tensor names, similar to how other multimodal models (PaliGemma, LLaVA) handle the prefix.

PS: `hf-fm inspect` reads safetensors headers via HTTP Range requests, i.e. no weight data downloaded; if needed: install with `cargo install hf-fetch-model --features cli`.
