# Per-layer shape variation in Gemma 4

*How `inspect` turned "the loader has a hardcoded value" into "the architecture varies by layer index" — and why the first answer was wrong.*

**Source:** candle [#3448](https://github.com/huggingface/candle/issues/3448) · archive: [candle-3448-p1](../issues/candle-3448-p1.md), [p2](../issues/candle-3448-p2.md) (superseded), [p3](../issues/candle-3448-p3.md), [p4](../issues/candle-3448-p4.md) (draft)

---

## The symptom

A user running candle's `gemma4` example against `google/gemma-4-E2B-it` hit:

```
Error: cannot find tensor model.embed_tokens.weight
```

Gemma 4 is multimodal — vision, audio, and a language tower. The embedding
tensor candle wanted *did* exist, just under a different name. One `inspect`
with a filter showed why:

```
hf-fm inspect google/gemma-4-E2B-it model.safetensors --filter embed
```
```
model.embed_audio.embedding_projection.weight       BF16  [1536, 1536]    4.50 MiB
model.embed_vision.embedding_projection.weight      BF16  [768, 768]      2.25 MiB
model.language_model.embed_tokens.weight            BF16  [262144, 1536]  768.00 MiB
model.language_model.embed_tokens_per_layer.weight  BF16  [262144, 8960]    4.38 GiB
model.vision_tower.patch_embedder.input_proj.weight BF16  [768, 768]      1.12 MiB
```

The language model's embeddings live under `model.language_model.` — candle was
looking for `model.embed_tokens.weight` without the multimodal prefix. The
reporter applied the prefix, confirmed "good catch," and moved on — straight
into a **second** error: a shape mismatch on layer 15's MLP, expected
`[6144, 1536]`, got `[12288, 1536]`.

## The diagnostic path — including the wrong turn

The obvious next move was to filter to the offending layer:

```
hf-fm inspect google/gemma-4-E2B-it model.safetensors --filter "layers.15.mlp"
```
```
model.language_model.layers.15.mlp.down_proj.weight  BF16  [1536, 12288]  36.00 MiB
model.language_model.layers.15.mlp.gate_proj.weight  BF16  [12288, 1536]  36.00 MiB
model.language_model.layers.15.mlp.up_proj.weight    BF16  [12288, 1536]  36.00 MiB
```

The intermediate size at layer 15 is 12288, not 6144. The tempting conclusion —
and the one we posted — was: *"so the config says 12288; candle is reading the
wrong value or hardcoding 6144."* (See [p2](../issues/candle-3448-p2.md).)

**That was wrong, and the silence said so** — the reply sat three days with no
response. The flaw: a single-layer filter answers "what is layer 15?" but not
"is every layer like layer 15?" It invited a false "constant value" mental
model for what is actually a *per-layer-variable* architecture.

The corrected path ([p3](../issues/candle-3448-p3.md)) cross-checked against
transformers' `modeling_gemma4.py`, which computes the intermediate size
*conditionally on the layer index*:

```python
first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
use_double_wide_mlp = config.use_double_wide_mlp and is_kv_shared_layer
self.intermediate_size = config.intermediate_size * (2 if use_double_wide_mlp else 1)
```

For `gemma-4-E2B-it`: `num_hidden_layers = 35`, `num_kv_shared_layers = 20`, so
`first_kv_shared_layer_idx = 15`. Layers 0–14 use 6144; layers 15–34 use 12288.
The boundary at **layer 15 matches the error exactly** — it is the first
"KV-shared, double-wide MLP" layer. candle's `Gemma4TextConfig` was missing
both `num_kv_shared_layers` and `use_double_wide_mlp`, and its decoder passed a
fixed `intermediate_size` to every layer.

The right tool for "is this shape constant across layers?" is the tree view,
not a single-layer filter. `inspect --tree` collapses structurally-identical
numeric siblings into a `layers.[0..N]` range and **declines to collapse when
they differ** — so a model with a per-layer shape change shows two ranges
(`layers.[0..14]` and `layers.[15..34]`) instead of one. The split *is* the
diagnosis, visible at a glance.

## The outcome — honestly

A verified root cause and a concrete candle code fix were posted in p3. There
is **no record that the fix was implemented or confirmed** — the thread went
quiet after p3, and a p4 follow-up (a debug-print to check whether the new
config fields were being parsed) was drafted but never posted because there was
nothing to respond to. So: the diagnosis was correct and complete, the fix was
spelled out in Rust, and it unblocks every Gemma 4 variant — but it sits in the
archive, not in a merged PR. Mitigated success: the tool did its job; the loop
didn't close.

## The transferable workflow

> When a model throws a shape mismatch on one specific layer, **do not trust a
> single-layer filter** — it can't tell uniform from per-layer. Run
> `inspect --tree` and read the ranges: one collapsed range means the shape is
> constant (suspect the loader); two or more ranges at a layer boundary mean the
> architecture varies by index (suspect a missing per-layer config field). The
> boundary the tree shows you is the layer index to look up in the reference
> implementation.
