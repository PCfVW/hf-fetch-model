# candle #3448 — reply 4 (draft)

- **Target issue:** https://github.com/huggingface/candle/issues/3448
- **Status:** Draft (not yet posted)
- **Context:** @AlpineVibrations tried the fix proposed in [p3](candle-3448-p3.md) and reported "still the same error as I reported above. I only added what you specified and it didn't seem to fix it." The exact-same-error signal (layer 15, expected [6144,1536], got [12288,1536]) strongly suggests the double-wide branch never activated. Without his diff we can't tell why — most likely config field wasn't parsed, or wasn't used at the MLP call site.
- **Purpose:** Lowest-friction next step — ask for his diff + give him one debug print that will diagnose the issue regardless. Doesn't require cloning candle ourselves.

---

Could you share your actual diff to `config.rs` and `text.rs`? The error being exactly the same (same layer, same expected/got values) usually means the branch didn't activate — most likely one of:

1. `num_kv_shared_layers` stayed at the `#[serde(default)] = 0` because the JSON field wasn't wired into parsing
2. `use_double_wide_mlp` stayed `false` for the same reason
3. The config fields were added but not actually used when constructing `MLP` in `DecoderLayer::new()`

If you can't share the diff, a quick diagnostic: add this line just before `let mlp = MLP::new(...)`:

```rust
eprintln!(
    "layer {layer_idx}: use_dw={}, kv_shared_from={}, intermediate={}",
    cfg.use_double_wide_mlp,
    cfg.num_hidden_layers.saturating_sub(cfg.num_kv_shared_layers),
    effective_intermediate,
);
```

For `gemma-4-E2B-it` it should print `intermediate=6144` for layers 0..14 and `12288` for 15..34. If `use_dw=false` throughout, it's (1) or (2); if `use_dw=true` but `intermediate=6144` everywhere, it's a wiring issue at the MLP call site.
