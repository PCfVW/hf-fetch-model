// SPDX-License-Identifier: MIT OR Apache-2.0

//! `GPU`-fit verdict for `inspect --check-gpu`.
//!
//! Wraps the [`hypomnesis`] crate's `device_info` API in a small, friendly
//! surface tuned for the `hf-fm inspect` output. The unit of work is one
//! device query plus a single rendering pass — no streaming, no async.
//!
//! All items in this module are reachable from the binary via the
//! `#[path = "../gpu_check.rs"]` declaration in `src/bin/main.rs`. The file
//! lives at the crate root rather than under `src/bin/` so that future
//! library consumers can absorb it with a one-line `mod gpu_check;` in
//! `src/lib.rs` should public exposure ever be wanted.

use std::collections::HashMap;

use hf_fetch_model::inspect::{ModelConfig, TensorInfo};
use hypomnesis::GpuDeviceInfo;

use crate::format::format_size;

/// Outcome of a single GPU probe for `inspect --check-gpu`.
///
/// Exactly one of [`Self::device`] and [`Self::error`] is `Some` — `device`
/// when [`hypomnesis::device_info`] returned `Ok`, `error` otherwise. The
/// rendering layer chooses its output shape based on which field is populated.
#[derive(Debug)]
#[allow(clippy::exhaustive_structs)] // EXHAUSTIVE: crate-private dispatch struct; binary owns all construction sites
pub struct GpuCheckResult {
    /// Zero-based device index the user requested.
    pub device_index: u32,
    /// Device info on success. `None` when the probe failed.
    pub device: Option<GpuDeviceInfo>,
    /// One-line human-readable message on probe failure. `None` on success.
    pub error: Option<String>,
}

/// Probes the requested device via [`hypomnesis::device_info`] and converts
/// the result into the binary's friendly [`GpuCheckResult`] shape.
///
/// On every error path, the result carries `device: None, error: Some(...)`;
/// callers should print the error string verbatim and skip the verdict block.
/// The function never panics and is safe to call on systems with no NVIDIA
/// GPU, no NVML / DXGI, or with the index past the device count.
#[must_use]
pub fn query_gpu(index: u32) -> GpuCheckResult {
    match hypomnesis::device_info(index) {
        Ok(device) => GpuCheckResult {
            device_index: index,
            device: Some(device),
            error: None,
        },
        Err(e) => GpuCheckResult {
            device_index: index,
            device: None,
            error: Some(friendly_error(&e)),
        },
    }
}

/// Translates a [`hypomnesis::HypomnesisError`] into a single user-facing line.
///
/// Variants surfaced today:
///
/// | Variant | Message |
/// |---------|---------|
/// | `DeviceIndexOutOfRange { index, count }` | `"index {N} out of range (have {M} device(s))"` (with singular/plural agreement) |
/// | `NoGpuSource` | `"no NVIDIA device detected (NVML / DXGI not usable)"` |
/// | `Nvml(s)` / `Dxgi(s)` / `NvidiaSmi(s)` | `"{backend} backend reported: {s}"` |
/// | `Ram(_)` / `Io(_)` | `"unexpected error: {err}"` (should not occur for `device_info`) |
/// | future variants (`HypomnesisError` is `#[non_exhaustive]`) | `"hypomnesis error: {err}"` (generic fallthrough) |
fn friendly_error(err: &hypomnesis::HypomnesisError) -> String {
    use hypomnesis::HypomnesisError as E;
    match err {
        E::DeviceIndexOutOfRange { index, count } => {
            let plural = if *count == 1 { "device" } else { "devices" };
            format!("index {index} out of range (have {count} {plural})")
        }
        E::NoGpuSource => "no NVIDIA device detected (NVML / DXGI not usable)".to_owned(),
        E::Nvml(s) => format!("NVML backend reported: {s}"),
        E::Dxgi(s) => format!("DXGI backend reported: {s}"),
        E::NvidiaSmi(s) => format!("nvidia-smi backend reported: {s}"),
        // EXPLICIT: Ram / Io should not surface for device_info, but format defensively
        // so a future hypomnesis revision that uses them doesn't break our rendering.
        E::Ram(_) | E::Io(_) => format!("unexpected error: {err}"),
        // EXHAUSTIVE: HypomnesisError is `#[non_exhaustive]`; cover future variants generically
        _ => format!("hypomnesis error: {err}"),
    }
}

/// Total tensor data bytes — the figure that will land in GPU VRAM at load
/// time, excluding the small `JSON` header.
///
/// Computed as the sum of [`TensorInfo::byte_len`] across every tensor.
/// Saturates rather than wrapping on overflow.
#[must_use]
pub fn sum_tensor_bytes(tensors: &[TensorInfo]) -> u64 {
    tensors
        .iter()
        .map(TensorInfo::byte_len)
        .fold(0u64, u64::saturating_add)
}

/// Most-represented dtype across the tensor list, by parameter count.
///
/// Returns:
/// - `"unknown"` for an empty tensor list,
/// - the dtype name when one dtype carries ≥99% of params (treat near-pure
///   mixtures like `"BF16"` plus a sliver of `"F32"` `LayerNorm` scales as
///   pure `BF16` for display),
/// - `"{dominant} + others"` when several dtypes are present but one has
///   the largest share.
///
/// Pure function — no I/O, no hardware access. Drives the parenthesized
/// dtype label on the `Model weights:` line of the verdict block.
#[must_use]
pub fn dominant_dtype_label(tensors: &[TensorInfo]) -> String {
    if tensors.is_empty() {
        return "unknown".to_owned();
    }

    let mut by_dtype: HashMap<&str, u64> = HashMap::new();
    for t in tensors {
        // BORROW: explicit .as_str() for &String → &str (HashMap key)
        let entry = by_dtype.entry(t.dtype.as_str()).or_insert(0u64);
        *entry = entry.saturating_add(t.num_elements());
    }

    let Some((dominant, &dominant_count)) = by_dtype.iter().max_by_key(|(_, c)| **c) else {
        // EXPLICIT: by_dtype is non-empty (tensors non-empty checked above), but
        // max_by_key returns Option — handle defensively rather than .unwrap().
        return "unknown".to_owned();
    };

    if by_dtype.len() == 1 {
        return (*dominant).to_owned();
    }

    let total = by_dtype.values().copied().fold(0u64, u64::saturating_add);
    if total == 0 {
        // BORROW: explicit .to_owned() for &&str → owned String
        return (*dominant).to_owned();
    }

    // CAST: u64 → f64, precision loss acceptable; value is a display-only ratio
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let ratio = dominant_count as f64 / total as f64;

    if ratio >= 0.99 {
        (*dominant).to_owned()
    } else {
        format!("{dominant} + others")
    }
}

/// How a KV-cache estimate was derived — drives the rendered caveat and the
/// `--json` `kv_cache.path` tag.
///
/// `#[non_exhaustive]` so adding KV-path variants (e.g. a future `MLA`
/// estimate) stays non-breaking; all current match sites live in this module.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum KvCachePath {
    /// Standard full-attention `MHA` / `GQA` estimate.
    Exact,
    /// Uniform sliding window: every layer's effective length is capped at
    /// `window` (only used when the requested context exceeds it).
    SlidingWindowCapped {
        /// Sliding-window span in tokens.
        window: u32,
    },
    /// Mixed local/global layers (Gemma-2 1:1, Gemma-3 5:1): a blended
    /// estimate counting `local_layers` at the capped length and
    /// `global_layers` at the full context. Approximate — the exact global
    /// layer positions are not modelled, only their count.
    SlidingWindowPartial {
        /// Number of windowed (local-attention) layers.
        local_layers: u32,
        /// Number of full-attention (global) layers.
        global_layers: u32,
        /// Sliding-window span in tokens.
        window: u32,
    },
    /// Multi-head latent attention (`DeepSeek`): the naive formula overestimates
    /// by roughly 10×, so the estimate is skipped. The latent-KV size is
    /// approximately `num_layers × (kv_lora_rank + qk_rope_head_dim) ×
    /// seq_len × elem_bytes` (no head multiply, no K/V doubling) — recorded
    /// here for a future `--mla-estimate`.
    MlaSkipped,
    /// The estimate could not be computed; `reason` is the user-facing wording.
    Unavailable {
        /// Short human-readable explanation (no config, missing dims, …).
        reason: &'static str,
    },
    /// Hybrid Mamba/attention model (Granite-4, Nemotron-H, Bamba,
    /// Qwen3-Next): KV is budgeted over the `attn_layers` attention layers
    /// only (the recurrent layers hold no growing cache), plus a fixed
    /// recurrent-state term that is constant in context length.
    Hybrid {
        /// Number of attention layers holding a context-growing KV cache.
        attn_layers: u32,
        /// Number of recurrent (Mamba / linear-attention) layers.
        recurrent_layers: u32,
        /// KV-cache bytes for the attention layers (grows with context).
        kv_bytes: u64,
        /// Fixed recurrent-state bytes (Mamba2), or `None` when the recurrent
        /// type is not Mamba2 (Gated `DeltaNet`, Mamba1) — then excluded.
        recurrent_bytes: Option<u64>,
    },
}

impl KvCachePath {
    /// Stable `snake_case` tag for the `--json` `kv_cache.path` field.
    #[must_use]
    pub fn json_tag(&self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::SlidingWindowCapped { .. } => "sliding_window_capped",
            Self::SlidingWindowPartial { .. } => "sliding_window_partial",
            Self::MlaSkipped => "mla_skipped",
            Self::Unavailable { .. } => "unavailable",
            Self::Hybrid { .. } => "hybrid",
        }
    }
}

/// Global-attention period for a mixed local/global sliding-window layout, or
/// `None` for a uniform sliding window.
///
/// Gemma-3 states the period directly via `sliding_window_pattern` (6 ⇒ every
/// 6th layer is global). Gemma-2 alternates 1:1 with no explicit key, so it is
/// recognized by `model_type` and treated as period 2.
fn sliding_window_period(cfg: &ModelConfig) -> Option<u32> {
    if let Some(p) = cfg.sliding_window_pattern {
        if p >= 2 {
            return Some(p);
        }
    }
    // BORROW: explicit .as_deref() for Option<String> → Option<&str>
    if cfg.model_type.as_deref() == Some("gemma2") {
        return Some(2);
    }
    None
}

/// Attention vs recurrent layer counts for a hybrid Mamba/attention model.
struct HybridLayers {
    /// Layers using attention (and so a context-growing KV cache).
    attn_layers: u32,
    /// Layers using a recurrent operator (Mamba / linear attention).
    recurrent_layers: u32,
}

/// Classifies a hybrid model's layers into attention vs recurrent counts, or
/// `None` for a non-hybrid (pure transformer) model.
///
/// Tries four config signals in order: `layer_types` (Granite-4),
/// `hybrid_override_pattern` (Nemotron-H: `*` = attention, `M` = Mamba,
/// `-` = FFN-only), `attn_layer_indices` (Bamba), and `full_attention_interval`
/// (Qwen3-Next: every `N`-th layer is attention). Returns `None` when no
/// recurrent layers are found — a pure transformer that merely lists
/// `layer_types` (e.g. Gemma-3's all-attention list) is not hybrid and falls
/// through to the standard path.
fn classify_hybrid_layers(cfg: &ModelConfig) -> Option<HybridLayers> {
    // 1. layer_types: explicit per-layer kind list.
    if let Some(types) = &cfg.layer_types {
        let mut attn = 0u32;
        let mut recurrent = 0u32;
        for t in types {
            match t.as_str() {
                "mamba" | "mamba2" | "linear_attention" | "linear" | "recurrent" => recurrent += 1,
                "attention" | "full_attention" | "sliding_attention" => attn += 1,
                // EXPLICIT: unknown layer kinds (FFN-only, etc.) contribute to neither.
                _ => {}
            }
        }
        if recurrent > 0 {
            return Some(HybridLayers {
                attn_layers: attn,
                recurrent_layers: recurrent,
            });
        }
    }

    // 2. hybrid_override_pattern: Nemotron-H char layout (`-` FFN layers ignored).
    if let Some(pattern) = &cfg.hybrid_override_pattern {
        let attn = u32::try_from(pattern.chars().filter(|c| *c == '*').count()).unwrap_or(0);
        let recurrent = u32::try_from(pattern.chars().filter(|c| *c == 'M').count()).unwrap_or(0);
        if recurrent > 0 {
            return Some(HybridLayers {
                attn_layers: attn,
                recurrent_layers: recurrent,
            });
        }
    }

    // 3. attn_layer_indices: explicit attention-layer positions (Bamba); the
    //    remaining layers are recurrent.
    if let (Some(indices), Some(total)) = (&cfg.attn_layer_indices, cfg.num_hidden_layers) {
        let attn = u32::try_from(indices.len()).unwrap_or(0);
        if attn > 0 && attn < total {
            return Some(HybridLayers {
                attn_layers: attn,
                recurrent_layers: total - attn,
            });
        }
    }

    // 4. full_attention_interval: every N-th layer is attention (Qwen3-Next).
    if let (Some(interval), Some(total)) = (cfg.full_attention_interval, cfg.num_hidden_layers) {
        if interval >= 2 {
            let attn = total / interval;
            if attn > 0 && attn < total {
                return Some(HybridLayers {
                    attn_layers: attn,
                    recurrent_layers: total - attn,
                });
            }
        }
    }

    None
}

/// Fixed Mamba2 recurrent-state bytes for one layer, or `None` when the Mamba2
/// dimensions are absent (Gated `DeltaNet` / Mamba1 / unknown recurrent type, for
/// which the recurrent term is excluded).
///
/// `ssm_state = n_heads × d_head × d_state`; `conv_state = conv_dim × d_conv`
/// with `conv_dim = n_heads·d_head + 2·n_groups·d_state` — the cache shapes of
/// the HF `Mamba2Mixer`. Constant in context length. `u64::from` is lossless,
/// so no `as` casts are needed.
fn mamba2_state_bytes_per_layer(cfg: &ModelConfig, elem_bytes: u8) -> Option<u64> {
    let n_heads = u64::from(cfg.mamba_n_heads?);
    let d_head = u64::from(cfg.mamba_d_head?);
    let d_state = u64::from(cfg.mamba_d_state?);
    let d_conv = u64::from(cfg.mamba_d_conv?);
    let n_groups = u64::from(cfg.mamba_n_groups.unwrap_or(1));

    let inner = n_heads.saturating_mul(d_head);
    let ssm_state = inner.saturating_mul(d_state);
    let conv_dim = inner.saturating_add(2u64.saturating_mul(n_groups).saturating_mul(d_state));
    let conv_state = conv_dim.saturating_mul(d_conv);

    Some(
        ssm_state
            .saturating_add(conv_state)
            .saturating_mul(u64::from(elem_bytes)),
    )
}

/// Estimates KV-cache bytes for a transformer at context length `seq_len`.
///
/// Pure — no I/O. Returns `(Some(bytes), path)` for a computable estimate and
/// `(None, path)` when it is skipped ([`KvCachePath::MlaSkipped`]) or
/// unavailable ([`KvCachePath::Unavailable`]); failure is encoded in the path
/// rather than an error so `--check-gpu` never gates.
///
/// The standard estimate is the well-established
/// `2 × num_hidden_layers × num_key_value_heads × head_dim × seq_len ×
/// kv_elem_bytes` (the `2` covers K and V, batch size 1). `kv_elem_bytes` is
/// the KV element size — the model's activation dtype, **independent of weight
/// quantization** (the caller derives it from the config's `torch_dtype`).
///
/// Limitations: multi-head latent attention (`DeepSeek` `MLA`) is skipped, and
/// mixed local/global sliding-window layouts are approximated by layer count
/// (within a few percent). Both are reported via the returned [`KvCachePath`].
#[must_use]
pub fn kv_cache_bytes(
    cfg: &ModelConfig,
    seq_len: u64,
    kv_elem_bytes: u8,
) -> (Option<u64>, KvCachePath) {
    // MLA: latent attention compresses K/V; the naive formula does not apply.
    if cfg.kv_lora_rank.is_some() {
        return (None, KvCachePath::MlaSkipped);
    }

    let unavailable = KvCachePath::Unavailable {
        reason: "config.json missing attention dims",
    };

    // The standard formula needs the layer count and the query-head count
    // (the latter both for the GQA fallback and the head_dim derivation).
    let (Some(layers), Some(attn_heads)) = (cfg.num_hidden_layers, cfg.num_attention_heads) else {
        return (None, unavailable);
    };

    let kv_heads = cfg.num_key_value_heads.unwrap_or(attn_heads);

    let head_dim = match cfg.head_dim {
        Some(h) => h,
        // Derive from hidden size when not stated explicitly.
        None => match cfg.hidden_size {
            Some(hidden) if attn_heads > 0 => hidden / attn_heads,
            _ => return (None, unavailable),
        },
    };

    if layers == 0 || kv_heads == 0 || head_dim == 0 {
        return (None, unavailable);
    }

    // Per-layer, per-token bytes for K and V together. `u64::from` is lossless,
    // so no `as` casts are needed.
    let per_layer_per_token = 2u64
        .saturating_mul(u64::from(kv_heads))
        .saturating_mul(u64::from(head_dim))
        .saturating_mul(u64::from(kv_elem_bytes));

    // Hybrid Mamba/attention: KV applies only to the attention layers (the
    // recurrent layers hold a fixed state that does not grow with context),
    // plus a Mamba2 recurrent-state term when its dims are present.
    if let Some(h) = classify_hybrid_layers(cfg) {
        let kv_bytes = per_layer_per_token
            .saturating_mul(u64::from(h.attn_layers))
            .saturating_mul(seq_len);
        let recurrent_bytes = mamba2_state_bytes_per_layer(cfg, kv_elem_bytes)
            .map(|per| per.saturating_mul(u64::from(h.recurrent_layers)));
        let total = kv_bytes.saturating_add(recurrent_bytes.unwrap_or(0));
        return (
            Some(total),
            KvCachePath::Hybrid {
                attn_layers: h.attn_layers,
                recurrent_layers: h.recurrent_layers,
                kv_bytes,
                recurrent_bytes,
            },
        );
    }

    // Sliding window binds only when present, enabled, and shorter than N.
    let window_binds = cfg.use_sliding_window != Some(false)
        && cfg
            .sliding_window
            .is_some_and(|w| w > 0 && seq_len > u64::from(w));

    if window_binds {
        // `is_some_and` above guarantees `Some(w)`; recover it.
        let window = cfg.sliding_window.unwrap_or(0);
        let capped = u64::from(window);

        if let Some(period) = sliding_window_period(cfg) {
            // Mixed local/global: a full-attention layer every `period`-th.
            let global_layers = layers / period;
            let local_layers = layers.saturating_sub(global_layers);
            let local_bytes = u64::from(local_layers).saturating_mul(capped);
            let global_bytes = u64::from(global_layers).saturating_mul(seq_len);
            let bytes =
                per_layer_per_token.saturating_mul(local_bytes.saturating_add(global_bytes));
            return (
                Some(bytes),
                KvCachePath::SlidingWindowPartial {
                    local_layers,
                    global_layers,
                    window,
                },
            );
        }

        // Uniform sliding window: every layer capped at the window.
        let bytes = per_layer_per_token
            .saturating_mul(u64::from(layers))
            .saturating_mul(capped);
        return (Some(bytes), KvCachePath::SlidingWindowCapped { window });
    }

    // Standard full attention (MHA / GQA).
    let bytes = per_layer_per_token
        .saturating_mul(u64::from(layers))
        .saturating_mul(seq_len);
    (Some(bytes), KvCachePath::Exact)
}

/// KV-cache figures for the `--check-gpu --context` verdict, computed by the
/// caller from the model's `config.json` and folded into both the text and
/// `--json` renderers.
#[derive(Debug, Clone)]
#[allow(clippy::exhaustive_structs)] // EXHAUSTIVE: crate-private; binary owns all construction sites
pub struct KvComputed {
    /// Requested context length (`--context N`).
    pub context: u32,
    /// KV element size in bytes (the activation dtype). `0` when unavailable.
    pub elem_bytes: u8,
    /// Display label for the KV element dtype (e.g. `"BF16"`).
    pub dtype_label: String,
    /// Estimated KV-cache bytes, or `None` when skipped / unavailable.
    pub bytes: Option<u64>,
    /// How the estimate was derived; drives the caveat and the `--json` tag.
    pub path: KvCachePath,
}

/// Prints the `KV cache …` line for the verdict block.
///
/// A computable estimate prints `KV cache @ ctx=N: X (DTYPE)` plus a
/// path-specific caveat; a skipped / unavailable estimate prints a one-line
/// reason.
fn print_kv_line(kv: &KvComputed) {
    // Hybrid prints two lines (attention KV + recurrent state) — handle it
    // before the single-line paths.
    if let KvCachePath::Hybrid {
        attn_layers,
        recurrent_layers,
        kv_bytes,
        recurrent_bytes,
    } = &kv.path
    {
        println!(
            "  KV cache @ ctx={}:  {}  ({}, {attn_layers} attn layers)",
            kv.context,
            format_size(*kv_bytes),
            kv.dtype_label,
        );
        match recurrent_bytes {
            Some(rb) => println!(
                "  Recurrent state:   {}  ({recurrent_layers} Mamba2 layers, constant)",
                format_size(*rb),
            ),
            None => {
                println!("  Recurrent state:   excluded (small, constant; non-Mamba2 recurrent)");
            }
        }
        return;
    }

    let Some(bytes) = kv.bytes else {
        match &kv.path {
            KvCachePath::MlaSkipped => println!(
                "  KV cache:       skipped (MLA / latent attention \u{2014} naive estimate unreliable)"
            ),
            KvCachePath::Unavailable { reason } => {
                println!("  KV cache:       unavailable ({reason})");
            }
            // EXPLICIT: these paths always carry Some(bytes); defensive fallback.
            KvCachePath::Exact
            | KvCachePath::SlidingWindowCapped { .. }
            | KvCachePath::SlidingWindowPartial { .. }
            | KvCachePath::Hybrid { .. } => {
                println!("  KV cache:       unavailable");
            }
        }
        return;
    };

    let caveat = match &kv.path {
        KvCachePath::SlidingWindowCapped { window } => {
            format!("  (capped at sliding_window={window})")
        }
        KvCachePath::SlidingWindowPartial {
            local_layers,
            global_layers,
            ..
        } => format!("  (approx; {local_layers} local + {global_layers} global layers)"),
        // Exact needs no caveat; Hybrid returned above; None-bytes never reach here.
        KvCachePath::Exact
        | KvCachePath::MlaSkipped
        | KvCachePath::Unavailable { .. }
        | KvCachePath::Hybrid { .. } => String::new(),
    };

    println!(
        "  KV cache @ ctx={}:  {}  ({}){caveat}",
        kv.context,
        format_size(bytes),
        kv.dtype_label,
    );
}

/// Renders the verdict block to stdout — four to seven lines depending on
/// whether the GPU probe succeeded.
///
/// Output shape on success:
///
/// ```text
///   Model weights:  9.54 GiB  (BF16, 5.12B params)
///   GPU 0:          NVIDIA GeForce RTX 5060 Ti — 16.0 GiB VRAM
///                   free: 14.2 GiB, used: 1.8 GiB
///   Fit:            ✓ 4.66 GiB headroom for weights + KV cache + runtime
///
///   Note: reports weights only. Large-context inference typically needs ~1.3–1.5×
///   weight size for KV cache and activations.
/// ```
///
/// On failure the `GPU N:` line carries the friendly error from [`query_gpu`]
/// and the `Fit:` / note lines are omitted.
///
/// With `kv` set (`--check-gpu --context N`), a `KV cache @ ctx=N` line and a
/// `Total: weights + KV` line are added, and the `Fit:` verdict is measured
/// against the total rather than the weights alone; the weights-only note is
/// dropped. When the KV estimate is skipped or unavailable, the caveat line is
/// printed but the verdict falls back to weights-only.
pub fn print_gpu_check(
    result: &GpuCheckResult,
    weight_bytes: u64,
    dtype_label: &str,
    total_params: u64,
    kv: Option<&KvComputed>,
) {
    println!();
    println!(
        "  Model weights:  {}  ({dtype_label}, {} params)",
        format_size(weight_bytes),
        format_params(total_params),
    );

    // KV-cache + Total lines (only with `--context`). `total_bytes` is what the
    // Fit verdict is measured against — weights alone when KV is absent or
    // could not be computed.
    // `fit_basis` is `Some(label)` when a computable KV estimate is counted —
    // the label describes what `total_bytes` includes and is reused on the Fit
    // line so the two stay consistent.
    let mut total_bytes = weight_bytes;
    let mut fit_basis: Option<&str> = None;
    if let Some(kv) = kv {
        print_kv_line(kv);
        if let Some(kv_bytes) = kv.bytes {
            total_bytes = weight_bytes.saturating_add(kv_bytes);
            // Hybrids with a Mamba2 term fold weights + KV + recurrent state.
            let label = if matches!(
                kv.path,
                KvCachePath::Hybrid {
                    recurrent_bytes: Some(_),
                    ..
                }
            ) {
                "weights + KV + state"
            } else {
                "weights + KV"
            };
            fit_basis = Some(label);
            println!("  Total:          {}  ({label})", format_size(total_bytes));
        }
    }

    let Some(ref dev) = result.device else {
        // BORROW: explicit .as_deref() for Option<String> → Option<&str>
        let msg = result
            .error
            .as_deref()
            .unwrap_or("device info unavailable (no further detail)");
        println!(
            "  GPU {}:          unavailable — {msg}",
            result.device_index
        );
        return;
    };

    let name = dev.name_or_unknown();
    println!(
        "  GPU {}:          {name} — {} VRAM",
        result.device_index,
        format_size(dev.total_bytes),
    );
    println!(
        "                  free: {}, used: {}",
        format_size(dev.free_bytes),
        format_size(dev.used_bytes),
    );

    if dev.free_bytes >= total_bytes {
        let headroom = dev.free_bytes - total_bytes;
        if let Some(basis) = fit_basis {
            println!(
                "  Fit:            \u{2713} {} headroom ({basis}; runtime extra)",
                format_size(headroom),
            );
        } else {
            println!(
                "  Fit:            \u{2713} {} headroom for weights + KV cache + runtime",
                format_size(headroom),
            );
        }
    } else {
        let short = total_bytes - dev.free_bytes;
        if let Some(basis) = fit_basis {
            println!(
                "  Fit:            \u{2717} short by {} for {basis}",
                format_size(short),
            );
        } else {
            println!(
                "  Fit:            \u{2717} short by {} for the weights alone",
                format_size(short),
            );
        }
    }

    println!();
    if fit_basis.is_some() {
        println!(
            "  Note: excludes activations and framework overhead (typically a few hundred MiB)."
        );
    } else {
        println!(
            "  Note: reports weights only. Large-context inference typically needs ~1.3\u{2013}1.5\u{00d7}"
        );
        println!("  weight size for KV cache and activations.");
    }
}

/// Builds the `kv_cache` JSON object for the `--check-gpu --context` verdict.
///
/// `bytes` / `elem_bytes` are present only for a computable estimate; `window`
/// (+ `local_layers` / `global_layers`) accompany the sliding-window paths;
/// `reason` accompanies the unavailable path. `path` is always present.
fn kv_cache_json(kv: &KvComputed) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert(
        "context".to_owned(),
        serde_json::Value::Number(kv.context.into()),
    );
    obj.insert(
        "path".to_owned(),
        serde_json::Value::String(kv.path.json_tag().to_owned()),
    );
    if let Some(bytes) = kv.bytes {
        obj.insert(
            "elem_bytes".to_owned(),
            serde_json::Value::Number(kv.elem_bytes.into()),
        );
        obj.insert("bytes".to_owned(), serde_json::Value::Number(bytes.into()));
    }
    match &kv.path {
        KvCachePath::SlidingWindowCapped { window } => {
            obj.insert(
                "window".to_owned(),
                serde_json::Value::Number((*window).into()),
            );
        }
        KvCachePath::SlidingWindowPartial {
            window,
            local_layers,
            global_layers,
        } => {
            obj.insert(
                "window".to_owned(),
                serde_json::Value::Number((*window).into()),
            );
            obj.insert(
                "local_layers".to_owned(),
                serde_json::Value::Number((*local_layers).into()),
            );
            obj.insert(
                "global_layers".to_owned(),
                serde_json::Value::Number((*global_layers).into()),
            );
        }
        KvCachePath::Unavailable { reason } => {
            obj.insert(
                "reason".to_owned(),
                serde_json::Value::String((*reason).to_owned()),
            );
        }
        KvCachePath::Hybrid {
            attn_layers,
            recurrent_layers,
            kv_bytes,
            recurrent_bytes,
        } => {
            obj.insert(
                "attn_layers".to_owned(),
                serde_json::Value::Number((*attn_layers).into()),
            );
            obj.insert(
                "recurrent_layers".to_owned(),
                serde_json::Value::Number((*recurrent_layers).into()),
            );
            obj.insert(
                "kv_bytes".to_owned(),
                serde_json::Value::Number((*kv_bytes).into()),
            );
            if let Some(rb) = recurrent_bytes {
                obj.insert(
                    "recurrent_bytes".to_owned(),
                    serde_json::Value::Number((*rb).into()),
                );
            }
        }
        // EXPLICIT: Exact / MlaSkipped carry no extra JSON fields.
        KvCachePath::Exact | KvCachePath::MlaSkipped => {}
    }
    serde_json::Value::Object(obj)
}

/// Same verdict, expressed as a [`serde_json::Value`] for the `--json` path.
///
/// Schema (annotations describe when each key is present — `serde_json` omits
/// the rest):
///
/// ```jsonc
/// {
///   "device_index": 0,                                                  // always present
///   "device": {                                                         // success only
///     "name": "NVIDIA GeForce RTX 5060 Ti",                             // when the backend reports a name
///     "total_bytes": 17179869184,                                       // always (inside `device`)
///     "free_bytes": 15246684160,
///     "used_bytes": 1933185024
///   },
///   "error": "no NVIDIA device detected (NVML / DXGI not usable)",      // failure only
///   "model": {                                                          // always present
///     "weight_bytes": 10240671744,
///     "dtype_label": "BF16",
///     "total_params": 5120000000,
///     "total_bytes": 11314153472                                        // with `--context`, computable: weights + KV
///   },
///   "kv_cache": {                                                       // with `--context` only
///     "context": 8192,
///     "path": "exact",                                                  // exact | sliding_window_capped | sliding_window_partial | mla_skipped | unavailable | hybrid
///     "elem_bytes": 2,                                                  // computable only
///     "bytes": 1073741824,                                              // computable only (total: KV + recurrent for hybrid)
///     "window": 4096,                                                   // sliding_window_* only
///     "reason": "no config.json in repo",                              // unavailable only
///     "attn_layers": 4,                                                 // hybrid only
///     "recurrent_layers": 36,                                           // hybrid only
///     "kv_bytes": 268435456,                                            // hybrid only (attention KV)
///     "recurrent_bytes": 29270016                                       // hybrid only, Mamba2 (excluded for DeltaNet / Mamba1)
///   },
///   "fits": true,                                                       // success only
///   "headroom_bytes": 4500000000,                                       // success + `fits == true`
///   "short_bytes": 2000000000                                           // success + `fits == false`
/// }
/// ```
///
/// `device` / `fits` / `headroom_bytes` / `short_bytes` are present only when
/// the probe succeeded ([`GpuCheckResult::device`] is `Some`); `error` is
/// present only when the probe failed. `headroom_bytes` and `short_bytes`
/// are mutually exclusive. With a computable KV estimate, `fits` / headroom /
/// short are measured against `model.total_bytes` (weights + KV); otherwise
/// against `weight_bytes` alone. Without `--context`, `kv_cache` and
/// `total_bytes` are absent and the schema is identical to prior releases.
#[must_use]
pub fn gpu_check_json(
    result: &GpuCheckResult,
    weight_bytes: u64,
    dtype_label: &str,
    total_params: u64,
    kv: Option<&KvComputed>,
) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    out.insert(
        "device_index".to_owned(),
        serde_json::Value::Number(result.device_index.into()),
    );

    // The footprint the fit verdict is measured against: weights + KV when a
    // computable estimate is present, weights alone otherwise.
    let kv_bytes = kv.and_then(|k| k.bytes);
    let total_bytes = kv_bytes.map_or(weight_bytes, |b| weight_bytes.saturating_add(b));

    if let Some(ref dev) = result.device {
        let mut dev_obj = serde_json::Map::new();
        if let Some(ref name) = dev.name {
            dev_obj.insert("name".to_owned(), serde_json::Value::String(name.clone()));
        }
        dev_obj.insert(
            "total_bytes".to_owned(),
            serde_json::Value::Number(dev.total_bytes.into()),
        );
        dev_obj.insert(
            "free_bytes".to_owned(),
            serde_json::Value::Number(dev.free_bytes.into()),
        );
        dev_obj.insert(
            "used_bytes".to_owned(),
            serde_json::Value::Number(dev.used_bytes.into()),
        );
        out.insert("device".to_owned(), serde_json::Value::Object(dev_obj));

        let fits = dev.free_bytes >= total_bytes;
        out.insert("fits".to_owned(), serde_json::Value::Bool(fits));
        if fits {
            out.insert(
                "headroom_bytes".to_owned(),
                serde_json::Value::Number((dev.free_bytes - total_bytes).into()),
            );
        } else {
            out.insert(
                "short_bytes".to_owned(),
                serde_json::Value::Number((total_bytes - dev.free_bytes).into()),
            );
        }
    }

    if let Some(ref msg) = result.error {
        out.insert("error".to_owned(), serde_json::Value::String(msg.clone()));
    }

    let mut model = serde_json::Map::new();
    model.insert(
        "weight_bytes".to_owned(),
        serde_json::Value::Number(weight_bytes.into()),
    );
    model.insert(
        "dtype_label".to_owned(),
        serde_json::Value::String(dtype_label.to_owned()),
    );
    model.insert(
        "total_params".to_owned(),
        serde_json::Value::Number(total_params.into()),
    );
    if kv_bytes.is_some() {
        model.insert(
            "total_bytes".to_owned(),
            serde_json::Value::Number(total_bytes.into()),
        );
    }
    out.insert("model".to_owned(), serde_json::Value::Object(model));

    if let Some(kv) = kv {
        out.insert("kv_cache".to_owned(), kv_cache_json(kv));
    }

    serde_json::Value::Object(out)
}

/// Compact parameter-count formatter (`635.4M`, `8.25B`, `1.20T`).
///
/// Mirrors the `inspect` table's existing column convention. Kept inline
/// rather than re-exported from `hf_fetch_model::inspect::format_params`
/// because that helper is private to the library; future cleanup can lift
/// either implementation into [`crate::format`].
fn format_params(n: u64) -> String {
    const M: u64 = 1_000_000;
    const B: u64 = 1_000_000_000;
    const T: u64 = 1_000_000_000_000;

    if n >= T {
        // CAST: u64 → f64, precision loss acceptable; display-only param count
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let v = n as f64 / T as f64;
        format!("{v:.2}T")
    } else if n >= B {
        // CAST: u64 → f64, precision loss acceptable; display-only param count
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let v = n as f64 / B as f64;
        format!("{v:.2}B")
    } else if n >= M {
        // CAST: u64 → f64, precision loss acceptable; display-only param count
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let v = n as f64 / M as f64;
        format!("{v:.1}M")
    } else if n >= 1_000 {
        // CAST: u64 → f64, precision loss acceptable; display-only param count
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let v = n as f64 / 1_000.0_f64;
        format!("{v:.1}K")
    } else {
        format!("{n}")
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::panic,
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing
    )]

    use super::{
        classify_hybrid_layers, dominant_dtype_label, format_params, gpu_check_json,
        kv_cache_bytes, mamba2_state_bytes_per_layer, sum_tensor_bytes, GpuCheckResult,
        KvCachePath, KvComputed,
    };
    use hf_fetch_model::inspect::{torch_dtype_bytes, ModelConfig, TensorInfo};
    use hypomnesis::GpuDeviceInfo;

    fn make_tensor(name: &str, dtype: &str, shape: Vec<usize>, byte_len: u64) -> TensorInfo {
        TensorInfo {
            name: name.to_owned(),
            dtype: dtype.to_owned(),
            shape,
            data_offsets: (0, byte_len),
        }
    }

    /// Builds a `ModelConfig` for the KV tests. `ModelConfig` is
    /// `#[non_exhaustive]` (it lives in the library crate), so struct-literal
    /// construction is unavailable here — fields are set on a `default()`
    /// instance instead.
    #[allow(clippy::field_reassign_with_default)] // EXPLICIT: non_exhaustive struct, literal unavailable
    fn model(layers: u32, attn: u32, kv: Option<u32>, head_dim: Option<u32>) -> ModelConfig {
        let mut c = ModelConfig::default();
        c.num_hidden_layers = Some(layers);
        c.num_attention_heads = Some(attn);
        c.num_key_value_heads = kv;
        c.head_dim = head_dim;
        c
    }

    // ---------- kv_cache_bytes ----------

    #[test]
    fn kv_exact_gqa_llama3_8b() {
        // Llama-3-8B: 32 layers, 32 attn heads, 8 KV heads, head_dim 128.
        // @ ctx 8192, bf16 (2 B): 2*32*8*128*8192*2 = 1 GiB exactly.
        let c = model(32, 32, Some(8), Some(128));
        let (bytes, path) = kv_cache_bytes(&c, 8192, 2);
        assert_eq!(bytes, Some(1024 * 1024 * 1024));
        assert_eq!(path, KvCachePath::Exact);
    }

    #[test]
    fn kv_mha_falls_back_to_attn_heads() {
        // No num_key_value_heads ⇒ MHA: kv_heads == attn_heads.
        // ppt = 2*4*64*2 = 1024; * 2 layers * 16 ctx = 32768.
        let c = model(2, 4, None, Some(64));
        let (bytes, path) = kv_cache_bytes(&c, 16, 2);
        assert_eq!(bytes, Some(32_768));
        assert_eq!(path, KvCachePath::Exact);
    }

    #[test]
    fn kv_derives_head_dim_from_hidden() {
        // head_dim absent ⇒ hidden_size / attn_heads = 256/8 = 32.
        // ppt = 2*8*32*2 = 1024; * 1 layer * 10 ctx = 10240.
        let mut c = model(1, 8, Some(8), None);
        c.hidden_size = Some(256);
        let (bytes, path) = kv_cache_bytes(&c, 10, 2);
        assert_eq!(bytes, Some(10_240));
        assert_eq!(path, KvCachePath::Exact);
    }

    #[test]
    fn kv_explicit_head_dim_preferred_over_derived() {
        // Gemma-2-style: explicit head_dim 256 ≠ hidden/heads (3584/16 = 224).
        // ppt = 2*8*256*2 = 8192; * 1 layer * 4 ctx = 32768 (uses 256, not 224).
        let mut c = model(1, 16, Some(8), Some(256));
        c.hidden_size = Some(3584);
        let (bytes, _) = kv_cache_bytes(&c, 4, 2);
        assert_eq!(bytes, Some(32_768));
    }

    #[test]
    fn kv_sliding_window_capped_when_context_exceeds() {
        // Uniform window 100; ctx 1000 > 100 ⇒ capped at 100.
        // ppt = 2*4*64*2 = 1024; * 2 layers * 100 = 204800.
        let mut c = model(2, 4, Some(4), Some(64));
        c.model_type = Some("mistral".to_owned());
        c.sliding_window = Some(100);
        let (bytes, path) = kv_cache_bytes(&c, 1000, 2);
        assert_eq!(bytes, Some(204_800));
        assert_eq!(path, KvCachePath::SlidingWindowCapped { window: 100 });
    }

    #[test]
    fn kv_sliding_window_below_does_not_cap() {
        // ctx 50 < window 100 ⇒ window doesn't bind ⇒ Exact.
        // ppt = 1024; * 2 layers * 50 = 102400.
        let mut c = model(2, 4, Some(4), Some(64));
        c.model_type = Some("mistral".to_owned());
        c.sliding_window = Some(100);
        let (bytes, path) = kv_cache_bytes(&c, 50, 2);
        assert_eq!(bytes, Some(102_400));
        assert_eq!(path, KvCachePath::Exact);
    }

    #[test]
    fn kv_sliding_window_partial_blend_gemma3() {
        // 12 layers, pattern 6 ⇒ global every 6th: 2 global, 10 local.
        // ppt = 1024; 1024 * (10*100 + 2*1000) = 1024 * 3000 = 3_072_000.
        let mut c = model(12, 4, Some(4), Some(64));
        c.model_type = Some("gemma3".to_owned());
        c.sliding_window = Some(100);
        c.sliding_window_pattern = Some(6);
        let (bytes, path) = kv_cache_bytes(&c, 1000, 2);
        assert_eq!(bytes, Some(3_072_000));
        assert_eq!(
            path,
            KvCachePath::SlidingWindowPartial {
                local_layers: 10,
                global_layers: 2,
                window: 100,
            }
        );
    }

    #[test]
    fn kv_gemma2_alternating_is_partial() {
        // Gemma-2: no sliding_window_pattern, detected by model_type ⇒ period 2.
        // 4 layers ⇒ global 2, local 2. ppt = 2*2*32*2 = 256.
        // 256 * (2*50 + 2*500) = 256 * 1100 = 281_600.
        let mut c = model(4, 2, Some(2), Some(32));
        c.model_type = Some("gemma2".to_owned());
        c.sliding_window = Some(50);
        let (bytes, path) = kv_cache_bytes(&c, 500, 2);
        assert_eq!(bytes, Some(281_600));
        assert_eq!(
            path,
            KvCachePath::SlidingWindowPartial {
                local_layers: 2,
                global_layers: 2,
                window: 50,
            }
        );
    }

    #[test]
    fn kv_use_sliding_window_false_is_full_attention() {
        // Qwen2/3 ship a window but disable it ⇒ full attention.
        // ppt = 1024; * 2 layers * 1000 = 2_048_000.
        let mut c = model(2, 4, Some(4), Some(64));
        c.sliding_window = Some(100);
        c.use_sliding_window = Some(false);
        let (bytes, path) = kv_cache_bytes(&c, 1000, 2);
        assert_eq!(bytes, Some(2_048_000));
        assert_eq!(path, KvCachePath::Exact);
    }

    #[test]
    fn kv_mla_skipped() {
        let mut c = model(27, 16, Some(16), None);
        c.model_type = Some("deepseek_v2".to_owned());
        c.kv_lora_rank = Some(512);
        c.qk_rope_head_dim = Some(64);
        let (bytes, path) = kv_cache_bytes(&c, 4096, 2);
        assert_eq!(bytes, None);
        assert_eq!(path, KvCachePath::MlaSkipped);
    }

    #[test]
    fn kv_unavailable_missing_dims() {
        let c = ModelConfig::default();
        let (bytes, path) = kv_cache_bytes(&c, 4096, 2);
        assert_eq!(bytes, None);
        assert!(matches!(path, KvCachePath::Unavailable { .. }));
    }

    #[test]
    fn kv_elem_bytes_scales_linearly() {
        // fp8 KV halves the figure vs bf16.
        let c = model(1, 1, Some(1), Some(1));
        let (bf16, _) = kv_cache_bytes(&c, 100, 2);
        let (fp8, _) = kv_cache_bytes(&c, 100, 1);
        assert_eq!(bf16, Some(400)); // 2*1*1*2*1*100
        assert_eq!(fp8, Some(200)); // half
    }

    #[test]
    fn torch_dtype_bytes_mapping() {
        assert_eq!(torch_dtype_bytes(Some("bfloat16")), 2);
        assert_eq!(torch_dtype_bytes(Some("float16")), 2);
        assert_eq!(torch_dtype_bytes(Some("float32")), 4);
        assert_eq!(torch_dtype_bytes(Some("float8_e4m3fn")), 1);
        assert_eq!(torch_dtype_bytes(Some("mystery")), 2);
        assert_eq!(torch_dtype_bytes(None), 2);
    }

    #[test]
    fn kv_cache_path_json_tags() {
        assert_eq!(KvCachePath::Exact.json_tag(), "exact");
        assert_eq!(KvCachePath::MlaSkipped.json_tag(), "mla_skipped");
        assert_eq!(
            KvCachePath::SlidingWindowCapped { window: 1 }.json_tag(),
            "sliding_window_capped"
        );
        assert_eq!(
            KvCachePath::Unavailable { reason: "x" }.json_tag(),
            "unavailable"
        );
    }

    // ---------- hybrid Mamba/attention ----------

    #[test]
    fn classify_hybrid_via_layer_types() {
        let mut c = model(4, 8, Some(2), Some(64));
        c.layer_types = Some(vec![
            "mamba".to_owned(),
            "attention".to_owned(),
            "mamba".to_owned(),
            "mamba".to_owned(),
        ]);
        let h = classify_hybrid_layers(&c).expect("hybrid");
        assert_eq!(h.attn_layers, 1);
        assert_eq!(h.recurrent_layers, 3);
    }

    #[test]
    fn classify_hybrid_via_override_pattern() {
        // Nemotron-H: '*' = attention, 'M' = mamba, '-' = FFN (ignored).
        // "M-M-M*-M-M*-" has 5 'M' and 2 '*'.
        let mut c = model(8, 32, Some(8), Some(128));
        c.hybrid_override_pattern = Some("M-M-M*-M-M*-".to_owned());
        let h = classify_hybrid_layers(&c).expect("hybrid");
        assert_eq!(h.attn_layers, 2);
        assert_eq!(h.recurrent_layers, 5);
    }

    #[test]
    fn classify_hybrid_via_attn_indices() {
        let mut c = model(32, 32, Some(8), Some(128));
        c.attn_layer_indices = Some(vec![9, 18, 27]);
        let h = classify_hybrid_layers(&c).expect("hybrid");
        assert_eq!(h.attn_layers, 3);
        assert_eq!(h.recurrent_layers, 29);
    }

    #[test]
    fn classify_hybrid_via_full_attention_interval() {
        let mut c = model(48, 16, Some(2), Some(256));
        c.full_attention_interval = Some(4);
        let h = classify_hybrid_layers(&c).expect("hybrid");
        assert_eq!(h.attn_layers, 12);
        assert_eq!(h.recurrent_layers, 36);
    }

    #[test]
    fn classify_pure_transformer_layer_types_is_not_hybrid() {
        // Gemma-3-style: layer_types all attention ⇒ not hybrid.
        let mut c = model(4, 8, Some(2), Some(64));
        c.layer_types = Some(vec![
            "sliding_attention".to_owned(),
            "full_attention".to_owned(),
            "sliding_attention".to_owned(),
            "full_attention".to_owned(),
        ]);
        assert!(classify_hybrid_layers(&c).is_none());
    }

    #[test]
    fn classify_non_hybrid_is_none() {
        let c = model(32, 32, Some(8), Some(128));
        assert!(classify_hybrid_layers(&c).is_none());
    }

    #[test]
    fn mamba2_state_granite_dims() {
        // n_heads 48, d_head 64, d_state 128, d_conv 4, n_groups 1.
        // inner = 3072; ssm = 393216; conv_dim = 3328; conv = 13312.
        // (393216 + 13312) * 2 = 813056.
        let mut c = ModelConfig::default();
        c.mamba_n_heads = Some(48);
        c.mamba_d_head = Some(64);
        c.mamba_d_state = Some(128);
        c.mamba_d_conv = Some(4);
        c.mamba_n_groups = Some(1);
        assert_eq!(mamba2_state_bytes_per_layer(&c, 2), Some(813_056));
    }

    #[test]
    fn mamba2_state_missing_dims_is_none() {
        // Qwen3-Next (no mamba keys) ⇒ None ⇒ recurrent excluded.
        let c = ModelConfig::default();
        assert_eq!(mamba2_state_bytes_per_layer(&c, 2), None);
    }

    #[test]
    fn kv_hybrid_counts_attn_layers_only_plus_recurrent() {
        // Granite-4-tiny-like: 40 layers (4 attention, 36 mamba), kv_heads 4,
        // head_dim 128, ctx 8192, bf16.
        let mut c = model(40, 12, Some(4), Some(128));
        let mut types = vec!["mamba".to_owned(); 36];
        types.extend(vec!["attention".to_owned(); 4]);
        c.layer_types = Some(types);
        c.mamba_n_heads = Some(48);
        c.mamba_d_head = Some(64);
        c.mamba_d_state = Some(128);
        c.mamba_d_conv = Some(4);
        c.mamba_n_groups = Some(1);
        let (bytes, path) = kv_cache_bytes(&c, 8192, 2);
        // KV: per_layer 2*4*128*2 = 2048; * 4 attn * 8192 = 67_108_864.
        // recurrent: 813_056 * 36.
        assert_eq!(bytes, Some(67_108_864 + 813_056 * 36));
        let KvCachePath::Hybrid {
            attn_layers,
            recurrent_layers,
            kv_bytes,
            recurrent_bytes,
        } = path
        else {
            panic!("expected Hybrid, got {path:?}");
        };
        assert_eq!(attn_layers, 4);
        assert_eq!(recurrent_layers, 36);
        assert_eq!(kv_bytes, 67_108_864);
        assert_eq!(recurrent_bytes, Some(813_056 * 36));
    }

    #[test]
    fn kv_hybrid_recurrent_excluded_when_not_mamba2() {
        // Qwen3-Next-like: full_attention_interval 4 (12 attn / 36 linear), no
        // mamba keys ⇒ recurrent excluded.
        let mut c = model(48, 16, Some(2), Some(256));
        c.full_attention_interval = Some(4);
        let (bytes, path) = kv_cache_bytes(&c, 8192, 2);
        // KV: per_layer 2*2*256*2 = 2048; * 12 attn * 8192 = 201_326_592.
        assert_eq!(bytes, Some(201_326_592));
        let KvCachePath::Hybrid {
            attn_layers,
            recurrent_layers,
            kv_bytes,
            recurrent_bytes,
        } = path
        else {
            panic!("expected Hybrid, got {path:?}");
        };
        assert_eq!(attn_layers, 12);
        assert_eq!(recurrent_layers, 36);
        assert_eq!(kv_bytes, 201_326_592);
        assert_eq!(recurrent_bytes, None);
    }

    #[test]
    fn kv_cache_path_hybrid_json_tag() {
        let path = KvCachePath::Hybrid {
            attn_layers: 4,
            recurrent_layers: 36,
            kv_bytes: 1,
            recurrent_bytes: None,
        };
        assert_eq!(path.json_tag(), "hybrid");
    }

    #[test]
    fn sum_tensor_bytes_empty() {
        assert_eq!(sum_tensor_bytes(&[]), 0);
    }

    #[test]
    fn sum_tensor_bytes_one_tensor() {
        let t = make_tensor("a", "BF16", vec![10, 10], 200);
        assert_eq!(sum_tensor_bytes(&[t]), 200);
    }

    #[test]
    fn sum_tensor_bytes_many() {
        let a = make_tensor("a", "BF16", vec![4, 4], 32);
        let b = make_tensor("b", "F16", vec![4, 4], 32);
        let c = make_tensor("c", "F32", vec![4, 4], 64);
        assert_eq!(sum_tensor_bytes(&[a, b, c]), 128);
    }

    #[test]
    fn dominant_dtype_label_empty() {
        assert_eq!(dominant_dtype_label(&[]), "unknown");
    }

    #[test]
    fn dominant_dtype_label_pure() {
        let tensors = vec![
            make_tensor("a", "BF16", vec![1000, 1000], 2_000_000),
            make_tensor("b", "BF16", vec![1000, 1000], 2_000_000),
        ];
        assert_eq!(dominant_dtype_label(&tensors), "BF16");
    }

    #[test]
    fn dominant_dtype_label_near_pure_collapses_to_dominant() {
        // 99.9% BF16 by params, 0.1% F32 — should report as pure BF16.
        let tensors = vec![
            make_tensor("weight", "BF16", vec![1000, 1000], 2_000_000),
            make_tensor("norm", "F32", vec![10], 40),
        ];
        assert_eq!(dominant_dtype_label(&tensors), "BF16");
    }

    #[test]
    fn dominant_dtype_label_true_mixed_flags_others() {
        // 60% BF16, 40% F8_E4M3 — should be "BF16 + others".
        let tensors = vec![
            make_tensor("a", "BF16", vec![60, 100], 12_000),
            make_tensor("b", "F8_E4M3", vec![40, 100], 4_000),
        ];
        assert_eq!(dominant_dtype_label(&tensors), "BF16 + others");
    }

    #[test]
    fn format_params_buckets() {
        assert_eq!(format_params(0), "0");
        assert_eq!(format_params(999), "999");
        assert_eq!(format_params(1_500), "1.5K");
        assert_eq!(format_params(2_000_000), "2.0M");
        assert_eq!(format_params(5_120_000_000), "5.12B");
        assert_eq!(format_params(1_500_000_000_000), "1.50T");
    }

    #[test]
    fn gpu_check_json_error_path() {
        let result = GpuCheckResult {
            device_index: 3,
            device: None,
            error: Some("index 3 out of range (have 1 device)".to_owned()),
        };
        let v = gpu_check_json(&result, 1024, "BF16", 100, None);
        assert_eq!(v.get("device_index"), Some(&serde_json::json!(3)));
        assert_eq!(
            v.get("error"),
            Some(&serde_json::json!("index 3 out of range (have 1 device)"))
        );
        assert!(v.get("device").is_none());
        assert!(v.get("fits").is_none());
        let model = v.get("model").expect("model object present");
        assert_eq!(model.get("weight_bytes"), Some(&serde_json::json!(1024)));
        assert_eq!(model.get("dtype_label"), Some(&serde_json::json!("BF16")));
        assert_eq!(model.get("total_params"), Some(&serde_json::json!(100)));
    }

    #[test]
    fn gpu_check_json_fit_path() {
        // Free 14 GiB ≥ 10 GiB weight → fits: true with 4 GiB headroom.
        let device = GpuDeviceInfo::builder()
            .index(0)
            .name(Some("NVIDIA GeForce RTX 5060 Ti".to_owned()))
            .total_bytes(16 * 1024 * 1024 * 1024)
            .free_bytes(14 * 1024 * 1024 * 1024)
            .used_bytes(2 * 1024 * 1024 * 1024)
            .build();
        let result = GpuCheckResult {
            device_index: 0,
            device: Some(device),
            error: None,
        };
        let weight_bytes: u64 = 10 * 1024 * 1024 * 1024;
        let v = gpu_check_json(&result, weight_bytes, "BF16", 5_120_000_000, None);

        assert_eq!(v.get("device_index"), Some(&serde_json::json!(0)));
        assert!(v.get("error").is_none());
        assert_eq!(v.get("fits"), Some(&serde_json::json!(true)));
        assert_eq!(
            v.get("headroom_bytes"),
            Some(&serde_json::json!(4u64 * 1024 * 1024 * 1024))
        );
        assert!(v.get("short_bytes").is_none());

        let device_obj = v.get("device").expect("device object present");
        assert_eq!(
            device_obj.get("name"),
            Some(&serde_json::json!("NVIDIA GeForce RTX 5060 Ti"))
        );
        assert_eq!(
            device_obj.get("total_bytes"),
            Some(&serde_json::json!(16u64 * 1024 * 1024 * 1024))
        );
        assert_eq!(
            device_obj.get("free_bytes"),
            Some(&serde_json::json!(14u64 * 1024 * 1024 * 1024))
        );
        assert_eq!(
            device_obj.get("used_bytes"),
            Some(&serde_json::json!(2u64 * 1024 * 1024 * 1024))
        );
    }

    #[test]
    fn gpu_check_json_miss_path() {
        // Free 13.5 GiB < 25 GiB weight → fits: false, short by 11.5 GiB.
        // Backend reported no adapter name (e.g. nvidia-smi fallback) — `device.name`
        // must be absent from the JSON in this branch.
        let device = GpuDeviceInfo::builder()
            .index(0)
            .name(None)
            .total_bytes(16 * 1024 * 1024 * 1024)
            .free_bytes(13_u64 * 1024 * 1024 * 1024 + 512 * 1024 * 1024)
            .used_bytes(2_u64 * 1024 * 1024 * 1024 + 512 * 1024 * 1024)
            .build();
        let result = GpuCheckResult {
            device_index: 0,
            device: Some(device),
            error: None,
        };
        let weight_bytes: u64 = 25 * 1024 * 1024 * 1024;
        let v = gpu_check_json(&result, weight_bytes, "U8 + others", 23_910_000_000, None);

        assert_eq!(v.get("fits"), Some(&serde_json::json!(false)));
        assert!(v.get("headroom_bytes").is_none());
        // 25 GiB - 13.5 GiB = 11.5 GiB
        assert_eq!(
            v.get("short_bytes"),
            Some(&serde_json::json!(
                11_u64 * 1024 * 1024 * 1024 + 512 * 1024 * 1024
            ))
        );

        let device_obj = v.get("device").expect("device object present");
        // `name` absent when the backend didn't report one — serde_json omits the key.
        assert!(device_obj.get("name").is_none());
        assert_eq!(
            device_obj.get("total_bytes"),
            Some(&serde_json::json!(16u64 * 1024 * 1024 * 1024))
        );
    }

    #[test]
    fn gpu_check_json_with_kv_fits_against_total() {
        // Free 14 GiB; weights 10 GiB + KV 2 GiB = 12 GiB → fits, 2 GiB headroom.
        let device = GpuDeviceInfo::builder()
            .index(0)
            .name(Some("Test GPU".to_owned()))
            .total_bytes(16 * 1024 * 1024 * 1024)
            .free_bytes(14 * 1024 * 1024 * 1024)
            .used_bytes(2 * 1024 * 1024 * 1024)
            .build();
        let result = GpuCheckResult {
            device_index: 0,
            device: Some(device),
            error: None,
        };
        let weight_bytes: u64 = 10 * 1024 * 1024 * 1024;
        let kv = KvComputed {
            context: 8192,
            elem_bytes: 2,
            dtype_label: "BF16".to_owned(),
            bytes: Some(2 * 1024 * 1024 * 1024),
            path: KvCachePath::Exact,
        };
        let v = gpu_check_json(&result, weight_bytes, "BF16", 5_000_000_000, Some(&kv));

        // Fit is measured against weights + KV (12 GiB), not weights alone.
        assert_eq!(v.get("fits"), Some(&serde_json::json!(true)));
        assert_eq!(
            v.get("headroom_bytes"),
            Some(&serde_json::json!(2u64 * 1024 * 1024 * 1024))
        );
        let model = v.get("model").expect("model present");
        assert_eq!(
            model.get("total_bytes"),
            Some(&serde_json::json!(12u64 * 1024 * 1024 * 1024))
        );
        let kvc = v.get("kv_cache").expect("kv_cache present");
        assert_eq!(kvc.get("path"), Some(&serde_json::json!("exact")));
        assert_eq!(kvc.get("context"), Some(&serde_json::json!(8192)));
        assert_eq!(
            kvc.get("bytes"),
            Some(&serde_json::json!(2u64 * 1024 * 1024 * 1024))
        );
    }

    #[test]
    fn gpu_check_json_without_kv_omits_kv_cache() {
        // Without --context the schema is unchanged: no kv_cache, no total_bytes.
        let device = GpuDeviceInfo::builder()
            .index(0)
            .name(None)
            .total_bytes(16 * 1024 * 1024 * 1024)
            .free_bytes(14 * 1024 * 1024 * 1024)
            .used_bytes(2 * 1024 * 1024 * 1024)
            .build();
        let result = GpuCheckResult {
            device_index: 0,
            device: Some(device),
            error: None,
        };
        let v = gpu_check_json(
            &result,
            10 * 1024 * 1024 * 1024,
            "BF16",
            5_000_000_000,
            None,
        );
        assert!(v.get("kv_cache").is_none());
        let model = v.get("model").expect("model present");
        assert!(model.get("total_bytes").is_none());
    }
}
