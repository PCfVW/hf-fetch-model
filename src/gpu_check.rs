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

use hf_fetch_model::inspect::TensorInfo;
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
pub fn print_gpu_check(
    result: &GpuCheckResult,
    weight_bytes: u64,
    dtype_label: &str,
    total_params: u64,
) {
    println!();
    println!(
        "  Model weights:  {}  ({dtype_label}, {} params)",
        format_size(weight_bytes),
        format_params(total_params),
    );

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

    // BORROW: explicit .as_deref().unwrap_or for Option<String> → &str with fallback
    let name = dev.name.as_deref().unwrap_or("unknown GPU");
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

    if dev.free_bytes >= weight_bytes {
        let headroom = dev.free_bytes - weight_bytes;
        println!(
            "  Fit:            \u{2713} {} headroom for weights + KV cache + runtime",
            format_size(headroom),
        );
    } else {
        let short = weight_bytes - dev.free_bytes;
        println!(
            "  Fit:            \u{2717} short by {} for the weights alone",
            format_size(short),
        );
    }

    println!();
    println!(
        "  Note: reports weights only. Large-context inference typically needs ~1.3\u{2013}1.5\u{00d7}"
    );
    println!("  weight size for KV cache and activations.");
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
///     "total_params": 5120000000
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
/// are mutually exclusive — exactly one of them appears alongside `fits`.
#[must_use]
pub fn gpu_check_json(
    result: &GpuCheckResult,
    weight_bytes: u64,
    dtype_label: &str,
    total_params: u64,
) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    out.insert(
        "device_index".to_owned(),
        serde_json::Value::Number(result.device_index.into()),
    );

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

        let fits = dev.free_bytes >= weight_bytes;
        out.insert("fits".to_owned(), serde_json::Value::Bool(fits));
        if fits {
            out.insert(
                "headroom_bytes".to_owned(),
                serde_json::Value::Number((dev.free_bytes - weight_bytes).into()),
            );
        } else {
            out.insert(
                "short_bytes".to_owned(),
                serde_json::Value::Number((weight_bytes - dev.free_bytes).into()),
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
    out.insert("model".to_owned(), serde_json::Value::Object(model));

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
        dominant_dtype_label, format_params, gpu_check_json, sum_tensor_bytes, GpuCheckResult,
    };
    use hf_fetch_model::inspect::TensorInfo;

    fn make_tensor(name: &str, dtype: &str, shape: Vec<usize>, byte_len: u64) -> TensorInfo {
        TensorInfo {
            name: name.to_owned(),
            dtype: dtype.to_owned(),
            shape,
            data_offsets: (0, byte_len),
        }
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
        let v = gpu_check_json(&result, 1024, "BF16", 100);
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

    // EXPLICIT: Fit / miss path JSON cannot be directly unit-tested today —
    // `hypomnesis::GpuDeviceInfo` is `#[non_exhaustive]`, so external crates
    // cannot construct a test fixture. The arithmetic (`free >= weight` →
    // `fits`, `headroom = free - weight` on hit, `short = weight - free` on
    // miss) is exercised end-to-end by the manual smoke tests on the
    // maintainer's RTX 5060 Ti (zeta-2 misses, gemma-4-E2B-it fits). See
    // `docs/dogfooding-feedbacks/hypomnesis-adoption.md` for the recommended
    // upstream helper that would unblock these tests.
}
