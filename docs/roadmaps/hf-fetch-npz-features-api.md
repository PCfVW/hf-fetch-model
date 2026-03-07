# Design: `hf-fetch-model` NPZ/NPY Parsing and Download API

**Status:** Proposed
**Date:** March 6, 2026
**Relates to:** candle-mi Phase 4 (SAE Support), Gemma Scope NPZ weight loading

## Question

Should `hf-fetch-model` provide NPZ/NPY parsing as a framework-agnostic feature, and what should the API look like?

## Motivation

candle-mi v0.0.5 includes an NPZ/NPY parser in `src/sae/npz.rs` for loading Gemma Scope SAE weights. This parser is tightly coupled to `candle_core::Tensor` — it returns `HashMap<String, Tensor>` — but the actual parsing logic (binary format decoding, header parsing, shape extraction) is entirely framework-agnostic.

Moving the parsing layer into `hf-fetch-model` provides three benefits:

- **Eliminates the candle dependency boundary.** hf-fetch-model stays a pure download/caching/parsing utility with no ML framework dependency. candle-mi (or any other consumer) converts the parsed arrays to its own tensor type.

- **Combines download + parse in one step.** hf-fetch-model already provides fast multi-connection downloads, caching, retries, SHA256 verification, and progress reporting. A `download_and_parse_npz` convenience function composes these capabilities with NPZ parsing, reducing boilerplate in candle-mi's SAE loading path.

- **Single source of truth.** The NPZ parser lives in one crate. candle-mi depends on hf-fetch-model (already the case) and uses a thin adapter to convert `NpyArray` to `Tensor`.

### Why NPZ matters

NPZ (NumPy ZIP archive) is the weight format used by **Gemma Scope**, Google's open SAE suite for Gemma 2 models. Each `params.npz` file contains 4–5 `.npy` arrays (`W_enc`, `W_dec`, `b_enc`, `b_dec`, optionally `threshold`). File sizes range from 100 MB to 600+ MB depending on the SAE's dictionary size (`d_sae`), making them natural candidates for chunked parallel download.

## New dependency

```toml
# Cargo.toml
zip = { version = "2", default-features = false, features = ["deflate"] }
```

The `zip` crate with only the `deflate` feature pulls in `flate2` (pure-Rust `miniz_oxide` backend). No C dependencies, no `unsafe`, no encryption, no write support. This is the minimum required to read deflate-compressed NPZ archives, which is the NumPy default compression method.

## Feature 1: NPZ/NPY parser (`src/npz.rs`)

### Public types

```rust
/// Supported NPY numeric dtypes.
#[non_exhaustive]
pub enum NpyDtype {
    /// 32-bit IEEE 754 float (`<f4`).
    F32,
    /// 64-bit IEEE 754 float (`<f8`).
    F64,
}

/// A parsed NPY array: shape, dtype, and raw data bytes.
///
/// This is a framework-agnostic representation. Consumers convert
/// to their own tensor type (e.g., candle `Tensor`, ndarray `Array`).
pub struct NpyArray {
    /// Tensor shape (e.g., `[2304, 16384]`).
    pub shape: Vec<usize>,
    /// Element dtype.
    pub dtype: NpyDtype,
    /// Raw data bytes in the source dtype's little-endian format.
    pub data: Vec<u8>,
}
```

### `NpyArray` methods

```rust
impl NpyArray {
    /// Number of elements (product of shape dimensions).
    pub fn len(&self) -> usize;

    /// Whether the array has zero elements.
    pub fn is_empty(&self) -> bool;

    /// Bytes per element (4 for F32, 8 for F64).
    pub fn bytes_per_element(&self) -> usize;

    /// Consume the array and return f32 bytes (little-endian).
    ///
    /// If the source dtype is F64, each element is converted to f32
    /// (precision loss is expected and acceptable for ML weights).
    /// If the source dtype is already F32, the data is returned as-is
    /// with no allocation.
    pub fn into_f32_bytes(self) -> Vec<u8>;
}
```

### Public functions

```rust
/// Parse a single `.npy` byte stream into an `NpyArray`.
///
/// Supports NPY format versions 1.0 and 2.0, float32 and float64 dtypes,
/// C-order (row-major) layout only.
pub fn parse_npy(bytes: &[u8]) -> Result<NpyArray, FetchError>;

/// Parse all `.npy` entries from an NPZ (ZIP) archive on disk.
///
/// Returns a map from array name (`.npy` suffix stripped) to parsed array.
/// Non-`.npy` entries in the archive are silently skipped.
pub fn load_npz(path: &Path) -> Result<HashMap<String, NpyArray>, FetchError>;
```

All internal helpers (`parse_npy_header`, `split_dict_entries`, `parse_shape_tuple`) remain private.

### Supported NPY subset

| Feature | Supported | Notes |
|---------|-----------|-------|
| NPY v1.0 (2-byte header length) | Yes | |
| NPY v2.0 (4-byte header length) | Yes | |
| NPY v3.0 | Yes | Same binary layout as v2.0 |
| `float32` (`<f4`, `=f4`) | Yes | |
| `float64` (`<f8`, `=f8`) | Yes | Promoted to f32 via `into_f32_bytes` |
| C-order (row-major) | Yes | |
| Fortran-order (column-major) | No | Returns error |
| Integer dtypes | No | See [Recommendations](#recommendations) |
| Structured dtypes | No | Out of scope |

### NPY binary format reference

```text
Offset  Size    Field
0       6       Magic: \x93NUMPY
6       1       Major version (1, 2, or 3)
7       1       Minor version (0)
8       2|4     Header length (LE): 2 bytes for v1, 4 bytes for v2/v3
10|12   N       Header: Python dict literal (ASCII), padded to 64-byte alignment
10+N|12+N ...   Raw data (contiguous, dtype-sized elements, little-endian)
```

Header example: `{'descr': '<f4', 'fortran_order': False, 'shape': (2304, 16384), }`

## Feature 2: New error variant

A new `FetchError` variant for parse failures:

```rust
pub enum FetchError {
    // ... existing variants ...

    /// An NPZ/NPY file could not be parsed.
    #[error("NPZ parse error: {reason}")]
    NpzParse {
        /// Description of the parse failure.
        reason: String,
    },
}
```

This replaces the `MIError::Config` variant used in candle-mi's current parser, giving NPZ errors their own identity in the error hierarchy.

## Feature 3: Download + parse convenience function

### Motivation

Gemma Scope SAE weights live on HuggingFace (e.g., `google/gemma-scope-2b-pt-res`). Today, loading them requires two steps: download the file with `download_file`, then parse it with `load_npz`. A convenience function composes these into a single call.

### Behavior

```text
download_and_parse_npz(repo_id, filename, config)
  |
  +-- download_file(repo_id, filename, config)
  |     |-- Cache check (skip if already cached + checksum valid)
  |     |-- If file >= chunk_threshold (default 100 MiB):
  |     |     +-- Chunked parallel download (8 HTTP Range connections)
  |     |-- If file < chunk_threshold:
  |     |     +-- Single-connection download
  |     |-- Retry with exponential backoff (3 retries, 300ms base, 10s cap)
  |     |-- SHA256 verification against HF LFS metadata
  |     +-- Return local cache path
  |
  +-- spawn_blocking: load_npz(cached_path)
  |     +-- Parse ZIP archive, decode each .npy entry
  |
  +-- Return HashMap<String, NpyArray>
```

The `download_file` step provides full multi-connection chunked download for large NPZ files. Gemma Scope `params.npz` files (100–600+ MB) always exceed the 100 MiB chunk threshold, so they automatically get 8 parallel HTTP Range connections. The parse step runs inside `spawn_blocking` to avoid blocking the async runtime (consistent with how `checksum.rs` handles CPU-bound SHA256 computation).

### Proposed API

```rust
/// Download an NPZ file from a HuggingFace repository and parse it.
///
/// Combines `download_file` (cached, chunked, verified) with `load_npz`
/// (ZIP extraction + NPY parsing). Returns a map from array name to
/// parsed `NpyArray`.
///
/// # Arguments
///
/// * `repo_id` -- Repository identifier (e.g., `"google/gemma-scope-2b-pt-res"`).
/// * `filename` -- NPZ filename within the repository (e.g., `"layer_5/params.npz"`).
/// * `config` -- Shared configuration for auth, progress, checksums, retries, and chunking.
///
/// # Errors
///
/// Returns `FetchError::Http` if the file does not exist in the repository.
/// Returns `FetchError::Api` on download failure (after retries).
/// Returns `FetchError::Checksum` if verification is enabled and fails.
/// Returns `FetchError::NpzParse` if the NPZ/NPY content is malformed.
pub async fn download_and_parse_npz(
    repo_id: String,
    filename: String,
    config: &FetchConfig,
) -> Result<HashMap<String, NpyArray>, FetchError>;

/// Blocking variant of [`download_and_parse_npz`].
///
/// Creates a Tokio runtime internally. Do **not** call from within an
/// existing async context.
pub fn download_and_parse_npz_blocking(
    repo_id: String,
    filename: String,
    config: &FetchConfig,
) -> Result<HashMap<String, NpyArray>, FetchError>;
```

### Consumer-side usage sketch

How candle-mi's `Sae::from_npz` would simplify with the new API:

```rust
// Before (candle-mi v0.0.5): two crates, manual path wiring
let config = FetchConfig::builder().build()?;
let path = hf_fetch_model::download_file_blocking(
    repo_id.to_owned(), "layer_5/params.npz", &config,
)?;
// candle-mi's own npz::load_npz returns HashMap<String, Tensor>
let tensors = npz::load_npz(&path, device)?;

// After: one call, framework-agnostic arrays
let config = FetchConfig::builder().build()?;
let arrays = hf_fetch_model::download_and_parse_npz_blocking(
    repo_id.to_owned(),
    "layer_5/params.npz".to_owned(),
    &config,
)?;
// candle-mi converts NpyArray -> Tensor (thin adapter, ~10 lines)
let w_enc = Tensor::from_raw_buffer(
    &arrays.get("W_enc").unwrap().into_f32_bytes(),
    DType::F32,
    &arrays.get("W_enc").unwrap().shape,
    device,
)?;
```

## Feature 4: NPZ filter preset

A new filter preset in `FetchConfigBuilder`, alongside the existing `safetensors_only()` and `gguf_only()`:

```rust
impl FetchConfigBuilder {
    /// Include only `.npz` files (e.g., Gemma Scope SAE weight archives).
    pub fn npz_only(self) -> Self;
}
```

This is useful when a repository contains both NPZ weight files and other artifacts (documentation, scripts, config files) and the caller wants to download only the weight archives.

## Exports in `src/lib.rs`

```rust
pub mod npz;
pub use npz::{NpyArray, NpyDtype, load_npz, parse_npy};

// download_and_parse_npz[_blocking] alongside existing download_* functions
```

## File size analysis for Gemma Scope NPZ files

The `chunk_threshold` default of 100 MiB naturally applies to Gemma Scope weight files:

| SAE config | `d_in` | `d_sae` | Approx `params.npz` size | Chunked? |
|------------|--------|---------|--------------------------|----------|
| width 16k | 2304 | 16384 | ~150 MB | **Yes** |
| width 32k | 2304 | 32768 | ~300 MB | **Yes** |
| width 65k | 2304 | 65536 | ~600 MB | **Yes** |
| width 131k | 2304 | 131072 | ~1.2 GB | **Yes** |

All practical Gemma Scope files exceed the chunked threshold, so every download benefits from 8 parallel HTTP Range connections.

## Advantages over candle-mi's current approach

| Advantage | Detail |
|-----------|--------|
| **No candle dependency in parser** | hf-fetch-model stays framework-agnostic; any Rust project can parse NPZ files |
| **Single source of truth** | NPZ parsing logic lives in one crate, not duplicated across consumers |
| **Faster downloads** | `download_and_parse_npz` automatically uses chunked parallel transfer for large files |
| **Cache integration** | Parsed NPZ files are cached in the standard HF layout; re-parsing skips the download |
| **Unified error handling** | `FetchError::NpzParse` integrates cleanly with existing `FetchError` variants |
| **Progress reporting** | Download progress works identically to model downloads via `FetchConfig::on_progress` |

## Tests

### Unit tests (moved from candle-mi, adapted)

- `parse_header_basic` — parse a standard Python dict header with 2D shape
- `parse_header_1d` — parse a 1D shape tuple `(16384,)`
- `parse_header_f64` — parse an f64 dtype descriptor
- `parse_shape_tuple_basic` — parse various shape tuples
- `roundtrip_f32_npy` — build a minimal NPY v1 file in memory, parse it, verify shape and data (no candle dependency; checks raw bytes instead of `Tensor`)
- `roundtrip_f64_to_f32` — build an f64 NPY file, parse it, call `into_f32_bytes`, verify promoted values
- `npy_array_len` — verify `len()` and `is_empty()` on various shapes

### Integration tests

- Download a known small NPZ from HuggingFace and parse it (gated behind a network-dependent feature flag or `#[ignore]`)

## Recommendations

The following additions are not required for the initial implementation but are worth considering for robustness and future-proofing.

### 1. Feature-gate the NPZ module

An `npz` cargo feature flag would keep the `zip` dependency optional for users who only need the download functionality:

```toml
[features]
default = []
npz = ["dep:zip"]

[dependencies]
zip = { version = "2", default-features = false, features = ["deflate"], optional = true }
```

This mirrors how `cli` and `indicatif` are already gated. Users who only call `download()` or `download_file()` pay no compile-time cost for NPZ parsing. The `download_and_parse_npz` functions would also be gated behind `#[cfg(feature = "npz")]`.

### 2. Integer dtype support

The current parser only handles `<f4` and `<f8`. Some NPZ archives contain metadata arrays with integer dtypes:

| Dtype descriptor | Rust type | Use case |
|-----------------|-----------|----------|
| `<i4` / `=i4` | `i32` | Index arrays, sparse indices |
| `<i8` / `=i8` | `i64` | Large index arrays |
| `<u4` / `=u4` | `u32` | Shape metadata, counts |

Adding these to `NpyDtype` now (a few extra match arms in `parse_npy`) avoids a breaking change to the `#[non_exhaustive]` enum later. The `into_f32_bytes` method would not apply to integer dtypes — consumers would read the raw bytes directly or use a separate conversion method.

### 3. Zero-copy f32 view

For large arrays that are already f32, `into_f32_bytes` still moves the `Vec<u8>`. A borrowing alternative avoids the move:

```rust
impl NpyArray {
    /// View the raw data as an `&[f32]` slice if the dtype is F32 and
    /// the data is properly aligned. Returns `None` for F64 arrays or
    /// if alignment requirements are not met.
    pub fn as_f32_slice(&self) -> Option<&[f32]>;
}
```

This requires a careful alignment check (`data.as_ptr() as usize % 4 == 0`), but `Vec<u8>` allocated by the standard allocator is typically 8-byte aligned on 64-bit platforms, so the check would almost always pass. This avoids a redundant copy for the common case of f32 SAE weights loaded from cache.

### 4. CLI subcommand

A `parse-npz` CLI subcommand could print array names, shapes, and dtypes from a local or remote NPZ file — useful for inspecting Gemma Scope repositories without writing code:

```
hf-fm parse-npz google/gemma-scope-2b-pt-res layer_5/params.npz
```

Output:

```
W_enc: float32 [2304, 16384]
W_dec: float32 [16384, 2304]
b_enc: float32 [16384]
b_dec: float32 [2304]
threshold: float32 [16384]
```

This would combine `download_file` + `load_npz` and format the results. Low implementation effort given the existing CLI infrastructure.

## Open questions

- Should `NpyDtype` include integer types in the initial release, or defer to a follow-up version?
- Should the `npz` module be feature-gated from the start, or always included and gated later if the dependency footprint becomes a concern?
- Should `as_f32_slice` use `unsafe` pointer casting (forbidden by project conventions) or `bytemuck`-style safe transmutation? If the latter, `bytemuck` becomes an additional dependency.
