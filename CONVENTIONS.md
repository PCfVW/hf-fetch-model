# hf-fetch-model Coding Conventions (Grit + Grit-MI Extensions)

This document describes the [Amphigraphic coding](https://github.com/PCfVW/Amphigraphic-Strict) conventions used in hf-fetch-model. It is a superset of
the [Grit — Strict Rust for AI-Assisted Development](https://github.com/PCfVW/Amphigraphic-Strict/tree/master/Grit).

## Trigger Checklist

**Before writing any line of code, check which triggers apply.**

| You are about to... | Check these rules |
|---|---|
| Write a `///` or `//!` comment | [Backtick hygiene](#backtick-hygiene), [field-level docs](#field-level-docs), [intra-doc link safety](#intra-doc-link-safety) |
| Write a `pub fn` or `pub const fn` | [`const fn`](#const-fn), [`#[must_use]`](#must_use-policy), [pass by value](#pass-by-value-vs-reference) |
| Write a `pub fn` returning `Result<T>` | [`# Errors` section](#errors-doc-section) |
| Write a `pub enum` | [`#[non_exhaustive]`](#non_exhaustive-policy) or [`// EXHAUSTIVE:`](#exhaustive-annotation) |
| Write an `as` cast | [`// CAST:`](#cast-annotation) |
| Write `slice[i]` or `slice[a..b]` | [`// INDEX:`](#index-annotation) |
| Write `.as_str()`, `.to_owned()` | [`// BORROW:`](#borrow-annotation) |
| Write an `unsafe` block | [`// SAFETY:`](#safety-annotation) |
| Write `Box<dyn T>` or `&dyn T` | [`// TRAIT_OBJECT:`](#trait_object-annotation) |
| Write a `match` or `if let` | [Control-flow rules](#if-let-vs-match), [`// EXPLICIT:`](#explicit-annotation) if no-op arm |
| Write error strings | [Error message wording](#error-message-wording) |
| Batch operations by key | [HashMap grouping idiom](#hashmap-grouping-idiom) |

---

## When Writing Doc Comments (`///`, `//!`)

### Backtick Hygiene

All identifiers, types, trait names, field names, crate names, and
file-format names in doc comments must be wrapped in backticks so that
rustdoc renders them as inline code and Clippy's `doc_markdown` lint passes.

Applies to: struct/enum/field names, method names (`fn foo`), types
(`Vec<T>`, `Option<f32>`), crate names (`hf_fetch_model`, `reqwest`),
file extensions (`.npy`, `.npz`, `.safetensors`), and acronyms that double
as types (`DType`, `NaN`, `CPU`, `GPU`).

> ✅ `/// Loads weights from a [`.safetensors`] file into a [`Tensor`].`
> ❌ `/// Loads weights from a .safetensors file into a Tensor.`

### Intra-Doc Link Safety

Rustdoc intra-doc links must resolve under all feature-flag combinations
(enforced by `#![deny(warnings)]` → `rustdoc::broken_intra_doc_links`).

Two patterns to watch:

1. **Feature-gated items** — items behind `#[cfg(feature = "...")]` are absent
   when that feature is off. Use plain backtick text, not link syntax:

   > ✅ `` /// Implemented by `NpyArray` (requires `npz` feature). ``
   > ❌ `` /// Implemented by [`NpyArray`](crate::npz::NpyArray). ``

2. **Cross-module links** — items re-exported at the crate root (e.g.,
   `FetchError`) are not automatically in scope inside submodules. Use explicit
   `crate::` paths:

   > ✅ `` /// Returns [`FetchError::Http`](crate::FetchError::Http) on failure. ``
   > ❌ `` /// Returns [`FetchError::Http`] on failure. ``

### Field-Level Docs

Every field of every `pub` struct must carry a `///` doc comment describing:
1. what the field represents,
2. its unit or valid range where applicable.

Fields of `pub(crate)` structs follow the same rule. Private fields inside a
`pub(crate)` or `pub` struct must have at minimum a `//` comment if their
purpose is not self-evident from the name alone.

> Example:
> ```rust
> pub struct FetchConfig {
>     /// Maximum number of concurrent file downloads.
>     pub concurrency: usize,
>     /// Number of parallel connections per file for chunked transfers.
>     pub connections_per_file: usize,
>     /// Minimum file size (in bytes) before chunked transfer kicks in.
>     pub chunk_threshold: u64,
> }
> ```

### `# Errors` Doc Section

All public fallible methods (`-> Result<T>`) must include an `# Errors` section
in their doc comment. Each bullet uses the format:

    /// # Errors
    /// Returns [`FetchError::Http`] if the API request fails.
    /// Returns [`FetchError::Io`] on file system failure.

Rules:
- Start each bullet with `Returns` followed by the variant in rustdoc link
  syntax, e.g., `` [`FetchError::Http`] ``.
- Follow with `if` (condition), `on` (event), or `when` (circumstance).
- Use the concrete variant name, not the generic `FetchError`.
- One bullet per distinct error path.

---

## When Writing Function Signatures

### `const fn`

Declare a function `const fn` when **all** of the following hold:
1. The body contains no heap allocation, I/O, or `dyn` dispatch.
2. All called functions are themselves `const fn`.
3. There are no trait-method calls that are not yet `const`.

This applies to constructors, accessors, and pure arithmetic helpers.
When in doubt, annotate and let the compiler reject it — do not omit `const`
preemptively.

> ✅ `pub const fn max_retries(&self) -> usize { self.max_retries }`
> ❌ `pub fn max_retries(&self) -> usize { self.max_retries }`

### `#[must_use]` Policy

All public functions and methods that return a value and have no side effects
must be annotated `#[must_use]`.  This includes constructors (`new`,
`with_capacity`), accessors (`len`, `is_empty`, `get_*`), and pure queries.
Without the annotation, a caller can silently discard the return value — which
for these functions is always a bug, since the call has no other effect.

The `clippy::must_use_candidate` lint enforces this at `warn` level
(promoted to error by `#![deny(warnings)]`).

### Pass by Value vs Reference

Follow these rules for function parameters:

| Type | Rule |
|---|---|
| `Copy` type ≤ 2 words (`usize`, `f32`, `bool`, small `enum`) | Pass by value |
| `Copy` type > 2 words | Pass by reference |
| Non-`Copy`, not mutated | Pass by `&T` or `&[T]` |
| Non-`Copy`, mutated | Pass by `&mut T` |
| Owned, consumed by callee | Pass by value (move semantics) |
| `&mut T` not actually mutated in body | Change to `&T` |

Never accept `&mut T` when the function body never writes through the reference;
Clippy's `needless_pass_by_ref_mut` will flag it and callers lose the ability
to pass shared references. Similarly, `trivially_copy_pass_by_ref` flags
`&T` where `T: Copy` and is small enough to pass by value.

> ✅ `fn scale(x: f32, factor: f32) -> f32`
> ❌ `fn scale(x: &f32, factor: &f32) -> f32`

---

## When Writing Public Enums

### `#[non_exhaustive]` Policy

- Public enums that may gain new variants: `#[non_exhaustive]`.
- Internal dispatch enums matched exhaustively by this crate:
  `#[allow(clippy::exhaustive_enums)] // EXHAUSTIVE: <reason>`.

---

## When Writing Expressions

These annotations are required **on or immediately before** the line where
the pattern occurs. Apply them as you write the line, not in a review pass.

### CAST Annotation

`// CAST: <from> → <to>, <reason>` — required on every `as` cast between numeric types. Prefer `From`/`Into` for
lossless conversions and `TryFrom`/`TryInto` with `?` for fallible ones.
Use `as` only when truncation or wrapping is the deliberate intent, or when
interfacing with a C-style API that mandates it.
> Example: `// CAST: usize → u32, tensor dim fits in u32 (checked at construction)`
> Example: `// CAST: f64 → f32, precision loss acceptable; value is a display scalar`

### INDEX Annotation

`// INDEX: <reason>` — required on every direct slice index (`slice[i]`, `slice[a..b]`) that cannot
be replaced by an iterator. Direct indexing panics on out-of-bounds; prefer
`.get(i)` with `?` or explicit error handling. Use direct indexing only when
the bound is provably valid and an iterator idiom would be significantly less
readable.
> Example: `// INDEX: i is bounded by dims.len() checked two lines above`

### BORROW Annotation

`// BORROW: <what is converted>` — required on explicit `.as_str()`, `.as_bytes()`, `.to_owned()` conversions (Grit Rule 2).
> Example: `// BORROW: explicit .as_str() instead of Deref coercion`

### TRAIT_OBJECT Annotation

`// TRAIT_OBJECT: <reason>` — required on every `Box<dyn Trait>` or `&dyn Trait` usage.
> Example: `// TRAIT_OBJECT: heterogeneous model backends require dynamic dispatch`

---

## When Writing `unsafe`

### SAFETY Annotation

`// SAFETY: <invariants>` — required on every `unsafe` block or function (inline comment, not a doc comment).

hf-fetch-model is `#![forbid(unsafe_code)]`; no `unsafe` is expected.
This annotation is included for completeness — if `unsafe` ever becomes
necessary, follow the candle-mi pattern: single dedicated module,
feature-gated, with a `// SAFETY:` comment on every block.

---

## When Writing Control Flow

### `if let` vs `match`

Use the most specific construct for the pattern at hand:

| Situation | Preferred form |
|---|---|
| Testing a single variant, no binding needed | `matches!(expr, Pat)` |
| Testing a single variant, binding needed | `if let Pat(x) = expr { … }` |
| Two or more variants with different bodies | `match expr { … }` |
| Exhaustive dispatch over an enum | `match expr { … }` (never `if let` chains) |

Never use a `match` with a single non-`_` arm and a no-op `_ => {}` where
`if let` or `matches!` would be clearer (Clippy: `single_match`,
`match_like_matches_macro`). Conversely, never chain three or more
`if let … else if let …` arms where a `match` would be exhaustive.

> ✅ `if let Some(w) = weight { apply(w); }`
> ✅ `matches!(dtype, DType::F16 | DType::BF16)`
> ❌ `match weight { Some(w) => apply(w), None => {} }`

### EXPLICIT Annotation

`// EXPLICIT: <reason>` — required when a match arm is intentionally a no-op, or when an imperative
loop is used instead of an iterator chain for a stateful computation.
> Example: `// EXPLICIT: retry loop carries mutable attempt counter; .map() would hide it`

### EXHAUSTIVE Annotation

`// EXHAUSTIVE: <reason>` — required on `#[allow(clippy::exhaustive_enums)]`.
> Example: `// EXHAUSTIVE: internal dispatch enum; crate owns and matches all variants`

---

## When Writing Error Strings

### Error Message Wording

Error strings passed to `FetchError` variants follow two patterns:

- **External failures** (I/O, serde, network): `"failed to <verb>: {e}"`
  > Example: `FetchError::Http(format!("failed to fetch model listing: {e}"))`
- **Validation failures** (range, lookup): `"<noun> <problem> (<context>)"`
  > Example: `FetchError::Http(format!("HF API returned status {status}"))`
  > Example: `FetchError::Checksum(format!("SHA mismatch for {filename}"))`

Rules:
- Use lowercase, no trailing period.
- Include the offending value and the valid range or constraint when applicable.
- Wrap external errors with `: {e}`, not `.to_string()`.

---

## When Batching Operations by Key

### HashMap Grouping Idiom

When operations must be batched by a key (e.g., grouping files by repository
to avoid redundant API calls), use the `Entry` API:

```rust
let mut by_repo: HashMap<String, Vec<Item>> = HashMap::new();
for item in items {
    by_repo.entry(item.key()).or_default().push(item);
}
```

Rules:
- Name the map `by_<grouping_key>` (e.g., `by_repo`, `by_extension`).
- Use `.entry(key).or_default().push()` — never `if let Some` + `else insert`.
- Iterate the map to perform the batched operation (one API call per key).
