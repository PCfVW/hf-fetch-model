# Coding Conventions — Grit

hf-fetch-model follows [Grit — Strict Rust for AI-Assisted Development](https://github.com/PCfVW/Amphigraphic-Strict/tree/master/Grit).

## Rules

| Rule | Name | Enforcement | Key Constraint |
|------|------|-------------|----------------|
| 1 | Explicit lifetimes | `#![deny(elided_lifetimes_in_paths)]` | All public function signatures must have explicit lifetimes |
| 2 | Explicit conversions | `#![warn(clippy::as_conversions)]` | No implicit Deref coercion; use `.as_str()`, `.as_bytes()`, `.to_owned()` |
| 3 | No panic in libraries | `#![deny(clippy::unwrap_used, ...)]` | Use `Result` + `?`; never `.unwrap()`, `.expect()`, or `panic!()` |
| 4 | No type erasure | Code review | No `Box<dyn Any>`; prefer generics |
| 5 | Unsafe isolation | `#![forbid(unsafe_code)]` | No unsafe code |
| 6 | Owned types in async | Code review | Async functions take owned types (`String`, not `&str`) |
| 7 | Exhaustive matching | `#![deny(clippy::wildcard_enum_match_arm)]` | No `_` catch-all without documentation |
| 8 | Standard error pattern | `thiserror` | `FetchError` uses `thiserror`; `#[from]` for wrapped errors |
| 9 | Prefer iterators | `#![warn(clippy::explicit_iter_loop, ...)]` | Iterator chains over imperative loops |
| 10 | Single async runtime | Code review | Tokio only |
| 11 | `#[non_exhaustive]` | Code review | Public enums that may gain variants |
| 12 | `#[must_use]` | `#![warn(clippy::must_use_candidate)]` | Public functions returning a value with no side effects |

## Annotation Patterns

Every annotation is mandatory when the corresponding situation applies.

| Annotation | When Required | Example |
|------------|---------------|---------|
| `// TRAIT_OBJECT: <reason>` | Every `Box<dyn Trait>` or `&dyn Trait` usage | `// TRAIT_OBJECT: heterogeneous progress handlers` |
| `// EXHAUSTIVE: <reason>` | On `#[allow(clippy::exhaustive_enums)]` | `// EXHAUSTIVE: internal dispatch enum; crate owns all variants` |
| `// EXPLICIT: <reason>` | Intentional no-op match arm, or imperative loop over iterator chain | `// EXPLICIT: no action needed for this variant` |
| `// BORROW: <what>` | Explicit `.as_str()`, `.as_bytes()`, `.to_owned()` conversions | `// BORROW: explicit .as_str() instead of Deref coercion` |
| `// SAFETY: <invariants>` | Every `unsafe` block (not expected: `#![forbid(unsafe_code)]`) | N/A |

## SPDX Headers

Every `.rs` source file must begin with:

```rust
// SPDX-License-Identifier: MIT OR Apache-2.0
```
