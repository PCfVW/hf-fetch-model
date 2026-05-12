# hypomnesis Adoption Report — `inspect --check-gpu` (hf-fm v0.10.1)

**Date:** 2026-05-12
**hf-fm version:** 0.10.1
**hypomnesis version:** 0.2.0
**Context:** Adopting `hypomnesis` as the GPU-measurement substrate for the new `hf-fm inspect <repo> [FILE] --check-gpu [N]` flag. This is the first external consumer of `hypomnesis` (called out as the proof-of-concept use case in the [hypomnesis brief](https://github.com/PCfVW/hypomnesis/blob/main/docs/hypomnesis-brief.md)), and the report is written for the upstream maintainer (us) so the next hypomnesis release can act on real-world wear-and-tear.

---

## Summary

hypomnesis 0.2.0 worked on the first try. `device_info(index)` returned correct device-wide totals, free, and used numbers on a Windows 11 box with an RTX 5060 Ti (NVML primary, DXGI for the device name), and the error variants degraded cleanly when we passed out-of-range indices. No surprises, no workarounds, no `// SAFETY:` comments anywhere in the hf-fm side of the integration — the unsafe is all in hypomnesis and stays there.

Five concrete observations follow. Two are testability gaps that immediately blocked unit tests in `gpu_check.rs`; two are ergonomic gaps where every caller will write the same workaround; one is a positive note worth recording so future scope creep doesn't undermine it.

---

## What worked well

### Default-features set is exactly right

The hf-fm `Cargo.toml` line is the minimum form:

```toml
hypomnesis = "0.2"
```

No feature flags, no `default-features = false`, no target-conditional fiddling. The default set (`nvml`, `dxgi`, `nvidia-smi-fallback`) covers Windows NVIDIA + Linux NVIDIA out of the box, and the `windows` crate is already target-conditional in `hypomnesis/Cargo.toml` so Linux users pay nothing for it. **Keep that default invariant** — any future feature reshuffle that breaks the "one line, no flags" install bar should be considered a regression for downstream adopters.

### Error variants are well-named and `Display`-friendly

`HypomnesisError`'s `Display` impl produces strings that we could surface verbatim, with one polishing pass for sentence form. The `DeviceIndexOutOfRange { index, count }` variant is particularly nice — it carries enough structured data that hf-fm can render `"index 9 out of range (have 1 device)"` (singular/plural agreement) without parsing the string.

### `#[non_exhaustive]` on the error and snapshot types is the right call

We hit a future-proofing gap exactly *once* during the integration (next section), and the `#[non_exhaustive]` annotation kept hf-fm's `match` on `HypomnesisError` honest — we added a `_ => format!("hypomnesis error: {err}")` catch-all and the compiler gave us no choice. **Keep it.**

---

## What didn't quite work — five concrete asks

### 1. `GpuDeviceInfo` is `#[non_exhaustive]` and has no constructor — blocks downstream unit tests

This is the loudest finding. We wanted three unit tests in `src/gpu_check.rs` for the fit / miss arithmetic:

```rust
#[test]
fn gpu_check_json_fit_path() {
    let dev = hypomnesis::GpuDeviceInfo {        // ← rejected: cannot construct an
        index: 0,                                //   `#[non_exhaustive]` struct from
        name: Some("RTX 5060 Ti".to_owned()),    //   an external crate
        total_bytes: 17_179_869_184,
        free_bytes: 15_246_684_160,
        used_bytes: 1_933_185_024,
    };
    /* ... */
}
```

`#[non_exhaustive]` on `GpuDeviceInfo` is correct for API evolution (we want to add `temperature_celsius` later without a major bump) but it forbids external construction with a struct literal, which kills test fixtures for any downstream renderer. The current state is: we have one unit test (`gpu_check_json_error_path`, no `GpuDeviceInfo` needed) and a comment block in lieu of the other three, plus a manual-smoke-test note in the verification section of the v0.10.1 plan.

**Ask:** add either a `pub const fn` constructor for synthetic `GpuDeviceInfo` values (e.g., `GpuDeviceInfo::synthetic(index, name, total, free, used)`) or a public `Default` impl with builder-style setters. The Rust standard pattern for testable `#[non_exhaustive]` structs is a `new_for_test` / `synthetic` constructor under `#[doc(hidden)]` — visible to downstream crates, not part of the documented API, lets the test fixture stay in the downstream crate where it belongs.

Quantitative impact: every downstream that renders or transforms `GpuDeviceInfo` will hit this on first attempt at unit-testing the render path. With one PR upstream, we unblock all of them in patch-release form.

### 2. No `format_total` / `format_used` parity for the `report` feature's `format_free`

`hypomnesis 0.2` ships `GpuDeviceInfo::format_free` and `GpuDeviceInfo::print_free` under the `report` feature. We didn't pull `report` in (we format the verdict ourselves with hf-fm's own `format_size`), but the gap is noticeable: every consumer that wants `"NVIDIA RTX 5060 Ti — 16.0 GiB VRAM, free 14.2 GiB, used 1.8 GiB"` will hand-roll the same three lines of formatting glue. We did. candle-mi will. Anyone else will too.

**Ask:** one of —

- **(a)** add `format_total(&self) -> String`, `format_used(&self) -> String` parity helpers, **or**
- **(b)** add `format_summary(&self) -> String` returning the one-liner above (`"<name> — <total> VRAM, free <free>, used <used>"`),
- **(c)** add `format_free_used_total(&self) -> (String, String, String)` so callers can choose their template.

Whichever fits the report-feature spirit best. Option **(b)** is most opinionated and most useful for the consumers we know about.

### 3. `name: Option<String>` — add a `name_or_unknown(&self) -> &str` convenience

Every caller will write the same fallback. We did:

```rust
let name = dev.name.as_deref().unwrap_or("unknown GPU");
```

Two lines repeated across consumers is two lines too many.

**Ask:** add `impl GpuDeviceInfo { pub fn name_or_unknown(&self) -> &str { self.name.as_deref().unwrap_or("unknown GPU") } }`. Stable, no API surface to argue about, save every downstream the same boilerplate. Hf-fm v0.10.4 will adopt it if it lands (already captured in the v0.10.4 roadmap line).

### 4. Surface used: <5% rather than the brief's "≈10%"

The hypomnesis brief named hf-fm as a consumer of "`device_info` + `device_count`, ≈10 % of the surface." Actual usage in v0.10.1: **only `device_info`**. We do not call `device_count` because `--check-gpu N` targets a single device and the error path for an out-of-range index is handled by `device_info` itself returning `DeviceIndexOutOfRange { index, count }` — we get the count back through the error, for free, only when we actually need it.

Not a problem with hypomnesis — just a note that the brief's "first consumer" estimate was slightly high. **Multi-GPU support (`--check-gpu all`, planned in [v0.10.4](../roadmaps/cache-management-roadmap.md))** is when `device_count` will come into play; for now, the brief should probably say "uses `device_info` directly; `device_count` deferred to multi-GPU follow-up."

### 5. `DeviceIndexOutOfRange` vs `NoGpuSource` — clear taxonomy, one small redundancy

When we ran `--check-gpu 9` on a single-GPU box, we got the precise `"index 9 out of range (have 1 device)"` message. When we mentally simulated running it on a no-GPU box (we don't have one handy), we'd get `"no NVIDIA device detected (NVML / DXGI not usable)"`. The two messages are clearly distinguishable, which is what we want — users on a no-GPU box should know it's a backend gap, users on a wrong index should know it's a user-input gap.

The one small redundancy: `DeviceIndexOutOfRange` returns the count in its variant, and we then format `(have {count} {plural})` into the message. If hypomnesis ever standardizes a `Display` form that already includes the count, we'd duplicate it (`"device index 9 out of range (have 1 devices) (have 1 device)"`). Doesn't happen today — the current `#[error("device index {index} out of range (have {count} devices)")]` is good — but it's worth pinning down: **do we want the variant's `Display` to be the canonical user-facing string, or should consumers be expected to format from the structured fields?** We chose to format from the fields so we could fix the `1 devices` → `1 device` plural; consistent guidance in the hypomnesis docs would save the next consumer from making the same call.

---

## Adoption mechanics — what we changed and why

For the upstream record:

- **Cargo.toml**: added `hypomnesis = "0.2"` to `[dependencies]` (always-on, not feature-gated). The lock-file diff was clean — `hypomnesis 0.2.0`, `libloading 0.9.0`, the Windows-target `windows 0.62.2` family. No version conflicts with existing deps.
- **`src/gpu_check.rs`**: new ~260-line module gated by `#[path = "../gpu_check.rs"] mod gpu_check;` in `src/bin/main.rs` so it lives in the binary's module tree and stays out of the library's public API. Calls `hypomnesis::device_info(index)` exactly once per `--check-gpu` invocation.
- **No `unsafe`**: hf-fm is `#![forbid(unsafe_code)]`. The unsafe NVML / DXGI is hypomnesis's problem, well-encapsulated.

No need to wrap or feature-gate hypomnesis for cross-platform reasons — the `windows` crate is already target-conditional upstream.

---

## What we'd be sad to lose

A few hypomnesis design decisions paid off concretely in v0.10.1; calling them out so they don't get refactored away:

- **`#[non_exhaustive]` everywhere** — kept our `match` honest; the `_ => format!("hypomnesis error: {err}")` catch-all guarantees we cope with future variants. (The flip side is the testability gap in finding #1 — handle that with a constructor, not by removing the annotation.)
- **`device_info` is sync, not async** — we call it from a sync binary context; an async-only API would force a runtime where none is needed. The brief's note about "long-lived NVML context planned for v0.2" should preserve a sync entry point even when the cached path lands.
- **Three independent backends, dispatched by priority** — we didn't have to make a choice between NVML and DXGI; hypomnesis tried NVML first (got the numbers) and used DXGI only for the adapter name on Windows. Exactly the pre-rolled cake we wanted.

---

## Action items for hypomnesis 0.2.x / 0.3

Concrete patch-release-safe wins, in order of impact:

1. **`#[doc(hidden)] pub fn synthetic(...)` constructor on `GpuDeviceInfo`** — unblocks downstream tests. Two lines. (Finding #1.)
2. **`name_or_unknown(&self) -> &str` convenience** — saves every consumer two lines. (Finding #3.)
3. **`format_total` / `format_used` parity helpers under the `report` feature** — eliminates the format-the-three-numbers boilerplate. (Finding #2.)
4. **Doc note on the canonical user-facing error string** — pin down whether `Display` or structured-fields-then-format is the contract. (Finding #5.)

Items 1–3 could land as a single hypomnesis 0.2.1 patch. Item 4 is doc-only.

---

*Filed by the hf-fetch-model maintainer (= the hypomnesis maintainer = you) so the next hypomnesis release acts on the wear-and-tear from a real downstream adoption rather than synthetic guesses.*
