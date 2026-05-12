# `cargo install` silent-skip on upgrade — install-surface dogfooding

**Date:** 2026-05-12
**hf-fm version:** v0.9.8 → v0.10.1 upgrade attempt
**Context:** Immediately after publishing v0.10.1 to crates.io and confirming the publish workflow green, the maintainer ran `cargo install hf-fetch-model --features cli` to upgrade the locally-installed binary. The install command exited `0` but `hf-fm --version` still reported `0.9.8`. The fix was `cargo install hf-fetch-model --features cli --force`, which built and replaced the binary correctly. This note captures the failure mode, the diagnostic confusion it caused, and the install-surface improvements landing in response.

---

## Summary

`cargo install <crate>` (without `--force`) short-circuits when **any** version of the binary is already installed locally: cargo prints a low-priority `Package <crate> v<X.Y.Z> is already installed, use --force to override` notice to stderr and exits `0` without rebuilding. The user perceives this as "install succeeded" because the exit code is success and the message is easy to miss in a busy terminal. The binary on `PATH` is unchanged.

The dynamic is worse immediately after a publish: crates.io's CDN takes a few minutes to propagate the new version globally, so even on a fresh `cargo install` retry, cargo's local registry cache may still see the older version as the latest, reinforcing the skip.

Cost in this session: ~10 minutes of "I just shipped v0.10.1 but my own dogfooding terminal can't run the new flag", plus one round of mis-diagnosis (a Bash-tool PATH resolution in the assistant's harness returned a stale `hf-fm --version` result that briefly muddied which side of the wire was wrong). Both costs were paid by the maintainer; the README + FAQ patches landing alongside this note are calibrated so the next person hitting this doesn't.

---

## What's already in cargo's UX (not hf-fm's to fix)

- `cargo install`'s exit-0-with-stderr-message-on-skip is documented behavior, not a bug. The notice format is clear if you see it; in a fast-moving terminal it disappears above the prompt within a second.
- `cargo install --force` (or `cargo install --version =X.Y.Z`) is the documented workaround. Cargo's own docs mention this but do not make `--force` the default.
- crates.io's index-propagation delay is upstream of cargo entirely — cargo's local index cache refreshes on its own schedule.

None of those are hf-fm bugs. They are environment factors that hf-fm's install instructions need to acknowledge.

---

## What hf-fm can improve

### 1. README + FAQ install pages now flag `--force` for upgrades (shipped with this note)

The README's [Install](../../README.md#install) section gains an [Upgrading from a previous version](../../README.md#upgrading-from-a-previous-version) subsection covering the `--force` flag and a `hf-fm --version` verification step. The FAQ's [Installation and authentication](../FAQ.md#installation-and-authentication) section gains a third question — [*How do I upgrade hf-fm? Why does `cargo install` silently keep the old version?*](../FAQ.md#how-do-i-upgrade-hf-fm-why-does-cargo-install-silently-keep-the-old-version) — that explains the short-circuit in user-facing language and links back to this note.

This is the smallest fix — pure documentation, zero code — and lands as a docs-only commit alongside this dogfooding entry.

### 2. Future: `hf-fm --check-update` flag (proposed)

A one-shot, opt-in network call that queries `https://crates.io/api/v1/crates/hf-fetch-model` for the latest version, compares to `env!("CARGO_PKG_VERSION")`, and prints either `up to date` or `current 0.10.1, latest 0.10.4 — upgrade with: cargo install hf-fetch-model --features cli --force`. ~30 LOC, no telemetry, only runs when the user explicitly invokes it.

Design constraints:

- **Opt-in only.** hf-fm must not add a daily HTTP call to its critical path or carry telemetry-shaped state. The user invokes `--check-update` when they want to know.
- **Single endpoint.** crates.io's public API is sufficient; no auth, no rate-limit concerns at hf-fm's scale.
- **No auto-upgrade.** The flag reports and points at the exact `cargo install ... --force` command. The user runs that themselves.

Status: candidate for a future v0.10.x patch alongside `--pick` (v0.10.5 currently planned for discovery UX). Filed as a follow-up.

### 3. Future: prebuilt binaries + `cargo-binstall` support (proposed)

Publish GitHub Release artifacts (win-x64, mac-x64, mac-arm64, linux-x64) for each tagged release. [`cargo-binstall`](https://github.com/cargo-bins/cargo-binstall) then installs prebuilt binaries without compiling, sidestepping `cargo install`'s local-cache logic entirely. Real fix for the silent-skip case in the upgrade path — `cargo binstall hf-fetch-model` checks the upstream release tag and downloads the matching artifact, no skip-on-existing-version short-circuit.

Cost: cross-compile CI matrix, release packaging job, code-signing on macOS and Windows for the gatekeeper-friendly path. Minor-release scope. Defer until download metrics justify the maintenance burden.

---

## Reproduction

```sh
# 1. Have an older hf-fm installed:
cargo install hf-fetch-model@0.9.8 --features cli
hf-fm --version
# → hf-fetch-model 0.9.8

# 2. Try to upgrade to the latest (where ANY version is on disk):
cargo install hf-fetch-model --features cli
# → Stderr: "Package `hf-fetch-model v0.9.8` is already installed, use --force to override"
# → Exit: 0
hf-fm --version
# → hf-fetch-model 0.9.8 (no change)

# 3. Force the upgrade:
cargo install hf-fetch-model --features cli --force
hf-fm --version
# → hf-fetch-model 0.10.1
```

---

## Lesson

Install-surface UX is part of release UX. Publishing the crate is not the same as users having it. The README's install section is the first signal users have about *how* to keep their install fresh; treating it as a one-liner under-serves anyone who has already installed the crate once and now wants the new release. The README + FAQ patches landing with this note are the minimum credible fix; `--check-update` and `cargo-binstall` support are the principled longer-term improvements, tracked as future patch-release candidates.
