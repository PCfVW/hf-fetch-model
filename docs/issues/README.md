# Issue replies archive

This folder archives comments drafted or posted on upstream issues (primarily [huggingface/candle](https://github.com/huggingface/candle/issues) and related repos) where `hf-fetch-model`'s inspection features helped diagnose a problem. The dual purpose is:

1. **Reference**: keep a searchable trail of which issues we engaged with, what we said, and what happened.
2. **Case-study source material**: the v0.10.0 roadmap includes a first-batch of `docs/case-studies/` written from these replies (see [cache-management-roadmap.md](../roadmaps/cache-management-roadmap.md)). The archive is the raw material; case studies synthesize them into narrative form.

## File naming

```
<upstream-repo>-<issue-number>-p<N>.md
```

- `<upstream-repo>` — short identifier (`candle`, `hf-hub`, `transformers`, …)
- `<issue-number>` — the upstream issue/PR number (no `#`)
- `p<N>` — post/reply index within that thread (`p1` = our first comment, `p2` = second, …)

Examples: `candle-3448-p1.md`, `candle-3448-p3.md`, `candle-3401-p1.md`.

## File structure

Each file starts with a lightweight metadata header followed by the reply body verbatim:

```markdown
# <upstream-repo> #<issue-number> — reply <N> (<status>)

- **Target issue:** <URL>
- **Status:** <Posted | Draft | Superseded>  (plus date or superseding file)
- **Context:** one or two sentences: who asked, what was stuck, what we showed
- **Outcome:** (for posted) what the OP/maintainer said in response
- **Lesson / Leverage angle:** (optional) anything worth noting for future replies

---

<reply body exactly as posted, or as drafted>
```

The separator line (`---`) marks the boundary between metadata and the reply body, so the body is pastable into GitHub as-is.

## Status taxonomy

| Status | Meaning |
|--------|---------|
| **Posted** | Sent to the upstream issue; matches what's live there verbatim |
| **Draft** | Written but not yet posted (pending review, release timing, etc.) |
| **Superseded** | Was posted, but later investigation revealed a better diagnosis. The metadata must link to the superseding file. Do **not** delete superseded files — they document our learning curve and protect against re-making the same mistake. |

## Flagging practice

When archiving, **flag anything that's not 100% faithful or not 100% correct**:

- **Posted content with inaccuracies**: note them in the metadata. If a diagnosis turned out to be wrong or oversimplified, say so. Link to the corrected follow-up. Example: [candle-3448-p2.md](candle-3448-p2.md) — posted but oversimplified.
- **Cosmetic edits vs literal tool output**: if the archived reply trims or reformats `hf-fm` output (e.g., brace-notation shortcuts like `{k,o,q,v}_proj`), flag it so the archive isn't mistaken for raw output. Example: [candle-3401-p1.md](candle-3401-p1.md).
- **Verified vs guessed claims**: if a claim was verified against a specific reference (transformers' source, a paper, a config file), note the source. If something was a reasoned guess, say so explicitly. Example: [candle-3448-p3.md](candle-3448-p3.md) — "verified against `modeling_gemma4.py`".

## Workflow

1. **Before posting**: draft the reply as `<repo>-<issue>-p<N>.md` with status `Draft`. Review offline. Post.
2. **After posting**: update the status to `Posted`, add the posting date, and fill in `Outcome:` (blank for now; fill it in when the OP/maintainer replies).
3. **If superseded**: mark as `Superseded`, add a `See: <newer-file>` line in metadata, but keep the original file intact.

## Current archive

| File | Target | Status |
|------|--------|--------|
| [candle-3448-p1.md](candle-3448-p1.md) | candle #3448 — Gemma 4 download/load | Posted |
| [candle-3448-p2.md](candle-3448-p2.md) | candle #3448 — shape mismatch | Posted, superseded by p3 |
| [candle-3448-p3.md](candle-3448-p3.md) | candle #3448 — `use_double_wide_mlp` bug | Posted |
| [candle-3401-p1.md](candle-3401-p1.md) | candle #3401 — LightOnOCR-2-1B | Posted |
