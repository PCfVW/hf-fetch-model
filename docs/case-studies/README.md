# Case studies

Narrative write-ups of real investigations where `hf-fetch-model`'s inspection
features did diagnostic work — synthesized from the raw reply archive in
[`docs/issues/`](../issues/). Where the archive is "what we said, verbatim,"
a case study is "what the problem was, how the tool was pointed at it, what it
revealed, and — honestly — how far it got."

**Honesty about outcomes is the point.** These are not success-theatre. hf-fm
is a *diagnostic* tool: it gives you ground truth about a model's tensor
layout, dtypes, and memory footprint without downloading the weights. That
leverage is real, but whether a diagnosis *lands* upstream depends on things
outside the tool — whether the person on the other end shares the repo,
implements the fix, or reports back. Several of these threads went quiet after
a correct diagnosis was posted. We write that down too, because the
transferable lesson is in the *workflow*, not the issue's close state.

Each case study is built to answer three questions for a reader who hits a
similar wall:

1. **The symptom** — what failed, in the reporter's own words.
2. **The diagnostic path** — which hf-fm commands, what they showed, and the
   reasoning from output to root cause (including wrong turns).
3. **The transferable workflow** — the one-line pattern to reuse.

## Index

| Case study | hf-fm workflow demonstrated | Source thread |
|------------|-----------------------------|---------------|
| [Per-layer shape variation in Gemma 4](gemma4-per-layer-shape-variation.md) | `inspect --tree` reveals when a "constant" architecture isn't — a shape that varies by layer index | candle [#3448](https://github.com/huggingface/candle/issues/3448) |
| [Reconstructing an OOM from a crash log](qwen3-next-memory-forensics.md) | `inspect --dtypes` for ground-truth on-disk size; fingerprinting a model the reporter never named | candle [#3530](https://github.com/huggingface/candle/issues/3530) |

A note on timing: case studies are written *after* the upstream conversation
has had time to settle, so the narrative reflects the real reception rather
than wishful framing at posting time.
