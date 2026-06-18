# candle #3617 — reply 1 (Posted)

- **Target issue:** https://github.com/huggingface/candle/issues/3617
- **Status:** Posted (2026-06-13) — this is an issue **we opened** (PCfVW), so `p1` is the issue body itself, not a reply to someone else's thread.
- **Context:** Reports the `.pth` pickle-VM DoS in `candle-core/src/pickle.rs` — the sibling of the GGUF DoS class (#3533 / #3556 / #3585) that #3585 deliberately does not reach. Two unbounded vectors (CWE-1325 working-set amplification, CWE-674 recursion + derived `Drop`), three minimal PoCs, and anamnesis's shipped+fuzzed fix as the reference implementation.
- **Outcome:** 0 maintainer comments as of 2026-06-17. (Related signal: on #3533, ivarflakstad thanked @PCfVW by name and shipped #3585 — the same maintainer/threat-class, so the conversion template is proven.) Upstreaming plan: [`anamnesis/docs/PLAN-candle-3617-pickle-hardening.md`](../../../anamnesis/docs/PLAN-candle-3617-pickle-hardening.md).
- **Lesson / Leverage angle:** Security reports with PoCs + a reference implementation convert where diagnostic replies (#3401/#3448/#3530) stalled. The highest-conversion follow-up is to submit the PR ourselves (port anamnesis's pickle guards), mirroring the #3533→#3556 report→PR arc.
- **Accuracy flags:** (1) The issue body cites a **673-line** `pickle.rs`; current `main` is ~841 lines (structures verified present, line numbers re-anchored in the upstreaming plan). (2) The body's fuzz figures — `fuzz_pth` 385k, `fuzz_pth_limits` 508k, *"see the v0.6.6 CHANGELOG"* — are **inaccurate vs anamnesis's own records**: `fuzz/README.md` documents `fuzz_pth` **381k** / `fuzz_pth_limits` **193k** (latest campaign); the counts are not in the CHANGELOG at all; and 508k matches no PTH target (nearest is `fuzz_npz_parse` 503k). **Corrected in the live issue 2026-06-18** (originally posted with `fuzz_pth` 385k / `fuzz_pth_limits` 508k pointing at the CHANGELOG; now `fuzz_pth` 381k / `fuzz_pth_limits` 193k pointing at `fuzz/README.md`). The body below matches the current live issue.

---

Thanks for the quick turnaround on [#3533](https://github.com/huggingface/candle/issues/3533) and [#3556](https://github.com/huggingface/candle/pull/3556).

A sibling of the same DoS class lives in a file that [#3556](https://github.com/huggingface/candle/pull/3556) does not reach: the `.pth` pickle reader, `candle-core/src/pickle.rs` (673 lines, no caps anywhere). It is on the normal path. `read_pth_tensor_info` (L540-588), `PthTensors::new` (L622-648), and `read_all` (L666-673) run `Stack::read_loop` then `finalize`, materializing the entire object tree before any tensor info is extracted, so a crafted `.pth` drives it directly.

Two unbounded mechanisms, neither accounted for in the current code, as far as we can tell:

**1. Sequential allocation with no total bound ([CWE-1325](https://cwe.mitre.org/data/definitions/1325.html)).** `Stack` (L266-271) holds `stack: Vec<Object>` and `memo: HashMap<u32, Object>` and grows one `Object` per opcode with no cap (the `with_capacity(512)` is only a hint).

`size_of::<Object>()` is roughly 48 to 56 bytes, since the `Class { module_name, class_name }` variant (L225-247) dominates, so a flood of single-byte opcodes such as `N` or `K` scales by about 56x: a 100 MB `data.pkl` becomes a multi-GB stack.

The memo path is worse: `memo_get` (L312-318) does `obj.clone()`, a deep clone of the memoized `Object`, and `BinGet` / `LongBinGet` (push at L486 / L490) put that clone back on the stack, so a small pickle that memoizes a large `List` or `Dict` and replays it with a few `BINGET`s amplifies a few KB of opcodes into multi-GB of heap. There is no size or count limit on the memo.

**(end of 1.)**

**2. Uncontrolled recursion via nesting plus derived `Drop` ([CWE-674](https://cwe.mitre.org/data/definitions/674.html)).** `Object` (L225-247) nests through `Box<Object>` (`Reduce`, `Build`, `PersistentLoad`) and `Vec<Object>`, derives `Clone`, and has no custom `Drop`.

`BinPersId` (L400-403) calls `persistent_load` (L323-325), which wraps one box deeper per opcode (`Object::PersistentLoad(Box::new(id))`), so a `Q`-chain builds an arbitrarily deep value.

So when the `Stack` drops, the derived recursive `Drop` walks that depth and overflows the call stack, which aborts the process regardless of the consumer's panic strategy (a stack overflow cannot be caught). The `GGUF_MAX_VALUE_DEPTH` cap you added for `Value::Array` is exactly the control this needs, applied to the pickle `Object` instead.

**(end of 2.)**

These are distinct from the single-oversized-field shape in [#3533](https://github.com/huggingface/candle/issues/3533), and the `data.pkl` size bound does not help, because both vectors produce heap or depth far larger than the opcode stream. That gap between input size and resource use is the amplification. Both still sit under the [CWE-770](https://cwe.mitre.org/data/definitions/770.html) / [CWE-400](https://cwe.mitre.org/data/definitions/400.html) umbrella from [#3533](https://github.com/huggingface/candle/issues/3533).

**One scope note, so this is not overstated:** it is availability only, not Remote Code Execution. `reduce` (L296-312) builds an `Object::Reduce { ... }` without invoking the callable, so there is no pickle code-execution path here. The concern is purely the DoS.

**A minimal fix would mirror the GGUF one:** a working-set byte budget charged on each `Object` push and memo clone, a depth cap on `Object` construction (in the spirit of `GGUF_MAX_VALUE_DEPTH`), and `checked_mul` on any derived size. We shipped and `cargo-fuzz`-ed exactly this in [anamnesis](https://github.com/PCfVW/anamnesis); details and minimal repros below.

Hoping this helps!

<details>
<summary>How anamnesis fixes this (v0.6.6), for reference</summary>

anamnesis is a pure-Rust tensor-file parser whose own pickle VM had the same structure; our fix:

1. **One `O(1)`-per-opcode accounting choke point.** Every value push, and the deep size of every memo clone, is charged to a permanent working-set floor (`MAX_PICKLE_WORKING_SET`, 512 MiB) and to the caller's byte budget. That bounds the stack-flood and memo-replay vectors together.
2. **A permanent construction-depth cap** (`MAX_PICKLE_VM_DEPTH`, 256), so an over-deep value never forms, and recursive `Drop` (and every recursive walk) stays shallow. Same idea as `GGUF_MAX_VALUE_DEPTH`, at pickle construction time.
3. The accounting is deliberately `O(1)` per opcode (depth tracked incrementally, deep-size walk only on memo-clone opcodes). A naive per-push deep walk would itself be an `O(n^2)` CPU-DoS in the guard.

`cargo-fuzz`-ed: the pickle/PTH targets ran clean — `fuzz_pth` 381k runs and `fuzz_pth_limits` 193k runs, RSS-limited so an unbounded-heap regression surfaces as an OOM. See the campaign log in [`fuzz/README.md`](https://github.com/PCfVW/anamnesis/blob/main/fuzz/README.md).

</details>

<details>
<summary>Minimal repros (3 pickles)</summary>

`Stack` and `read_loop` are public, so these feed the VM directly (a unit test is enough); to trigger via the real `.pth` path, wrap the bytes as the `data.pkl` entry of a stored zip and point `read_pth_tensor_info` at it:

```rust
// 1) Memo-replay amplification (CWE-1325): ~140 KB input -> ~1e9 cloned Objects.
let mut b = vec![0x80, 0x02, b']', b'(']; // PROTO 2; EMPTY_LIST; MARK
b.extend(std::iter::repeat([b'K', 1]).take(50_000).flatten()); // 50k x BININT1(1)
b.push(b'e');                  // APPENDS -> 50k-element list
b.extend([b'q', 0]);           // BINPUT 0 (memoize the list)
b.extend(std::iter::repeat([b'h', 0]).take(20_000).flatten());  // BINGET 0 x20k
b.push(b'.');                  // STOP
let mut s = candle_core::pickle::Stack::empty();
let _ = s.read_loop(&mut &b[..]); // clones the 50k list 20k times -> multi-GB

// 2) Uncontrolled recursion (CWE-674): ~1 MB input -> ~1e6-deep value.
let mut b = vec![0x80, 0x02, b'K', 0];              // PROTO 2; BININT1 0
b.extend(std::iter::repeat(b'Q').take(1_000_000));  // 1e6 x BINPERSID
b.push(b'.');
let mut s = candle_core::pickle::Stack::empty();
let _ = s.read_loop(&mut &b[..]); // derived recursive Drop overflows the stack

// 3) Flat-stack flood (CWE-1325): same class, but needs a large file (~100 MB -> ~5.6 GB).
//    The memo-replay above shows the class with a ~90 KB input, so this is just for completeness.
let mut b = vec![0x80, 0x02];
b.extend(std::iter::repeat(b'N').take(100_000_000));
b.push(b'.');
```
</details>
