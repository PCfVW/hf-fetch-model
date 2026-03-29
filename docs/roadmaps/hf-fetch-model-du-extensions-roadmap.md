# `hf-fm du` Extensions Roadmap

**Date:** March 29, 2026
**Status:** Proposed
**Context:** `hf-fm du` (v0.9.0) shows disk usage per cached repo or per file. It's useful but passive — users see sizes, then must retype long repo IDs to drill into details or delete. This roadmap proposes numbered indexing for fast drill-down, tree display for structural visibility, and shell completion for repo IDs.

---

## Current behavior

```
$ hf-fm du
    5.10 GiB  google/gemma-2-2b-it                              (8 files)
    2.80 GiB  EleutherAI/pythia-1.4b                            (8 files)
    1.20 GiB  google/gemma-scope-2b-pt-res                      (3 files)
    0.21 GiB  RWKV/RWKV7-Goose-0.1B                            (3 files)
  ──────────────────────────────────────────────────────
    9.31 GiB  total (4 repos, 22 files)

$ hf-fm du google/gemma-2-2b-it
    2.50 GiB  model-00001-of-00002.safetensors
    2.60 GiB  model-00002-of-00002.safetensors
    1.20 KiB  config.json
    ...
```

Pain: after running `hf-fm du`, the user must copy-paste or retype the repo ID for the next command. For repos like `mistralai/Ministral-3-3B-Instruct-2512`, this is tedious and error-prone.

---

## Feature 1: Numbered indexing

Add a `#` column to the sorted list. Accept a numeric index anywhere a `REPO_ID` is expected in `du`.

### Proposed output

```
$ hf-fm du
   #   SIZE        REPO                                              FILES
   1   5.10 GiB    google/gemma-2-2b-it                              8
   2   2.80 GiB    EleutherAI/pythia-1.4b                            8
   3   1.20 GiB    google/gemma-scope-2b-pt-res                      3
   4   0.21 GiB    RWKV/RWKV7-Goose-0.1B                            3
  ──────────────────────────────────────────────────────────────────
   9.31 GiB    total (4 repos, 22 files)
```

### Drill-down by index

```
$ hf-fm du 2
  EleutherAI/pythia-1.4b:

   #   SIZE        FILE
   1   2.50 GiB    model-00001-of-00002.safetensors
   2   0.26 GiB    model-00002-of-00002.safetensors
   3   4.10 MiB    tokenizer.json
   ...
   8   1.20 KiB    config.json
  ──────────────────────────────────────────────────────────────────
   2.80 GiB    total (8 files)
```

The repo name is printed as a header so the user always knows what they're looking at. Both `hf-fm du 2` and `hf-fm du EleutherAI/pythia-1.4b` work — the index is a shorthand, not a replacement.

### Disambiguation

Numeric indexes are unambiguous: repo IDs always contain `/` (e.g., `google/gemma-2-2b-it`), bare numbers never do. A simple check is sufficient:

```rust
if arg.contains('/') {
    // Treat as repo ID
} else if let Ok(n) = arg.parse::<usize>() {
    // Treat as index into the sorted du list
} else {
    // Error: not a valid repo ID or index
}
```

### Ephemeral index caveat

The index is tied to the current cache state — it changes when repos are added or removed. This is fine for interactive use (run `du`, then immediately drill in) but should not be relied on in scripts. The output makes this clear by using `#` notation rather than a persistent identifier.

---

## Feature 2: Tree display (`--tree`)

Show the physical HF cache directory structure with box-drawing characters. Detailed design already exists in the [v0.10.0 roadmap](v0.10.0-roadmap.md) (Feature 3).

### Combined with indexing

```
$ hf-fm du 2 --tree
  models--EleutherAI--pythia-1.4b/
  ├── refs/
  │   └── main                                       (→ abc1234)
  └── snapshots/
      └── abc1234def5678.../
          ├── model-00001-of-00002.safetensors        2.50 GiB
          ├── model-00002-of-00002.safetensors        0.26 GiB
          ├── ...
          └── config.json                             1.20 KiB
```

### Global tree

```
$ hf-fm du --tree
  ~/.cache/huggingface/hub/
  ├── models--google--gemma-2-2b-it/                 5.10 GiB
  │   ├── refs/
  │   │   └── main                                   (→ abc1234)
  │   └── snapshots/
  │       └── abc1234def5678.../
  │           ├── model-00001-of-00002.safetensors    2.50 GiB
  │           └── ...
  ├── models--EleutherAI--pythia-1.4b/               2.80 GiB
  │   └── ...
  └── ...
```

Tree display is most useful for repos with subdirectories (e.g., sharded models with nested folders) or when investigating the refs/snapshots/blobs structure for debugging.

---

## Feature 3: Cross-command index sharing

If `hf-fm du` shows numbered repos, other cache commands could accept the same index:

```
$ hf-fm du
   #   SIZE        REPO                                              FILES
   1   5.10 GiB    google/gemma-2-2b-it                              8
   2   2.80 GiB    EleutherAI/pythia-1.4b                            8
   ...

$ hf-fm cache delete 2
  EleutherAI/pythia-1.4b  (2.80 GiB, 8 files)
  Delete? [y/N]
```

### Implementation approach

Both `du` and `cache delete` compute the sorted repo list via `cache::cache_summary()`. Since the sort is deterministic (by size descending, then by name for ties), the same index maps to the same repo as long as the cache hasn't changed between the two commands. This is acceptable for interactive use.

No persistent state file is needed — both commands independently compute the list and resolve the index. The only risk is a race condition (cache changes between `du` and `delete`), which the confirmation prompt mitigates: the user sees the repo name and size before confirming.

### Which commands support indexing

| Command | Index support | Notes |
|---------|--------------|-------|
| `hf-fm du <N>` | Yes | Drill into Nth repo |
| `hf-fm du <N> --tree` | Yes | Tree view of Nth repo |
| `hf-fm cache delete <N>` | Yes | Delete Nth repo (with confirmation) |
| `hf-fm cache verify <N>` | Future | Verify Nth repo |
| `hf-fm cache path <N>` | Future | Print path of Nth repo |
| `hf-fm inspect <N>` | Maybe | Could be useful but mixes two command families |

---

## Feature 4: Shell completion for repo IDs

Instead of numeric indexing, offer tab-completion for repo IDs from the cache. Type `hf-fm du goo<TAB>` and get `google/gemma-2-2b-it`.

### Approach

Use [`clap_complete`](https://docs.rs/clap_complete/) to generate shell completion scripts. For dynamic repo ID completion, `clap_complete` supports a native completion engine (via the `COMPLETE=$SHELL` environment variable protocol) that can call back into the binary at completion time. The `#[arg(add = ...)]` annotations would live on the `REPO_ID` arguments across all subcommands, including the `cache` subcommand group proposed in the [cache management roadmap](cache-management-roadmap.md).

```rust
// In the clap argument definition:
#[arg(add = clap_complete::ArgValueCandidates::new(complete_cached_repos))]
repo_id: Option<String>,

fn complete_cached_repos() -> Vec<clap_complete::CompletionCandidate> {
    let Ok(summaries) = cache::cache_summary() else {
        return Vec::new();
    };
    summaries
        .into_iter()
        .map(|s| clap_complete::CompletionCandidate::new(s.repo_id))
        .collect()
}
```

### Shell support

| Shell | Mechanism | Status |
|-------|-----------|--------|
| Bash | `source <(hf-fm --completions bash)` or `eval "$(COMPLETE=bash hf-fm)"` | Supported by `clap_complete` |
| Zsh | `source <(hf-fm --completions zsh)` | Supported by `clap_complete` |
| Fish | `hf-fm --completions fish \| source` | Supported by `clap_complete` |
| PowerShell | `hf-fm --completions powershell \| Invoke-Expression` | Supported by `clap_complete` |

### Trade-offs vs. numeric indexing

| | Numeric indexing | Shell completion |
|--|-----------------|-----------------|
| Setup | Zero — works immediately | Requires one-time shell configuration |
| Speed | Instant (no I/O at completion time) | Scans cache directory on each `<TAB>` |
| Discoverability | Visible in output (`#` column) | Invisible until user tries `<TAB>` |
| Cross-platform | Works everywhere | Depends on shell support |
| Scriptability | Fragile (ephemeral index) | Stable (repo IDs don't change) |

**Recommendation:** Implement both. Numeric indexing is the fast interactive path (zero setup, visible in output). Shell completion is the proper long-term solution (stable identifiers, works across all commands). They complement each other — indexing for "see big thing, act on it now", completion for "I know the repo name, help me type it".

---

## Implementation order

| Priority | Feature | Complexity | Release target |
|----------|---------|-----------|----------------|
| 1 | Numbered `#` column in `du` output | Low — format change only | v0.9.2 |
| 2 | `du <N>` drill-down by index | Low — parse index, resolve from sorted list | v0.9.2 |
| 3 | `cache delete <N>` index support | Low — same resolution logic | v0.9.2 |
| 4 | `du <N> --tree` | Medium — tree rendering (v0.10.0 roadmap) | v0.10.0 |
| 5 | `du --tree` global | Medium — same rendering, full cache walk | v0.10.0 |
| 6 | Shell completion (`clap_complete`) | Medium — new dependency, per-shell scripts | v0.10.0 |

Features 1–3 ship together in v0.9.2 alongside `cache delete` and `cache clean-partial` from the [cache management roadmap](cache-management-roadmap.md). The numbered column is only useful if you can act on it, so indexing and `cache delete` should land in the same release. Features 4–6 are more complex and belong in v0.10.0.
