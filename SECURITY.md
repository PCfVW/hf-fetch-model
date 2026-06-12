# Security Policy

## Supported versions

hf-fetch-model is pre-1.0 and ships fixes on the latest released line only.
Security fixes land on `main` and in the next patch release; there is no
back-porting to older `0.x` minors. Pin with `~0.10` (or the current minor)
and upgrade promptly when a patch ships.

| Version | Supported          |
| ------- | ------------------ |
| latest `0.10.x` | :white_check_mark: |
| older `0.x`     | :x:                |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security problems.**

Report privately through GitHub's **[Security Advisories](https://github.com/PCfVW/hf-fetch-model/security/advisories/new)**
("Report a vulnerability" on the repository's *Security* tab). This opens a
private channel visible only to the maintainers.

Please include, as far as you can:

- the affected version (`hf-fm --version`) and platform,
- a description of the issue and its impact,
- a minimal reproduction — ideally the exact command, repo ID, or crafted
  input file that triggers it,
- whether the input came from an untrusted source.

You can expect an initial acknowledgement within a few days. Once a fix is
prepared we will coordinate a release and credit you in the advisory and
`CHANGELOG.md` unless you prefer to remain anonymous.

## Threat model — what is in scope

hf-fetch-model does two security-relevant things: it makes **authenticated
network requests** to the HuggingFace Hub and CDN, and it **parses
on-disk tensor files** (`.safetensors` / `.gguf` / `.npz` / `.pth`) that may
have originated from an untrusted repository.

In scope:

- **Malformed or malicious tensor files** crashing the process, exhausting
  memory, or escaping their parse boundaries during `inspect ... --cached`.
  The format parsers live in the [`anamnesis`](https://crates.io/crates/anamnesis)
  dependency, which enforces input-size caps (e.g. a 100 MiB cap on a PTH
  `data.pkl` before the pickle VM runs, a 1 MiB cap on an NPZ header, per-opcode
  caps on pickle string allocations). A parser DoS or memory-safety issue is a
  valid report against this project even when the root cause is upstream — we
  will coordinate with anamnesis.
- **Path traversal** when materializing downloaded files or reading cached
  ones (a repo whose file list contains `../` escapes).
- **Credential leakage** — `HF_TOKEN` or `--token` appearing in logs, error
  messages, panics, or written to disk.
- **TLS / transport** weaknesses in how requests are made.

Out of scope:

- The contents or trustworthiness of models hosted on the Hub themselves.
- Heuristic antivirus false positives on the locally-compiled binary (the tool
  is distributed as source via `cargo install`; see the
  [FAQ](docs/FAQ.md#is-the-hf-fm-binary-code-signed-why-did-my-antivirus-flag-it)).
- Denial of service that requires already-privileged local access.

## Hardening already in place

- `#![forbid(unsafe_code)]` across the crate (enforced in `Cargo.toml` lints).
- Input-size caps in the format parsers (via `anamnesis`).
- Per-file and total download timeouts, exponential backoff with jitter, and
  resumable partial downloads guarded by etag / size / schema invariants.
- A `cargo audit` CI job that fails the build on any RUSTSEC advisory, plus
  Dependabot for routine and security dependency updates.
- `Cargo.lock` committed for reproducible builds.
