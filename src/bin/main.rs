// SPDX-License-Identifier: MIT OR Apache-2.0

//! CLI binary for hf-fetch-model.
//!
//! Installed as both `hf-fetch-model` and `hf-fm`.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use clap::{
    ArgAction, ArgGroup, Args, CommandFactory, FromArgMatches, Parser, Subcommand, ValueEnum,
};
use tracing_subscriber::EnvFilter;

use hf_fetch_model::cache;
use hf_fetch_model::discover;
use hf_fetch_model::inspect;
use hf_fetch_model::progress::IndicatifProgress;
use hf_fetch_model::repo;
use hf_fetch_model::{
    compile_glob_patterns, file_matches, has_glob_chars, DownloadPlan, FetchConfig,
    FetchConfigBuilder, FetchError, Filter,
};

#[path = "../format.rs"]
mod format;

#[path = "../gpu_check.rs"]
mod gpu_check;

use format::format_size;

/// Applies optional `--timeout-per-file-secs` / `--timeout-total-secs` CLI overrides to a `FetchConfigBuilder`.
///
/// Both arguments are seconds; `None` leaves the corresponding builder field
/// untouched (so the [`FetchConfig`] default — 300 s per file, no total
/// limit — applies).
#[must_use]
fn apply_timeout_overrides(
    builder: FetchConfigBuilder,
    per_file_secs: Option<u64>,
    total_secs: Option<u64>,
) -> FetchConfigBuilder {
    let mut builder = builder;
    if let Some(secs) = per_file_secs {
        builder = builder.timeout_per_file(Duration::from_secs(secs));
    }
    if let Some(secs) = total_secs {
        builder = builder.timeout_total(Duration::from_secs(secs));
    }
    builder
}

/// Downloads all files from a `HuggingFace` model repository.
///
/// Use `--preset safetensors` to download only safetensors weights,
/// config, and tokenizer files.
#[derive(Parser)]
#[command(
    name = "hf-fetch-model",
    bin_name = "hf-fm",
    version,
    about,
    before_help = concat!("hf-fetch-model v", env!("CARGO_PKG_VERSION"))
)]
#[command(args_conflicts_with_subcommands = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[command(flatten)]
    download: DownloadArgs,
}

/// Arguments for the default download command.
#[derive(Args)]
struct DownloadArgs {
    /// Enable verbose output (download diagnostics).
    #[arg(short, long)]
    verbose: bool,

    /// The repository identifier (e.g., "google/gemma-2-2b-it").
    #[arg(value_name = "REPO_ID")]
    repo_id: Option<String>,

    /// Git revision (branch, tag, or commit SHA).
    #[arg(long)]
    revision: Option<String>,

    /// Authentication token (or set `HF_TOKEN` env var).
    #[arg(long)]
    token: Option<String>,

    /// Include glob pattern (repeatable).
    #[arg(long, action = clap::ArgAction::Append)]
    filter: Vec<String>,

    /// Exclude glob pattern (repeatable).
    #[arg(long, action = clap::ArgAction::Append)]
    exclude: Vec<String>,

    /// Filter preset.
    #[arg(long, value_enum)]
    preset: Option<Preset>,

    /// Output directory (default: HF cache).
    #[arg(long)]
    output_dir: Option<PathBuf>,

    /// Number of concurrent file downloads (auto-tuned if omitted).
    #[arg(long)]
    concurrency: Option<usize>,

    /// Minimum file size (MiB) for parallel chunked download (auto-tuned if omitted).
    #[arg(long)]
    chunk_threshold_mib: Option<u64>,

    /// Number of parallel HTTP connections per large file (auto-tuned if omitted).
    #[arg(long)]
    connections_per_file: Option<usize>,

    /// Per-file download timeout in seconds (default: 300).
    ///
    /// Override when downloading large files on slow connections — at
    /// 10 MiB/s effective throughput the default 300 s caps progress at
    /// roughly 3 GiB. Try 1800 for files in the 5–15 GiB range.
    #[arg(long)]
    timeout_per_file_secs: Option<u64>,

    /// Total batch download timeout in seconds (default: no limit).
    ///
    /// Bounds the entire multi-file download. Independent of `--timeout-per-file-secs`.
    #[arg(long)]
    timeout_total_secs: Option<u64>,

    /// Preview what would be downloaded without actually downloading.
    #[arg(long)]
    dry_run: bool,

    /// Copy downloaded files to flat layout: `{output-dir}/{filename}`.
    ///
    /// Files are downloaded to the HF cache as normal, then copied to
    /// the target directory. Defaults to the current directory when
    /// `--output-dir` is not set.
    #[arg(long)]
    flat: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// List model families in local HF cache.
    #[command(after_help = "Examples:
  hf-fm list-families                           # default grouped view
  hf-fm list-families --show quant              # add a quant column
  hf-fm list-families --tag bitsandbytes        # filter to bitsandbytes-tagged repos
  hf-fm list-families --show quant --tag gguf   # combine

See also: hf-fm discover, hf-fm du")]
    ListFamilies {
        /// Additional columns to show. Currently only `quant` (read from
        /// each repo's cached `config.json` `quantization_config.quant_method`;
        /// falls back to `gguf` for repos with a `.gguf` file in the snapshot).
        #[arg(long, value_delimiter = ',', value_enum)]
        show: Vec<ShowFamiliesColumn>,
        /// Filter cached repos by a `HuggingFace` tag (case-insensitive).
        /// Tags are fetched at query time via the HF `model_info` API, bounded
        /// to 8 concurrent requests; per-repo fetch failures silently drop
        /// the row from the filter result.
        #[arg(long)]
        tag: Option<String>,
    },
    /// Discover new model families from the `HuggingFace` Hub.
    #[command(after_help = "Examples:
  hf-fm discover --limit 100                  # top families by download count
  hf-fm discover --tag bitsandbytes           # only models carrying this tag
  hf-fm discover --tag gguf --limit 200       # tag composes with --limit

See also: hf-fm list-families, hf-fm search")]
    Discover {
        /// Maximum number of models to scan.
        #[arg(long, default_value = "500")]
        limit: usize,
        /// Filter by model tag (e.g., `"gguf"`, `"bitsandbytes"`, `"conversational"`).
        #[arg(long)]
        tag: Option<String>,
    },
    /// Search the `HuggingFace` Hub for models matching a query.
    ///
    /// Supports comma-separated multi-term filtering (e.g., `"mistral,3B,12"`).
    /// Slashes in queries are treated as spaces for broader matching.
    #[command(after_help = "Examples:
  hf-fm search \"fp4\" --tag bitsandbytes        # text-match AND tag-match
  hf-fm search \"llama\" --tag gguf --limit 5    # tag composes with other filters
  hf-fm search \"qwen,3B\" --exact               # exact repo-id match
  hf-fm search \"fp4\" --show tags,size          # enrich rows with tag list + total size

See also: hf-fm list-families, hf-fm discover")]
    Search {
        /// Search query (e.g., `"RWKV-7"`, `"llama 3"`, `"mistral,3B,12"`).
        query: String,
        /// Maximum number of results.
        #[arg(long, default_value = "20")]
        limit: usize,
        /// Match a full repository ID exactly (e.g., `"org/model"`) and show its metadata card.
        #[arg(long)]
        exact: bool,
        /// Filter by library framework (e.g., `"transformers"`, `"peft"`, `"vllm"`).
        #[arg(long)]
        library: Option<String>,
        /// Filter by pipeline task (e.g., `"text-generation"`, `"text-classification"`).
        #[arg(long)]
        pipeline: Option<String>,
        /// Filter by model tag (e.g., `"gguf"`, `"conversational"`, `"imatrix"`).
        #[arg(long)]
        tag: Option<String>,
        /// Additional columns to show, comma-separated. `tags` is free
        /// (already in the API payload); `size` adds one extra HTTP request
        /// per result, fanned out through a bounded semaphore.
        #[arg(long, value_delimiter = ',', value_enum)]
        show: Vec<ShowColumn>,
    },
    /// Show model card metadata and README text for a repository.
    Info {
        /// The repository identifier (e.g., `"mistralai/Ministral-3-3B-Instruct-2512"`).
        repo_id: String,
        /// Git revision (branch, tag, or commit SHA).
        #[arg(long)]
        revision: Option<String>,
        /// Authentication token (or set `HF_TOKEN` env var).
        #[arg(long)]
        token: Option<String>,
        /// Output metadata and README as JSON.
        #[arg(long)]
        json: bool,
        /// Maximum lines of README to display (0 = all).
        #[arg(long, default_value = "40")]
        lines: usize,
    },
    /// Download a single file (or glob pattern) from a `HuggingFace` repository.
    DownloadFile {
        /// Enable verbose output (download diagnostics).
        #[arg(short, long)]
        verbose: bool,

        /// The repository identifier (e.g., "mntss/clt-gemma-2-2b-426k").
        repo_id: String,

        /// Filename or glob pattern (e.g., `"model.safetensors"` or `"pytorch_model-*.bin"`).
        filename: String,

        /// Git revision (branch, tag, or commit SHA).
        #[arg(long)]
        revision: Option<String>,

        /// Authentication token (or set `HF_TOKEN` env var).
        #[arg(long)]
        token: Option<String>,

        /// Output directory (default: HF cache).
        #[arg(long)]
        output_dir: Option<PathBuf>,

        /// Minimum file size (MiB) for parallel chunked download (auto-tuned if omitted).
        #[arg(long)]
        chunk_threshold_mib: Option<u64>,

        /// Number of parallel HTTP connections per large file (auto-tuned if omitted).
        #[arg(long)]
        connections_per_file: Option<usize>,

        /// Per-file download timeout in seconds (default: 300).
        ///
        /// Override when downloading large files on slow connections — at
        /// 10 MiB/s effective throughput the default 300 s caps progress at
        /// roughly 3 GiB. Try 1800 for files in the 5–15 GiB range.
        #[arg(long)]
        timeout_per_file_secs: Option<u64>,

        /// Total download timeout in seconds (default: no limit).
        ///
        /// Bounds the whole download (including retries). Independent of
        /// `--timeout-per-file-secs`.
        #[arg(long)]
        timeout_total_secs: Option<u64>,

        /// Preview what would be downloaded without actually downloading.
        #[arg(long)]
        dry_run: bool,

        /// Copy the downloaded file to flat layout: `{output-dir}/{filename}`.
        ///
        /// The file is downloaded to the HF cache as normal, then copied to
        /// the target directory. Defaults to the current directory when
        /// `--output-dir` is not set.
        #[arg(long)]
        flat: bool,
    },
    /// Show download status (all models, or a specific one).
    Status {
        /// The repository identifier (omit to list all cached models).
        repo_id: Option<String>,
        /// Git revision (branch, tag, or commit SHA).
        #[arg(long)]
        revision: Option<String>,
        /// Authentication token (or set `HF_TOKEN` env var).
        #[arg(long)]
        token: Option<String>,
        /// Re-evaluate which remote files are deliberate skips. Files not
        /// matching this preset's glob list are reported as `excluded`
        /// instead of `MISSING`. Overrides any value persisted by
        /// `download --preset` in the per-repo `.hf-fm-snapshot.json` sidecar.
        #[arg(long, value_enum)]
        preset: Option<Preset>,
        /// Output the status report as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Compare tensor layouts between two models.
    ///
    /// Inspects `.safetensors` headers in both repos and classifies tensors
    /// into four buckets: only-in-A, only-in-B, dtype/shape differences,
    /// and matching. Does not download weight data.
    Diff {
        /// First model repository (labeled A).
        repo_a: String,
        /// Second model repository (labeled B).
        repo_b: String,
        /// Git revision for model A.
        #[arg(long)]
        revision_a: Option<String>,
        /// Git revision for model B.
        #[arg(long)]
        revision_b: Option<String>,
        /// Authentication token (or set `HF_TOKEN` env var).
        #[arg(long)]
        token: Option<String>,
        /// Cache-only mode: fail if files are not cached locally.
        #[arg(long)]
        cached: bool,
        /// Show only tensors whose name contains this substring (case-insensitive).
        #[arg(long)]
        filter: Option<String>,
        /// Show only the summary line (counts per category).
        #[arg(long, conflicts_with = "dtypes")]
        summary: bool,
        /// Show per-dtype histograms side-by-side instead of the per-tensor body.
        ///
        /// Aggregates tensors by dtype on both sides; prints two columns
        /// (A: Tensors, A: Size; B: Tensors, B: Size) plus a Δ Size column.
        /// Composes with --filter (histograms aggregate over filtered tensors
        /// only). Conflicts with --summary (incoherent intents: one-line total
        /// vs per-dtype table).
        #[arg(long, conflicts_with = "summary")]
        dtypes: bool,
        /// Show only the first N tensors per section (only-A / only-B / differ), applied after `--filter`.
        #[arg(long)]
        limit: Option<usize>,
        /// Output the full diff as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Show disk usage for cached models.
    Du {
        /// Repository identifier or numeric index (omit to show all cached repos).
        ///
        /// Use a repo ID (e.g., `"google/gemma-2-2b-it"`) or a `#` index from the
        /// `du` summary to drill into a specific repo's files.
        repo_id: Option<String>,
        /// Show a last-modified age column (e.g., `"2 days ago"`, `"3 months ago"`).
        #[arg(long)]
        age: bool,
        /// Show a hierarchical tree view of every cached repo and its files.
        ///
        /// Reuses the box-drawing connectors (`├──`, `└──`, `│   `) and
        /// dynamic-column alignment of `inspect --tree`. Composes with
        /// `--age`, which adds a last-modified column on each repo branch.
        #[arg(long, conflicts_with = "repo_id")]
        tree: bool,
        /// Output disk usage as JSON (flat by default; `--tree` nests files per repo).
        #[arg(long)]
        json: bool,
    },
    /// Inspect tensor file headers (`.safetensors` remote/cached; `.gguf` / `.npz` / `.pth` cached).
    ///
    /// Reads tensor metadata without downloading full weight data. For
    /// `.safetensors`, checks the local cache first and falls back to HTTP
    /// Range requests. For `.gguf` / `.npz` / `.pth`, only cached inspect
    /// is supported in v0.10.x (remote inspect for those formats is planned
    /// for v0.11); pass `--cached` after downloading the file.
    #[command(after_help = "Examples:\n  \
        hf-fm inspect <repo>                                    # inspect every .safetensors in the repo\n  \
        hf-fm inspect <repo> --filter blocks.0.                 # matched tensor names (per shard/file)\n  \
        hf-fm inspect <repo> --list                             # list tensor files (no headers read)\n  \
        hf-fm inspect <repo> 3                                  # inspect file #3 from --list\n  \
        hf-fm inspect <repo> --pick                             # pick the file interactively\n  \
        hf-fm inspect <repo> fluxV13 --pick --dtypes            # substring narrows, then pick\n  \
        hf-fm inspect <repo> model.safetensors --tree           # hierarchical view of one file\n  \
        hf-fm inspect <repo> --check-gpu                        # GPU-fit verdict for the whole repo\n  \
        hf-fm inspect <repo> model.gguf --cached                # inspect a cached GGUF file (v0.10.2+)\n\n\
        Indices returned by --list are stable as long as the repo has not\n\
        changed remotely between invocations. Pass --revision <sha> on both\n\
        --list and the follow-up run to lock the view end-to-end.\n\n\
        --check-gpu reads device 0's VRAM via hypomnesis (NVML / DXGI) and\n\
        reports a one-line fit verdict against the model's weight bytes.\n\
        Pass --check-gpu N to target a specific device.\n\n\
        For a walkthrough on a real 4-shard model, see\n\
        docs/tutorials/inspect-before-downloading.md\n\n\
        See also: hf-fm diff <A> <B>    # compare two repos' tensor layouts")]
    Inspect {
        /// The repository identifier (e.g., `"google/gemma-2-2b-it"`).
        repo_id: String,
        /// Specific tensor file (`.safetensors` / `.gguf` / `.npz` / `.pth`),
        /// numeric index from `--list`, or omit for all.
        filename: Option<String>,
        /// Git revision (branch, tag, or commit SHA).
        #[arg(long)]
        revision: Option<String>,
        /// Authentication token (or set `HF_TOKEN` env var).
        #[arg(long)]
        token: Option<String>,
        /// Cache-only mode: fail if the file is not cached locally.
        #[arg(long)]
        cached: bool,
        /// List supported tensor files in the repo (filename + size) and exit.
        ///
        /// Covers `.safetensors` / `.gguf` / `.npz` / `.pth`. Prints a numbered
        /// table; the `#` column can be used as the `filename` argument on a
        /// follow-up run (e.g. `hf-fm inspect <repo> 3`). Indices are
        /// alphabetical, so shard ordering is natural. No headers are read.
        #[arg(long, conflicts_with_all = ["filename", "no_metadata", "json", "filter", "dtypes", "limit", "tree"])]
        list: bool,
        /// Pick the file to inspect interactively from a numbered list.
        ///
        /// With no `FILENAME`, offers every supported tensor file in the repo
        /// (`.safetensors` / `.gguf` / `.npz` / `.pth`). With a `FILENAME`,
        /// treats it as a case-insensitive substring filter: a unique match
        /// auto-resolves (with a `Resolving to <name>` note on stderr),
        /// several matches prompt for a numbered choice. Under `--pick` the
        /// `FILENAME` argument is never a numeric index. Requires an
        /// interactive terminal (stdin + stderr); non-interactive contexts
        /// should use `--list` + `hf-fm inspect <repo> <n>` instead. The
        /// prompt goes to stderr, so `--json` stdout can still be redirected.
        /// Composes with every rendering flag (`--tree`, `--dtypes`,
        /// `--filter`, `--limit`, `--check-gpu`, `--json`).
        #[arg(long, conflicts_with = "list")]
        pick: bool,
        /// Suppress the `Metadata:` line in human-readable output.
        #[arg(long)]
        no_metadata: bool,
        /// Output the full header as JSON instead of a human-readable table.
        #[arg(long)]
        json: bool,
        /// Show only tensors whose name contains this substring (case-insensitive).
        #[arg(long)]
        filter: Option<String>,
        /// Show a per-dtype summary instead of individual tensors.
        #[arg(long)]
        dtypes: bool,
        /// Show only the first N tensors (applied after `--filter`).
        #[arg(long)]
        limit: Option<usize>,
        /// Show a hierarchical tree view grouped by dotted namespace prefix.
        ///
        /// Numeric sibling groups with identical structure are collapsed to
        /// `[0..N]` with a `×K` marker. Composes with `--filter` and `--json`.
        #[arg(long, conflicts_with_all = ["dtypes", "limit"])]
        tree: bool,
        /// Show a GPU-fit verdict for the model weights against device `N` (default 0).
        ///
        /// Reads the device's total/free/used VRAM via `hypomnesis` (NVML on
        /// Linux/Windows, DXGI on Windows). On systems with no NVIDIA device
        /// or where neither backend is usable, prints a friendly note and
        /// skips the verdict — never fails the command. The verdict uses the
        /// **unfiltered** model totals; `--filter` and `--limit` only affect
        /// the printed tensor table. Composes with `--json` (adds a
        /// `gpu_check` object alongside the existing header schema).
        #[arg(
            long,
            num_args = 0..=1,
            default_missing_value = "0",
            value_name = "N",
            conflicts_with = "list"
        )]
        check_gpu: Option<u32>,
        /// KV-cache context length to fold into the `--check-gpu` verdict.
        ///
        /// Reads the model's `config.json` for the attention dimensions and
        /// estimates KV-cache bytes at this sequence length, then reports a
        /// real fit verdict against `weights + KV` instead of weights alone.
        /// Requires `--check-gpu`. Sliding-window models (Gemma, Mistral) are
        /// capped at their window; multi-head latent attention (`DeepSeek`) is
        /// skipped with a note. The KV element size tracks the model's
        /// activation dtype (`bf16` / `fp16`), independent of weight quant.
        #[arg(long, value_name = "N", requires = "check_gpu")]
        context: Option<u32>,
    },
    /// List files in a remote `HuggingFace` repository (no download).
    ListFiles {
        /// The repository identifier (e.g., `"google/gemma-2-2b-it"`).
        repo_id: String,
        /// Git revision (branch, tag, or commit SHA).
        #[arg(long)]
        revision: Option<String>,
        /// Authentication token (or set `HF_TOKEN` env var).
        #[arg(long)]
        token: Option<String>,
        /// Include glob pattern (repeatable).
        #[arg(long, action = clap::ArgAction::Append)]
        filter: Vec<String>,
        /// Exclude glob pattern (repeatable).
        #[arg(long, action = clap::ArgAction::Append)]
        exclude: Vec<String>,
        /// Filter preset (`safetensors`, `gguf`, `npz`, `pth`, `config-only`).
        #[arg(long, value_enum)]
        preset: Option<Preset>,
        /// Suppress the SHA256 column.
        #[arg(long)]
        no_checksum: bool,
        /// Show cache status for each file (complete, partial, or missing).
        #[arg(long)]
        show_cached: bool,
        /// Output the file list as JSON (full SHA256, regardless of `--no-checksum`).
        #[arg(long)]
        json: bool,
    },
    /// Manage the local `HuggingFace` cache.
    Cache {
        #[command(subcommand)]
        subcommand: CacheCommands,
    },
}

// EXHAUSTIVE: internal CLI dispatch enum; crate owns all variants
#[derive(Subcommand)]
enum CacheCommands {
    /// Remove `.chunked.part` files from interrupted downloads.
    CleanPartial {
        /// Repository identifier or numeric index (omit to clean all repos).
        repo_id: Option<String>,

        /// Skip confirmation prompt.
        #[arg(long)]
        yes: bool,

        /// Preview what would be removed without deleting.
        #[arg(long)]
        dry_run: bool,
    },
    /// Delete a cached model by repo ID or numeric index.
    Delete {
        /// Repository identifier or numeric index from `du` output.
        repo_id: String,

        /// Skip confirmation prompt.
        #[arg(long)]
        yes: bool,
    },
    /// Garbage-collect cached models by age and/or size budget.
    ///
    /// Requires at least one of `--older-than` or `--max-size`. When both
    /// are given, age eviction runs first; if the cache is still over the
    /// size budget, oldest non-protected repos are evicted next. Repos
    /// listed in `--except` are never evicted, and active partial
    /// downloads (`has_partial == true` with mtime in the last 60 minutes)
    /// are skipped to avoid racing with `hf-fm download`.
    #[command(group(
        ArgGroup::new("gc_criteria")
            .args(["older_than", "max_size"])
            .required(true)
            .multiple(true)
    ))]
    Gc {
        /// Evict repos with mtime older than this many days.
        #[arg(long, value_name = "DAYS")]
        older_than: Option<u64>,

        /// Hard cap on total cache size (e.g., `5GiB`, `500MiB`). Binary units only.
        #[arg(long, value_name = "SIZE", value_parser = parse_size_arg)]
        max_size: Option<u64>,

        /// Repository identifier to protect from eviction (repeatable).
        #[arg(long = "except", value_name = "REPO_ID", action = ArgAction::Append)]
        except: Vec<String>,

        /// Preview the eviction plan without deleting anything.
        #[arg(long)]
        dry_run: bool,

        /// Skip the confirmation prompt before deleting.
        #[arg(long)]
        yes: bool,

        /// List every kept repo in the preview (default: hidden for terseness).
        #[arg(long)]
        list_kept: bool,
    },
    /// Print the snapshot directory path for a cached model.
    ///
    /// Output is a bare path (no labels), suitable for shell substitution:
    /// `cd $(hf-fm cache path google/gemma-2-2b-it)`.
    ///
    /// Resolves the `main` ref by default; pass `--revision <REV>` to
    /// resolve a different branch, tag, or commit SHA.
    Path {
        /// Repository identifier or numeric index from `du` output.
        repo_id: String,
        /// Git revision (branch, tag, or commit SHA). Defaults to `main`.
        #[arg(long)]
        revision: Option<String>,
    },
    /// Re-verify `SHA256` digests of cached files against `HuggingFace` LFS metadata.
    ///
    /// Requires network access to fetch the expected digests. Files without
    /// LFS metadata (small git-stored files such as `config.json`) are
    /// skipped. The command exits non-zero if any file fails verification,
    /// so it composes cleanly into CI / cron checks.
    Verify {
        /// Repository identifier or numeric index from `du` output.
        repo_id: String,

        /// Git revision (branch, tag, or commit SHA).
        #[arg(long)]
        revision: Option<String>,

        /// Authentication token (or set `HF_TOKEN` env var).
        #[arg(long)]
        token: Option<String>,
    },
}

// EXHAUSTIVE: internal CLI dispatch enum; crate owns all variants
#[derive(Clone, ValueEnum)]
enum Preset {
    Safetensors,
    Gguf,
    Npz,
    Pth,
    ConfigOnly,
}

// EXHAUSTIVE: internal CLI dispatch enum; crate owns all variants
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ShowColumn {
    /// Inline list of all tags from the model's metadata (free; already in the HF API payload).
    Tags,
    /// Total repository size, summed across all files. Requires one extra HTTP request per result.
    Size,
}

// EXHAUSTIVE: internal CLI dispatch enum; crate owns all variants
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ShowFamiliesColumn {
    /// Quant column read from each repo's cached `config.json`
    /// (`quantization_config.quant_method`); falls back to `gguf` for repos
    /// with a `.gguf` file in the snapshot.
    Quant,
}

/// Canonical kebab-case name for a [`Preset`], matching the value `clap` parses
/// from `--preset <NAME>` and the string stored in [`cache::Snapshot::preset`].
///
/// Single source of truth shared by the download-time snapshot writer, the
/// status-time snapshot reader, and the glob lookup
/// ([`hf_fetch_model::config::preset_globs`]).
const fn preset_name(preset: &Preset) -> &'static str {
    match preset {
        Preset::Safetensors => "safetensors",
        Preset::Gguf => "gguf",
        Preset::Npz => "npz",
        Preset::Pth => "pth",
        Preset::ConfigOnly => "config-only",
    }
}

/// Sorts a [`clap::Command`]'s subcommands alphabetically by assigning
/// ascending `display_order` values in sorted-name order. Recurses into
/// each subcommand so nested command trees (e.g., `cache …`) are sorted
/// at every level.
#[must_use]
fn sort_subcommands_alphabetically(mut cmd: clap::Command) -> clap::Command {
    let mut names: Vec<String> = cmd
        .get_subcommands()
        .map(|sc| sc.get_name().to_owned()) // BORROW: clap borrows sc; owned String outlives the closure
        .collect();
    names.sort();
    for (i, name) in names.iter().enumerate() {
        cmd = cmd.mut_subcommand(name, |sc| {
            sort_subcommands_alphabetically(sc).display_order(i)
        });
    }
    cmd
}

fn main() -> ExitCode {
    let cmd = sort_subcommands_alphabetically(Cli::command());
    let matches = cmd.get_matches();
    let cli = match Cli::from_arg_matches(&matches) {
        Ok(cli) => cli,
        Err(e) => e.exit(),
    };

    // Extract --verbose from the active command context.
    // EXHAUSTIVE: non-download subcommands have no --verbose flag
    let verbose = match &cli.command {
        Some(Commands::DownloadFile { verbose, .. }) => *verbose,
        None => cli.download.verbose,
        Some(
            Commands::ListFamilies { .. }
            | Commands::Discover { .. }
            | Commands::Search { .. }
            | Commands::Info { .. }
            | Commands::Status { .. }
            | Commands::Diff { .. }
            | Commands::Du { .. }
            | Commands::Inspect { .. }
            | Commands::ListFiles { .. }
            | Commands::Cache { .. },
        ) => false,
    };

    // Initialize tracing subscriber when --verbose is set.
    // Respects RUST_LOG if present, otherwise defaults to debug for hf_fetch_model.
    if verbose {
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("hf_fetch_model=debug"));
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(false)
            .with_writer(std::io::stderr)
            .init();
    }

    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(FetchError::PartialDownload { path, failures }) => {
            eprintln!();
            eprintln!(
                "error: {} {} failed to download:",
                failures.len(),
                pluralize(failures.len(), "file", "files")
            );
            for f in &failures {
                eprintln!("  - {}: {}", f.filename, f.reason);
            }
            if let Some(p) = path {
                eprintln!();
                eprintln!("Partial download at: {}", p.display());
            }
            let any_retryable = failures.iter().any(|f| f.retryable);
            if any_retryable {
                eprintln!();
                eprintln!(
                    "hint: re-run the same command to retry failed files \
                     (already-downloaded files will be skipped)"
                );
            }
            ExitCode::FAILURE
        }
        Err(FetchError::RepoNotFound { ref repo_id }) => {
            // BORROW: explicit .clone() for owned String in Display formatting
            eprintln!(
                "error: {e}",
                e = FetchError::RepoNotFound {
                    repo_id: repo_id.clone()
                }
            );
            // Extract model name (part after '/') as a search hint.
            // BORROW: explicit .as_str() instead of Deref coercion
            let search_term = repo_id.split('/').nth(1).unwrap_or(repo_id.as_str());
            eprintln!("hint: try `hf-fm search {search_term}` to find matching models");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

// EXPLICIT: top-level CLI dispatch — one arm per subcommand; extracting helpers
// would hide the dispatch shape rather than clarify it.
#[allow(clippy::too_many_lines)]
fn run(cli: Cli) -> Result<(), FetchError> {
    match cli.command {
        // BORROW: explicit .as_deref() for owned → borrowed conversion
        Some(Commands::ListFamilies { show, tag }) => run_list_families(&show, tag.as_deref()),
        // BORROW: explicit .as_deref() for owned → borrowed conversion
        Some(Commands::Discover { limit, tag }) => run_discover(limit, tag.as_deref()),
        // BORROW: explicit .as_str()/.as_deref() for owned → borrowed conversions
        Some(Commands::Search {
            query,
            limit,
            exact,
            library,
            pipeline,
            tag,
            show,
        }) => run_search(
            query.as_str(),
            limit,
            exact,
            library.as_deref(),
            pipeline.as_deref(),
            tag.as_deref(),
            &show,
        ),
        // BORROW: explicit .as_str()/.as_deref() for owned → borrowed conversions
        Some(Commands::Info {
            repo_id,
            revision,
            token,
            json,
            lines,
        }) => run_info(
            repo_id.as_str(),
            revision.as_deref(),
            token.as_deref(),
            json,
            lines,
        ),
        // BORROW: explicit .as_str()/.as_deref() for owned → borrowed conversions
        Some(Commands::DownloadFile {
            verbose: _,
            repo_id,
            filename,
            revision,
            token,
            output_dir,
            chunk_threshold_mib,
            connections_per_file,
            timeout_per_file_secs,
            timeout_total_secs,
            dry_run,
            flat,
        }) => run_download_file(DownloadFileParams {
            repo_id: repo_id.as_str(),
            filename: filename.as_str(),
            revision: revision.as_deref(),
            token: token.as_deref(),
            output_dir,
            chunk_threshold_mib,
            connections_per_file,
            timeout_per_file_secs,
            timeout_total_secs,
            dry_run,
            flat,
        }),
        Some(Commands::Status {
            repo_id: Some(repo_id),
            revision,
            token,
            preset,
            json,
            // BORROW: explicit .as_str()/.as_deref() for owned → borrowed conversions
        }) => run_status(
            repo_id.as_str(),
            revision.as_deref(),
            token.as_deref(),
            preset.as_ref(),
            json,
        ),
        Some(Commands::Status {
            repo_id: None,
            json,
            ..
        }) => run_status_all(json),
        // BORROW: explicit .as_str()/.as_deref() for owned → borrowed conversions
        Some(Commands::Diff {
            repo_a,
            repo_b,
            revision_a,
            revision_b,
            token,
            cached,
            filter,
            summary,
            dtypes,
            limit,
            json,
        }) => run_diff(
            repo_a.as_str(),
            repo_b.as_str(),
            revision_a.as_deref(),
            revision_b.as_deref(),
            token.as_deref(),
            cached,
            filter.as_deref(),
            summary,
            dtypes,
            limit,
            json,
        ),
        // BORROW: explicit .as_str() for String → &str conversion
        Some(Commands::Du {
            repo_id: Some(repo_id),
            age: _,
            tree: _, // EXPLICIT: clap conflicts_with rejects --tree alongside repo_id
            json,
        }) => {
            // BORROW: explicit .as_str() instead of Deref coercion
            let resolved = resolve_du_arg(repo_id.as_str())?;
            run_du_repo(resolved.as_str(), json)
        }
        Some(Commands::Du {
            repo_id: None,
            age,
            tree: true,
            json,
        }) => run_du_tree(age, json),
        Some(Commands::Du {
            repo_id: None,
            age,
            tree: false,
            json,
        }) => run_du(age, json),
        // BORROW: explicit .as_str()/.as_deref() for owned → borrowed conversions
        Some(Commands::Inspect {
            repo_id,
            filename,
            revision,
            token,
            cached,
            list,
            pick,
            no_metadata,
            json,
            filter,
            dtypes,
            limit,
            tree,
            check_gpu,
            context,
        }) => run_inspect(
            repo_id.as_str(),
            filename.as_deref(),
            revision.as_deref(),
            token.as_deref(),
            cached,
            list,
            pick,
            no_metadata,
            json,
            filter.as_deref(),
            dtypes,
            limit,
            tree,
            check_gpu,
            context,
        ),
        // BORROW: explicit .as_str()/.as_deref() for owned → borrowed conversions
        Some(Commands::ListFiles {
            repo_id,
            revision,
            token,
            filter,
            exclude,
            preset,
            no_checksum,
            show_cached,
            json,
        }) => run_list_files(
            repo_id.as_str(),
            revision.as_deref(),
            token.as_deref(),
            &filter,
            &exclude,
            preset.as_ref(),
            no_checksum,
            show_cached,
            json,
        ),
        // BORROW: explicit .as_str() for String → &str conversion
        Some(Commands::Cache { subcommand }) => match subcommand {
            CacheCommands::CleanPartial {
                repo_id,
                yes,
                dry_run,
            } => {
                let resolved = repo_id.map(|r| resolve_du_arg(r.as_str())).transpose()?;
                run_cache_clean_partial(resolved.as_deref(), yes, dry_run)
            }
            // BORROW: explicit .as_str() for String → &str conversion
            CacheCommands::Delete { repo_id, yes } => {
                let resolved = resolve_du_arg(repo_id.as_str())?;
                // BORROW: explicit .as_str() instead of Deref coercion
                run_cache_delete(resolved.as_str(), yes)
            }
            CacheCommands::Gc {
                older_than,
                max_size,
                except,
                dry_run,
                yes,
                list_kept,
            } => run_cache_gc(older_than, max_size, except, dry_run, yes, list_kept),
            // BORROW: explicit .as_str()/.as_deref() for owned → borrowed conversions
            CacheCommands::Path { repo_id, revision } => {
                let resolved = resolve_du_arg(repo_id.as_str())?;
                run_cache_path(resolved.as_str(), revision.as_deref())
            }
            // BORROW: explicit .as_str()/.as_deref() for owned → borrowed conversions
            CacheCommands::Verify {
                repo_id,
                revision,
                token,
            } => {
                let resolved = resolve_du_arg(repo_id.as_str())?;
                run_cache_verify(resolved.as_str(), revision.as_deref(), token.as_deref())
            }
        },
        None => run_download(cli.download),
    }
}

/// Progress reporter for non-TTY contexts (pipes, CI).
///
/// Emits periodic one-line progress to stderr every 5 seconds or every 10%
/// of total size, whichever comes first.
struct NonTtyProgress {
    /// Timestamp of the last progress line emitted.
    last_report: Mutex<Instant>,
    /// Last reported 10%-bucket per file (to detect 10% boundary crossings).
    last_bucket: Mutex<HashMap<String, u64>>,
}

impl NonTtyProgress {
    fn new() -> Self {
        Self {
            last_report: Mutex::new(Instant::now()),
            last_bucket: Mutex::new(HashMap::new()),
        }
    }

    /// Handles a `ProgressEvent`, emitting a progress line to stderr when the
    /// reporting threshold is reached.
    fn handle(&self, event: &hf_fetch_model::progress::ProgressEvent) {
        // Skip completion events (the summary line handles those).
        if event.percent >= 100.0 {
            return;
        }

        // CAST: f64 → u64, precision loss acceptable; bucket index for 10% increments
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::as_conversions
        )]
        let bucket = (event.percent / 10.0) as u64;

        let elapsed_ok = self
            .last_report
            .lock()
            .is_ok_and(|guard| guard.elapsed().as_secs() >= 5);

        let bucket_crossed = self.last_bucket.lock().is_ok_and(|mut map| {
            // BORROW: explicit .clone() for owned String as HashMap key
            let prev = map.entry(event.filename.clone()).or_insert(0);
            if bucket > *prev {
                *prev = bucket;
                true
            } else {
                false
            }
        });

        if elapsed_ok || bucket_crossed {
            if let Ok(mut ts) = self.last_report.lock() {
                *ts = Instant::now();
            }
            // CAST: f64 → u64, precision loss acceptable; display-only percentage
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::as_conversions
            )]
            let pct = event.percent as u64;
            eprintln!(
                "[hf-fm] {}: {}/{} ({pct}%)",
                event.filename,
                format_size(event.bytes_downloaded),
                format_size(event.bytes_total)
            );
        }
    }
}

// EXPLICIT: linear composition of preset selection, builder overrides, output-dir
// handling, progress reporter wiring, runtime construction, and post-finalize
// messaging. Splitting would obscure the sequential setup flow.
#[allow(clippy::too_many_lines)]
fn run_download(args: DownloadArgs) -> Result<(), FetchError> {
    let dry_run = args.dry_run;

    let repo_id = args.repo_id.as_deref().ok_or_else(|| {
        FetchError::InvalidArgument(
            "REPO_ID is required for download. Usage: hf-fm <REPO_ID>".to_owned(),
        )
    })?;

    if !repo_id.contains('/') {
        return Err(FetchError::InvalidArgument(format!(
            "invalid REPO_ID \"{repo_id}\": expected \"org/model\" format (e.g., \"EleutherAI/pythia-1.4b\")"
        )));
    }

    if dry_run {
        return run_dry_run(repo_id, &args);
    }

    // Consume repo_id for the download path.
    // BORROW: explicit .to_owned() for &str → owned String
    let repo_id = repo_id.to_owned();
    let flat = args.flat;

    // When --flat, output_dir is the flat copy target, not the HF cache root.
    // BORROW: explicit .clone() for owned Option<PathBuf>
    let flat_target = if flat { args.output_dir.clone() } else { None };

    // Build FetchConfig from CLI args.
    let mut builder = match args.preset {
        Some(Preset::Safetensors) => Filter::safetensors(),
        Some(Preset::Gguf) => Filter::gguf(),
        Some(Preset::Npz) => Filter::npz(),
        Some(Preset::Pth) => Filter::pth(),
        Some(Preset::ConfigOnly) => Filter::config_only(),
        None => FetchConfig::builder(),
    };

    if let Some(ref preset) = args.preset {
        warn_redundant_filters(preset, &args.filter);
    }

    if let Some(rev) = args.revision.as_deref() {
        builder = builder.revision(rev);
    }
    if let Some(tok) = args.token.as_deref() {
        builder = builder.token(tok);
    } else {
        builder = builder.token_from_env();
    }
    for pattern in &args.filter {
        // BORROW: explicit .as_str() instead of Deref coercion
        builder = builder.filter(pattern.as_str());
    }
    for pattern in &args.exclude {
        // BORROW: explicit .as_str() instead of Deref coercion
        builder = builder.exclude(pattern.as_str());
    }
    if let Some(c) = args.concurrency {
        builder = builder.concurrency(c);
    }
    if let Some(ct) = args.chunk_threshold_mib {
        builder = builder.chunk_threshold(ct.saturating_mul(1024 * 1024));
    }
    if let Some(cpf) = args.connections_per_file {
        builder = builder.connections_per_file(cpf);
    }
    builder = apply_timeout_overrides(builder, args.timeout_per_file_secs, args.timeout_total_secs);
    if !flat {
        if let Some(dir) = args.output_dir {
            builder = builder.output_dir(dir);
        }
    }

    // Set up progress reporting: indicatif bars for TTY, periodic stderr for non-TTY.
    let is_tty = std::io::stderr().is_terminal();
    let indicatif = if is_tty {
        let p = Arc::new(IndicatifProgress::new());
        let handle = Arc::clone(&p);
        builder = builder.on_progress(move |e| handle.handle(e));
        Some(p)
    } else {
        let p = Arc::new(NonTtyProgress::new());
        let handle = Arc::clone(&p);
        builder = builder.on_progress(move |e| handle.handle(e));
        None
    };

    let config = builder.build()?;

    // Run the download using a new Tokio runtime.
    let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
        path: PathBuf::from("<runtime>"),
        source: e,
    })?;

    let start = Instant::now();

    if flat {
        // --flat: download to cache, then copy to flat layout.
        let repo_id_for_snapshot = repo_id.clone();
        let outcome = rt.block_on(hf_fetch_model::download_files_with_config(repo_id, &config))?;
        let elapsed = start.elapsed();

        if let Some(ref p) = indicatif {
            p.finish();
        }

        // Snapshot sidecar — same best-effort write as the non-flat branch.
        if let Err(e) = write_download_snapshot(
            repo_id_for_snapshot.as_str(), // BORROW: explicit .as_str()
            args.preset.as_ref(),
            &args.filter,
            &args.exclude,
            args.revision.as_deref(), // BORROW: explicit .as_deref()
        ) {
            eprintln!("warning: could not write snapshot sidecar: {e}");
        }

        let file_map = outcome.inner();
        let target_dir = resolve_flat_target(flat_target.as_deref())?;
        let flat_paths = flatten_files(file_map, &target_dir)?;

        println!(
            "{} {} copied to {}:",
            flat_paths.len(),
            pluralize(flat_paths.len(), "file", "files"),
            target_dir.display()
        );
        for p in &flat_paths {
            println!("  {}", p.display());
        }
        print_download_summary(&target_dir, elapsed);
    } else {
        let repo_id_for_snapshot = repo_id.clone();
        let outcome = rt.block_on(hf_fetch_model::download_with_config(repo_id, &config))?;
        let elapsed = start.elapsed();

        // Finalize progress bar before printing to avoid interleaved output.
        if let Some(ref p) = indicatif {
            p.finish();
        }

        // Persist the per-repo `.hf-fm-snapshot.json` sidecar recording the
        // preset / filter / exclude that produced this cache state. `status`
        // later consults it to distinguish deliberately-skipped files from
        // genuinely-missing ones. Best-effort: snapshot failures are logged
        // to stderr but do not fail the download.
        if let Err(e) = write_download_snapshot(
            repo_id_for_snapshot.as_str(), // BORROW: explicit .as_str()
            args.preset.as_ref(),
            &args.filter,
            &args.exclude,
            args.revision.as_deref(), // BORROW: explicit .as_deref()
        ) {
            eprintln!("warning: could not write snapshot sidecar: {e}");
        }

        if outcome.is_cached() {
            println!("Cached at: {}", outcome.inner().display());
        } else {
            println!("Downloaded to: {}", outcome.inner().display());
            print_download_summary(outcome.inner(), elapsed);
        }
    }
    Ok(())
}

/// Writes the per-repo `.hf-fm-snapshot.json` sidecar capturing the active
/// `--preset` / `--filter` / `--exclude` arguments for later consumption by
/// `hf-fm status`. Best-effort — propagates errors so the caller can decide
/// whether to warn (current callers warn but do not abort).
fn write_download_snapshot(
    repo_id: &str,
    preset: Option<&Preset>,
    filter: &[String],
    exclude: &[String],
    revision: Option<&str>,
) -> Result<(), FetchError> {
    let cache_root = cache::hf_cache_dir()?;
    let repo_dir = hf_fetch_model::cache_layout::repo_dir(&cache_root, repo_id);
    let snapshot = cache::Snapshot {
        version: cache::SNAPSHOT_VERSION,
        // BORROW: explicit .to_owned() — Snapshot owns its strings
        revision: revision.unwrap_or("main").to_owned(),
        preset: preset.map(|p| preset_name(p).to_owned()),
        filter: filter.to_vec(),
        exclude: exclude.to_vec(),
    };
    cache::write_snapshot(&repo_dir, &snapshot)
}

/// Displays a download plan without downloading anything.
fn run_dry_run(repo_id: &str, args: &DownloadArgs) -> Result<(), FetchError> {
    // Build FetchConfig from CLI args (same builder logic, minus on_progress).
    let mut builder = match args.preset {
        Some(Preset::Safetensors) => Filter::safetensors(),
        Some(Preset::Gguf) => Filter::gguf(),
        Some(Preset::Npz) => Filter::npz(),
        Some(Preset::Pth) => Filter::pth(),
        Some(Preset::ConfigOnly) => Filter::config_only(),
        None => FetchConfig::builder(),
    };

    if let Some(ref preset) = args.preset {
        warn_redundant_filters(preset, &args.filter);
    }

    if let Some(rev) = args.revision.as_deref() {
        builder = builder.revision(rev);
    }
    if let Some(tok) = args.token.as_deref() {
        builder = builder.token(tok);
    } else {
        builder = builder.token_from_env();
    }
    for pattern in &args.filter {
        // BORROW: explicit .as_str() instead of Deref coercion
        builder = builder.filter(pattern.as_str());
    }
    for pattern in &args.exclude {
        // BORROW: explicit .as_str() instead of Deref coercion
        builder = builder.exclude(pattern.as_str());
    }
    if let Some(ref dir) = args.output_dir {
        // BORROW: explicit .clone() for owned PathBuf
        builder = builder.output_dir(dir.clone());
    }

    let config = builder.build()?;

    let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
        path: PathBuf::from("<runtime>"),
        source: e,
    })?;

    let plan = rt.block_on(hf_fetch_model::download_plan(repo_id, &config))?;

    // Display header.
    println!("  Repo:     {}", plan.repo_id);
    println!("  Revision: {}", plan.revision);
    if args.preset.is_some() || !args.filter.is_empty() {
        println!("  Filter:   active (preset or --filter)");
    }
    if args.flat {
        let target = resolve_flat_target(args.output_dir.as_deref())?;
        println!(
            "  Flat:     {} (files will be copied here)",
            target.display()
        );
    }
    println!();

    render_download_plan(&plan)
}

/// Renders a [`DownloadPlan`]'s file table, byte summary, and (when not fully
/// cached) the recommended [`FetchConfig`] tuning block.
///
/// The shared body behind both `download --dry-run` ([`run_dry_run`]) and
/// `download-file --dry-run` ([`run_download_file_dry_run`]); each caller prints
/// its own header before delegating here so the table/summary stay identical.
///
/// # Errors
///
/// Returns [`FetchError::InvalidPattern`] if [`DownloadPlan::recommended_config`]
/// fails (should not happen — the recommended builder sets no glob patterns).
fn render_download_plan(plan: &DownloadPlan) -> Result<(), FetchError> {
    // Display file table.
    let fw = plan
        .files
        .iter()
        .map(|fp| fp.filename.len())
        .max()
        .unwrap_or(4)
        .max(4); // BORROW: "File".len()
    let row_width = fw + 2 + 10 + 2 + 11;
    println!("  {:<fw$} {:>10}  Status", "File", "Size");
    println!(
        "  {:\u{2500}<fw$} {:\u{2500}<10}  {:\u{2500}<12}",
        "", "", ""
    );
    for fp in &plan.files {
        let status = if fp.cached {
            "cached \u{2713}"
        } else {
            "to download"
        };
        println!(
            "  {:<fw$} {:>10}  {status}",
            fp.filename,
            format_size(fp.size)
        );
    }

    // Summary.
    println!("{:\u{2500}<row_width$}", "  ");
    let cached_count = plan.files.len() - plan.files_to_download();
    let to_dl = plan.files_to_download();
    println!(
        "  Total: {} ({} {}, {} cached, {} to download)",
        format_size(plan.total_bytes),
        plan.files.len(),
        pluralize(plan.files.len(), "file", "files"),
        cached_count,
        to_dl
    );
    println!("  Download: {}", format_size(plan.download_bytes));

    // Recommended config.
    if !plan.fully_cached() {
        let rec = plan.recommended_config()?;
        println!();
        println!("  Recommended config:");
        println!("    concurrency:        {}", rec.concurrency());
        println!("    connections/file:   {}", rec.connections_per_file());
        if rec.chunk_threshold() == u64::MAX {
            println!("    chunk threshold:  disabled (single-connection per file)");
        } else {
            println!(
                "    chunk threshold:  {} MiB",
                rec.chunk_threshold() / 1_048_576
            );
        }
    }

    Ok(())
}

/// Bundles CLI arguments for `download-file` to avoid too-many-arguments lint.
struct DownloadFileParams<'a> {
    repo_id: &'a str,
    filename: &'a str,
    revision: Option<&'a str>,
    token: Option<&'a str>,
    output_dir: Option<PathBuf>,
    chunk_threshold_mib: Option<u64>,
    connections_per_file: Option<usize>,
    timeout_per_file_secs: Option<u64>,
    timeout_total_secs: Option<u64>,
    dry_run: bool,
    flat: bool,
}

fn run_download_file(params: DownloadFileParams<'_>) -> Result<(), FetchError> {
    let DownloadFileParams {
        repo_id,
        filename,
        revision,
        token,
        output_dir,
        chunk_threshold_mib,
        connections_per_file,
        timeout_per_file_secs,
        timeout_total_secs,
        dry_run,
        flat,
    } = params;
    if !repo_id.contains('/') {
        return Err(FetchError::InvalidArgument(format!(
            "invalid REPO_ID \"{repo_id}\": expected \"org/model\" format (e.g., \"mntss/clt-gemma-2-2b-426k\")"
        )));
    }

    // Preview only — resolve the file list (single or glob) and render the
    // dry-run table without downloading. Handled before the glob dispatch so
    // both forms share one entry point.
    if dry_run {
        return run_download_file_dry_run(repo_id, filename, revision, token, output_dir, flat);
    }

    // Glob pattern: list repo files, filter, and download each match.
    if has_glob_chars(filename) {
        return run_download_file_glob(DownloadFileParams {
            repo_id,
            filename,
            revision,
            token,
            output_dir,
            chunk_threshold_mib,
            connections_per_file,
            timeout_per_file_secs,
            timeout_total_secs,
            dry_run,
            flat,
        });
    }

    // When --flat, output_dir is the flat copy target, not the HF cache root.
    // BORROW: explicit .clone() for owned Option<PathBuf>
    let flat_target = if flat { output_dir.clone() } else { None };

    // Build FetchConfig from CLI args.
    let mut builder = FetchConfig::builder();

    if let Some(rev) = revision {
        builder = builder.revision(rev);
    }
    if let Some(tok) = token {
        builder = builder.token(tok);
    } else {
        builder = builder.token_from_env();
    }
    if let Some(ct) = chunk_threshold_mib {
        builder = builder.chunk_threshold(ct.saturating_mul(1024 * 1024));
    }
    if let Some(cpf) = connections_per_file {
        builder = builder.connections_per_file(cpf);
    }
    builder = apply_timeout_overrides(builder, timeout_per_file_secs, timeout_total_secs);
    if !flat {
        if let Some(dir) = output_dir {
            builder = builder.output_dir(dir);
        }
    }

    // Set up progress reporting: indicatif bars for TTY, periodic stderr for non-TTY.
    let is_tty = std::io::stderr().is_terminal();
    let indicatif = if is_tty {
        let p = Arc::new(IndicatifProgress::new());
        let handle = Arc::clone(&p);
        builder = builder.on_progress(move |e| handle.handle(e));
        Some(p)
    } else {
        let p = Arc::new(NonTtyProgress::new());
        let handle = Arc::clone(&p);
        builder = builder.on_progress(move |e| handle.handle(e));
        None
    };

    let config = builder.build()?;

    // Run the download using a new Tokio runtime.
    let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
        path: PathBuf::from("<runtime>"),
        source: e,
    })?;

    // BORROW: explicit .to_owned() for &str → owned String
    let start = Instant::now();
    let outcome = rt.block_on(hf_fetch_model::download_file(
        repo_id.to_owned(),
        filename,
        &config,
    ))?;
    let elapsed = start.elapsed();

    // Finalize progress bar before printing to avoid interleaved output.
    if let Some(ref p) = indicatif {
        p.finish();
    }

    if flat {
        let target_dir = resolve_flat_target(flat_target.as_deref())?;
        let flat_path = flatten_single_file(outcome.inner(), &target_dir)?;
        println!("Copied to: {}", flat_path.display());
    } else if outcome.is_cached() {
        println!("Cached at: {}", outcome.inner().display());
    } else {
        println!("Downloaded to: {}", outcome.inner().display());
        print_download_summary(outcome.inner(), elapsed);
    }
    Ok(())
}

/// Renders the `download-file --dry-run` preview for a single file or glob
/// without downloading.
///
/// Resolves the target list by passing `filename` (a literal name or a glob) as
/// the lone include filter to `download_plan`, mirroring what a real
/// `download-file` would fetch, then prints a tailored `Repo` / `Revision`
/// (+ `Flat`) header and delegates to [`render_download_plan`] for the shared
/// file table. An explicit filename matching nothing is an error; a glob
/// matching nothing prints a notice and succeeds — matching the real paths.
///
/// # Errors
///
/// Returns [`FetchError::InvalidArgument`] when an explicit (non-glob)
/// `filename` matches no file in the repository.
/// Returns [`FetchError::Http`] if the `HuggingFace` API listing request fails.
/// Returns [`FetchError::Io`] if the async runtime cannot be created.
fn run_download_file_dry_run(
    repo_id: &str,
    filename: &str,
    revision: Option<&str>,
    token: Option<&str>,
    output_dir: Option<PathBuf>,
    flat: bool,
) -> Result<(), FetchError> {
    // The filename (literal name or glob) is the lone include filter, mirroring
    // what a real download-file would fetch.
    let mut builder = FetchConfig::builder().filter(filename);

    if let Some(rev) = revision {
        builder = builder.revision(rev);
    }
    if let Some(tok) = token {
        builder = builder.token(tok);
    } else {
        builder = builder.token_from_env();
    }

    // When --flat, output_dir is the flat copy target, not the HF cache root,
    // so the plan keeps the default cache root in that case.
    // BORROW: explicit .clone() for owned Option<PathBuf>
    let flat_target = if flat { output_dir.clone() } else { None };
    if !flat {
        if let Some(dir) = output_dir {
            builder = builder.output_dir(dir);
        }
    }

    let config = builder.build()?;

    let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
        path: PathBuf::from("<runtime>"),
        source: e,
    })?;

    let plan = rt.block_on(hf_fetch_model::download_plan(repo_id, &config))?;

    if plan.files.is_empty() {
        if has_glob_chars(filename) {
            println!("No files matched pattern \"{filename}\" in {repo_id}");
            return Ok(());
        }
        return Err(FetchError::InvalidArgument(format!(
            "\"{filename}\" not found in {repo_id}"
        )));
    }

    // Tailored header — no preset/--filter line; download-file's filename is
    // positional, not a glob-filter set.
    println!("  Repo:     {}", plan.repo_id);
    println!("  Revision: {}", plan.revision);
    if flat {
        let target = resolve_flat_target(flat_target.as_deref())?;
        println!(
            "  Flat:     {} (files will be copied here)",
            target.display()
        );
    }
    println!();

    render_download_plan(&plan)
}

/// Downloads files matching a glob pattern from a repository.
///
/// Lists all remote files, filters by the glob, and downloads each match
/// using the multi-file download pipeline.
fn run_download_file_glob(params: DownloadFileParams<'_>) -> Result<(), FetchError> {
    let DownloadFileParams {
        repo_id,
        filename: pattern,
        revision,
        token,
        output_dir,
        chunk_threshold_mib,
        connections_per_file,
        timeout_per_file_secs,
        timeout_total_secs,
        // EXPLICIT: dry-run is intercepted in run_download_file before this glob
        // path; only real downloads reach here.
        dry_run: _,
        flat,
    } = params;
    // When --flat, output_dir is the flat copy target, not the HF cache root.
    // BORROW: explicit .clone() for owned Option<PathBuf>
    let flat_target = if flat { output_dir.clone() } else { None };

    // Build FetchConfig with the glob pattern as an include filter.
    let mut builder = FetchConfig::builder().filter(pattern);

    if let Some(rev) = revision {
        builder = builder.revision(rev);
    }
    if let Some(tok) = token {
        builder = builder.token(tok);
    } else {
        builder = builder.token_from_env();
    }
    if let Some(ct) = chunk_threshold_mib {
        builder = builder.chunk_threshold(ct.saturating_mul(1024 * 1024));
    }
    if let Some(cpf) = connections_per_file {
        builder = builder.connections_per_file(cpf);
    }
    builder = apply_timeout_overrides(builder, timeout_per_file_secs, timeout_total_secs);
    if !flat {
        if let Some(dir) = output_dir {
            builder = builder.output_dir(dir);
        }
    }

    // Set up progress reporting: indicatif bars for TTY, periodic stderr for non-TTY.
    let is_tty = std::io::stderr().is_terminal();
    let indicatif = if is_tty {
        let p = Arc::new(IndicatifProgress::new());
        let handle = Arc::clone(&p);
        builder = builder.on_progress(move |e| handle.handle(e));
        Some(p)
    } else {
        let p = Arc::new(NonTtyProgress::new());
        let handle = Arc::clone(&p);
        builder = builder.on_progress(move |e| handle.handle(e));
        None
    };

    let config = builder.build()?;

    let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
        path: PathBuf::from("<runtime>"),
        source: e,
    })?;

    // BORROW: explicit .to_owned() for &str → owned String
    let start = Instant::now();
    let outcome = rt.block_on(hf_fetch_model::download_files_with_config(
        repo_id.to_owned(),
        &config,
    ))?;
    let elapsed = start.elapsed();

    // Finalize progress bar before printing to avoid interleaved output.
    if let Some(ref p) = indicatif {
        p.finish();
    }

    let file_map = outcome.inner();
    if file_map.is_empty() {
        println!("No files matched pattern \"{pattern}\" in {repo_id}");
        return Ok(());
    }

    if flat {
        let target_dir = resolve_flat_target(flat_target.as_deref())?;
        let flat_paths = flatten_files(file_map, &target_dir)?;
        println!(
            "{} {} copied to {}:",
            flat_paths.len(),
            pluralize(flat_paths.len(), "file", "files"),
            target_dir.display()
        );
        for p in &flat_paths {
            println!("  {}", p.display());
        }
    } else {
        println!(
            "{} {} matched pattern \"{pattern}\":",
            file_map.len(),
            pluralize(file_map.len(), "file", "files")
        );
        for (name, path) in file_map {
            println!("  {name}: {}", path.display());
        }
    }

    // Summarize total download time.
    let elapsed_secs = elapsed.as_secs_f64();
    if elapsed_secs > 0.0 {
        println!("  completed in {elapsed_secs:.1}s");
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn run_list_families(show: &[ShowFamiliesColumn], tag: Option<&str>) -> Result<(), FetchError> {
    let show_quant = show.contains(&ShowFamiliesColumn::Quant);

    let cache_dir = cache::hf_cache_dir()?;
    let mut families = cache::list_cached_families()?;

    println!("Cache: {}", cache_dir.display());
    println!();

    if families.is_empty() {
        println!("No model families found in local cache.");
        return Ok(());
    }

    // Apply --tag filter (one HTTP request per cached repo, bounded to 8 concurrent).
    if let Some(tag_filter) = tag {
        let lower_tag = tag_filter.to_lowercase();
        // BORROW: explicit .clone() — fetch_tags_concurrent takes Vec<String>
        // because each spawned task needs an owned `'static` String.
        let repo_ids: Vec<String> = families
            .values()
            .flat_map(|entries| entries.iter().map(|e| e.repo_id.clone()))
            .collect();

        let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
            path: PathBuf::from("<runtime>"),
            source: e,
        })?;
        let tags_by_repo = rt.block_on(discover::fetch_tags_concurrent(repo_ids));

        for entries in families.values_mut() {
            entries.retain(|entry| {
                // BORROW: explicit .as_str() instead of Deref coercion
                let Some(tags) = tags_by_repo.get(entry.repo_id.as_str()) else {
                    return false;
                };
                tags.iter()
                    // BORROW: explicit .as_str() instead of Deref coercion
                    .any(|t| t.eq_ignore_ascii_case(lower_tag.as_str()))
            });
        }
        // Prune empty families.
        families.retain(|_, entries| !entries.is_empty());

        if families.is_empty() {
            println!("No cached families match tag {tag_filter:?}.");
            return Ok(());
        }
    }

    // Column widths.
    let fw = families
        .keys()
        .map(String::len)
        .max()
        .unwrap_or(6)
        .max(6) // BORROW: "Family".len()
        + 2;
    let mw = families
        .values()
        .flat_map(|entries| entries.iter().map(|e| e.repo_id.len()))
        .max()
        .unwrap_or(6)
        .max(6); // BORROW: "Models".len()
                 // BORROW: explicit .as_deref() in the closure below for Option<String> → Option<&str>
    let qw = if show_quant {
        families
            .values()
            .flat_map(|entries| {
                entries
                    .iter()
                    .map(|e| e.quant_method.as_deref().unwrap_or("\u{2014}").len())
            })
            .max()
            .unwrap_or(5)
            .max(5) // BORROW: "Quant".len()
            + 2
    } else {
        0
    };

    // Header.
    if show_quant {
        println!("{:<fw$}{:<qw$}Models", "Family", "Quant");
        println!("{:-<fw$}{:-<qw$}{:-<mw$}", "", "", "");
    } else {
        println!("{:<fw$}Models", "Family");
        println!("{:-<fw$}{:-<mw$}", "", "");
    }

    // Body.
    for (model_type, entries) in &families {
        for (i, entry) in entries.iter().enumerate() {
            // BORROW: explicit .as_deref() for Option<String> → Option<&str>
            let quant_cell = entry.quant_method.as_deref().unwrap_or("\u{2014}");
            let family_cell = if i == 0 { model_type.as_str() } else { "" }; // BORROW: explicit .as_str()
            if show_quant {
                println!("{family_cell:<fw$}{quant_cell:<qw$}{}", entry.repo_id);
            } else {
                println!("{family_cell:<fw$}{}", entry.repo_id);
            }
        }
    }

    Ok(())
}

fn run_discover(limit: usize, tag: Option<&str>) -> Result<(), FetchError> {
    let families = cache::list_cached_families()?;
    let local_types: HashSet<String> = families.into_keys().collect();

    let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
        path: PathBuf::from("<runtime>"),
        source: e,
    })?;

    let discovered = rt.block_on(discover::discover_new_families(&local_types, limit, tag))?;

    if discovered.is_empty() {
        match tag {
            Some(t) => println!("No new model families found with tag {t:?}."),
            None => println!("No new model families found."),
        }
        return Ok(());
    }

    match tag {
        Some(t) => {
            println!("New families with tag {t:?} not in local cache (top models by downloads):\n");
        }
        None => println!("New families not in local cache (top models by downloads):\n"),
    }
    let fw = discovered
        .iter()
        .map(|f| f.model_type.len())
        .max()
        .unwrap_or(6)
        .max(6) // BORROW: "Family".len()
        + 2;
    let mw = discovered
        .iter()
        .map(|f| f.top_model.len())
        .max()
        .unwrap_or(9)
        .max(9); // BORROW: "Top Model".len()
    println!("{:<fw$}Top Model", "Family");
    println!("{:-<fw$}{:-<mw$}", "", "");
    for family in &discovered {
        println!("{:<fw$}{}", family.model_type, family.top_model);
    }

    Ok(())
}

#[allow(clippy::too_many_lines)]
fn run_search(
    query: &str,
    limit: usize,
    exact: bool,
    library: Option<&str>,
    pipeline: Option<&str>,
    tag: Option<&str>,
    show: &[ShowColumn],
) -> Result<(), FetchError> {
    let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
        path: PathBuf::from("<runtime>"),
        source: e,
    })?;

    let show_tags = show.contains(&ShowColumn::Tags);
    let show_size = show.contains(&ShowColumn::Size);

    // When the query contains commas, treat both `,` and `/` as term separators
    // so that "mistralai/3B,12" becomes ["mistralai", "3B", "12"].
    // Without commas, just normalize `/` to space for the API query
    // so that "mistralai/3B" becomes "mistralai 3B" (broader API matching).
    let has_commas = query.contains(',');
    let normalized = if has_commas {
        query.replace('/', ",")
    } else {
        query.replace('/', " ")
    };

    // Split on `,` for multi-term filtering; first term goes to the API,
    // all terms are used for client-side filtering.
    let terms: Vec<&str> = normalized
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .collect();

    let api_query = terms.first().copied().unwrap_or(normalized.as_str()); // BORROW: explicit .as_str()

    let filter_terms: Vec<String> = terms.iter().map(|t| t.to_lowercase()).collect();

    // Oversample when filtering: request more from API to compensate for
    // client-side filtering that will discard non-matching results.
    let has_client_filter =
        filter_terms.len() > 1 || library.is_some() || pipeline.is_some() || tag.is_some();
    let api_limit = if has_client_filter {
        limit.saturating_mul(5)
    } else {
        limit
    };

    let results = rt.block_on(discover::search_models(
        api_query, api_limit, library, pipeline, tag,
    ))?;

    // Client-side filtering: only applied when there are multiple comma-separated
    // terms. Single-term queries trust the API results as-is.
    // Model IDs are normalized the same way as the query (slash → space) so that
    // mixed queries like "mistralai/3B,12" match "mistralai/Ministral-3-3B...".
    let has_multi_term = filter_terms.len() > 1;

    // Pre-normalize model IDs once (avoids re-allocating per result per term).
    let normalized_ids: Vec<String> = if has_multi_term {
        results
            .iter()
            .map(|r| r.model_id.replace('/', " ").to_lowercase())
            .collect()
    } else {
        Vec::new()
    };

    let filtered: Vec<&discover::SearchResult> = results
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            if !has_multi_term {
                return true;
            }
            // INDEX: i is bounded by results.len() via enumerate()
            #[allow(clippy::indexing_slicing)]
            let id_normalized = &normalized_ids[*i];
            filter_terms
                .iter()
                .all(|term| id_normalized.contains(term.as_str())) // BORROW: explicit .as_str()
        })
        .map(|(_, r)| r)
        .take(limit)
        .collect();

    // Fan out repo-size lookups (one HTTP request per result) when
    // `--show size` is set. Per-repo failures are silently dropped from the
    // map; rows whose `model_id` is absent render with "—" in the size cell.
    let size_by_repo: HashMap<String, u64> = if show_size && !filtered.is_empty() {
        // BORROW: explicit .clone() — fetch_repo_sizes_concurrent takes Vec<String>
        // because each spawned task needs an owned `'static` String.
        let repo_ids: Vec<String> = filtered.iter().map(|r| r.model_id.clone()).collect();
        rt.block_on(discover::fetch_repo_sizes_concurrent(repo_ids))
    } else {
        HashMap::new()
    };

    if exact {
        // Exact match: compare against the original query (not normalized)
        let exact_match = filtered
            .iter()
            .find(|r| r.model_id.eq_ignore_ascii_case(query));

        if let Some(matched) = exact_match {
            println!("Exact match:\n");
            let size_bytes = size_by_repo.get(matched.model_id.as_str()).copied(); // BORROW: explicit .as_str()
            print_search_result(
                matched,
                matched.model_id.len(),
                show_tags,
                show_size,
                size_bytes,
            );

            // Fetch and display model card metadata
            match rt.block_on(discover::fetch_model_card(
                matched.model_id.as_str(), // BORROW: explicit .as_str()
            )) {
                Ok(card) => print_model_card(&card),
                Err(e) => eprintln!("\n  (could not fetch model card: {e})"),
            }
            // Discoverability cross-link: users who land on --exact often want
            // the longer-form view (full README, license fields) — point them
            // at `info` so they don't have to guess the next subcommand.
            println!("\n  See also: hf-fm info {}", matched.model_id);
        } else {
            println!("No exact match for \"{query}\".");
            if !filtered.is_empty() {
                println!("\nDid you mean:\n");
                let nw = filtered.iter().map(|r| r.model_id.len()).max().unwrap_or(0);
                for result in &filtered {
                    let size_bytes = size_by_repo.get(result.model_id.as_str()).copied(); // BORROW: explicit .as_str()
                    print_search_result(result, nw, show_tags, show_size, size_bytes);
                }
            }
        }
    } else {
        // Normal search display
        if filtered.is_empty() {
            println!("No models found matching \"{query}\".");
        } else {
            let nw = filtered.iter().map(|r| r.model_id.len()).max().unwrap_or(0);
            println!("Models matching \"{query}\" (by downloads):\n");
            for result in &filtered {
                let size_bytes = size_by_repo.get(result.model_id.as_str()).copied(); // BORROW: explicit .as_str()
                print_search_result(result, nw, show_tags, show_size, size_bytes);
            }
        }
    }

    Ok(())
}

fn print_search_result(
    result: &discover::SearchResult,
    name_width: usize,
    show_tags: bool,
    show_size: bool,
    size_bytes: Option<u64>,
) {
    let suffix = match (&result.library_name, &result.pipeline_tag) {
        (Some(lib), Some(pipe)) => format!("  [{lib}, {pipe}]"),
        (Some(lib), None) => format!("  [{lib}]"),
        (None, Some(pipe)) => format!("  [{pipe}]"),
        (None, None) => String::new(),
    };
    // EXPLICIT: inline u64 == 1 check instead of pluralize(usize, …) — a
    // u64 → usize cast on 32-bit platforms could turn a value like 2^32+1
    // into 1 and produce the wrong word for a popular model.
    let downloads_label = if result.downloads == 1 {
        "download"
    } else {
        "downloads"
    };
    // Size column: predictable width, placed before the variable-width tag list.
    // Renders the formatted size when known, "—" when --show size was requested
    // but the per-repo lookup failed, and nothing at all when --show size is off.
    let size_col = if show_size {
        match size_bytes {
            Some(bytes) => format!("  {}", format_size(bytes)),
            None => "  \u{2014}".to_owned(), // BORROW: explicit .to_owned() — em-dash placeholder
        }
    } else {
        String::new()
    };
    // Tag column: shown verbatim when --show tags is set (the user opted in).
    let tags_col = if show_tags && !result.tags.is_empty() {
        format!("  tags: {}", result.tags.join(", "))
    } else {
        String::new()
    };
    println!(
        "  hf-fm {:<nw$} ({} {downloads_label}){suffix}{size_col}{tags_col}",
        result.model_id,
        format_downloads(result.downloads),
        nw = name_width,
    );
}

fn print_model_card(card: &discover::ModelCardMetadata) {
    println!();
    if let Some(ref license) = card.license {
        println!("  License:      {license}");
    }
    if card.gated.is_gated() {
        println!(
            "  Gated:        {} (requires accepting terms on HF)",
            card.gated
        );
    }
    if let Some(ref pipeline) = card.pipeline_tag {
        println!("  Pipeline:     {pipeline}");
    }
    if let Some(ref library) = card.library_name {
        println!("  Library:      {library}");
    }
    if !card.tags.is_empty() {
        println!("  Tags:         {}", card.tags.join(", "));
    }
    if !card.languages.is_empty() {
        println!("  Languages:    {}", card.languages.join(", "));
    }
}

/// Displays model card metadata and README text for a repository.
fn run_info(
    repo_id: &str,
    revision: Option<&str>,
    token: Option<&str>,
    json: bool,
    max_lines: usize,
) -> Result<(), FetchError> {
    if !repo_id.contains('/') {
        return Err(FetchError::InvalidArgument(format!(
            "invalid REPO_ID \"{repo_id}\": expected \"owner/model\" format \
             (e.g., \"mistralai/Ministral-3-3B-Instruct-2512\")"
        )));
    }

    // BORROW: explicit String::from for Option<&str> → Option<String>
    let token_owned = token
        .map(String::from)
        .or_else(|| std::env::var("HF_TOKEN").ok());

    let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
        path: PathBuf::from("<runtime>"),
        source: e,
    })?;

    let card = rt.block_on(discover::fetch_model_card(repo_id))?;
    // BORROW: explicit .as_deref() for Option<String> → Option<&str>
    let readme = rt.block_on(discover::fetch_readme(
        repo_id,
        revision,
        token_owned.as_deref(),
    ))?;

    if json {
        return print_info_json(repo_id, &card, readme.as_deref());
    }

    // Human-readable output.
    println!("  Repo: {repo_id}");
    print_model_card(&card);

    if let Some(ref text) = readme {
        // Strip YAML front matter (--- ... ---) since the structured metadata
        // is already displayed above via print_model_card().
        let body = strip_yaml_front_matter(text);

        println!();
        println!("  README:");
        if looks_like_default_template(body) {
            println!("  Note: README appears to be the HuggingFace default template (low information density).");
        }
        println!("  {}", "\u{2500}".repeat(70));
        let lines: Vec<&str> = body.lines().collect();
        let display_count = if max_lines == 0 {
            lines.len()
        } else {
            lines.len().min(max_lines)
        };
        // INDEX: display_count bounded by lines.len() computed above
        #[allow(clippy::indexing_slicing)]
        for line in &lines[..display_count] {
            println!("  {line}");
        }
        if display_count < lines.len() {
            println!(
                "  ... ({} more lines, use --lines 0 for full output)",
                lines.len().saturating_sub(display_count)
            );
        }
    } else {
        println!();
        println!("  (no README.md found)");
    }

    Ok(())
}

/// Serializable model info for `--json` output.
#[derive(serde::Serialize)]
struct InfoResult {
    /// Repository identifier.
    repo_id: String,
    /// SPDX license identifier, if present.
    #[serde(skip_serializing_if = "Option::is_none")]
    license: Option<String>,
    /// Pipeline task tag, if present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pipeline_tag: Option<String>,
    /// Library framework name, if present.
    #[serde(skip_serializing_if = "Option::is_none")]
    library_name: Option<String>,
    /// Tags from the model card.
    tags: Vec<String>,
    /// Supported languages.
    languages: Vec<String>,
    /// Access control status.
    gated: String,
    /// Full README text, if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    readme: Option<String>,
}

/// Prints model info as JSON.
fn print_info_json(
    repo_id: &str,
    card: &discover::ModelCardMetadata,
    readme: Option<&str>,
) -> Result<(), FetchError> {
    let result = InfoResult {
        // BORROW: explicit .to_owned() for &str → owned String
        repo_id: repo_id.to_owned(),
        // BORROW: explicit .clone() for Option<String> and Vec<String> fields
        license: card.license.clone(),
        pipeline_tag: card.pipeline_tag.clone(),
        library_name: card.library_name.clone(),
        tags: card.tags.clone(),
        languages: card.languages.clone(),
        // BORROW: explicit .to_string() for GateStatus → String
        gated: card.gated.to_string(),
        // BORROW: explicit .to_owned() for Option<&str> → Option<String>
        readme: readme.map(str::to_owned),
    };

    let output = serde_json::to_string_pretty(&result)
        .map_err(|e| FetchError::Http(format!("failed to serialize JSON: {e}")))?;
    println!("{output}");
    Ok(())
}

/// Strips YAML front matter (`--- ... ---`) from a README string.
///
/// Returns the content after the closing `---` delimiter, trimmed of
/// leading blank lines. If no front matter is found, returns the
/// original string unchanged.
#[must_use]
fn strip_yaml_front_matter(text: &str) -> &str {
    // BORROW: explicit .trim_start() for &str → &str
    let trimmed = text.trim_start();
    if !trimmed.starts_with("---") {
        return text;
    }
    // Find the closing "---" after the opening one.
    // INDEX: skip first 3 bytes ("---") which are guaranteed present by the check above
    #[allow(clippy::indexing_slicing)]
    let after_open = &trimmed[3..];
    if let Some(close_pos) = after_open.find("\n---") {
        // Skip past the closing "---" and the newline after it.
        // CAST: not needed, all offsets are usize
        let body_start = close_pos + 4; // "\n---".len()
                                        // INDEX: body_start bounded by after_open.len() (find returned a valid position)
        #[allow(clippy::indexing_slicing)]
        let body = after_open[body_start..].trim_start_matches('\n');
        // BORROW: explicit .trim_start_matches() for &str → &str
        return body.trim_start_matches('\r');
    }
    text
}

/// Heuristic: does the README body look like an unmodified `HuggingFace` default template?
///
/// Two conditions, either of which fires:
/// - The first 20 body lines contain the literal `# Model Card for Model ID`
///   heading that ships with the HF Hub default template.
/// - More than 30% of the first 20 body lines are HTML comments (lines whose
///   first non-whitespace characters are `<!--`).
///
/// A `true` return triggers a one-line `Note:` warning above the README dump
/// in [`run_info`]. Misfire cost is low — a slightly customised template
/// keeps the warning but the user still sees the full README.
#[must_use]
fn looks_like_default_template(body: &str) -> bool {
    let lines: Vec<&str> = body.lines().take(20).collect();
    if lines.is_empty() {
        return false;
    }
    if lines
        .iter()
        .any(|l| l.trim() == "# Model Card for Model ID")
    {
        return true;
    }
    let comment_count = lines
        .iter()
        .filter(|l| l.trim_start().starts_with("<!--"))
        .count();
    // Integer arithmetic: comment_count / lines.len() > 0.30 is equivalent to
    // 100 * comment_count > 30 * lines.len() with no float involvement.
    comment_count.saturating_mul(100) > lines.len().saturating_mul(30)
}

/// Renders the `inspect` summary's first size line (the one immediately
/// after `Source:`).
///
/// For safetensors (`is_gguf == false`), preserves the v0.10.2 wording:
/// `Header: <header_size> (JSON), <total> total` (or just
/// `Header: <header_size> (JSON)` when the total isn't known). The `(JSON)`
/// suffix is accurate because safetensors *does* start with a length-prefixed
/// JSON header.
///
/// For GGUF (`is_gguf == true`), drops the safetensors-flavoured `(JSON)`
/// suffix and the `0 B` prefix — GGUF has no length-prefixed JSON header, so
/// the v0.10.2 wording printed a meaningless `0 B (JSON)` for every GGUF
/// invocation. The replacement surfaces the total file size under a `Size:`
/// label (note: `File:` is already used by the preceding filename line, so
/// we cannot reuse it here).
///
/// The two-space indent is the caller's responsibility — this helper returns
/// the label-and-value line without leading whitespace.
#[must_use]
fn format_header_line(header_size: u64, file_size: Option<u64>, is_gguf: bool) -> String {
    if is_gguf {
        match file_size {
            Some(fs) => format!("Size:     {}", format_size(fs)),
            // BORROW: explicit .to_owned() for &str → owned String
            None => "Size:     (size unknown)".to_owned(),
        }
    } else {
        let hd = format_size(header_size);
        match file_size {
            Some(fs) => format!("Header:   {hd} (JSON), {} total", format_size(fs)),
            None => format!("Header:   {hd} (JSON)"),
        }
    }
}

/// Renders the `inspect` summary's `Metadata:` block as a vector of lines
/// (without the leading 2-space indent; the caller prepends it on each line).
///
/// Sorts keys alphabetically and deterministically — eliminates the previous
/// `HashMap` iteration non-determinism between runs, and clusters keys sharing
/// a prefix (`general.*`, `<arch>.*`, `tokenizer.*`, `gguf.*`) automatically.
/// When the metadata has at most `TABULAR_THRESHOLD` keys, returns the v0.10.2
/// single-line form (`Metadata: k=v, k=v, …`). Above the threshold, switches
/// to a tabular block — one `key=value` per indented line — which keeps GGUF's
/// typical 30+ keys readable.
///
/// Values containing newlines (e.g. `tokenizer.chat_template`'s Jinja
/// templates) get a `key=` header line followed by each value-line on its own
/// indented continuation line, instead of wrapping awkwardly inline.
///
/// Returns an empty `Vec` when `meta` is empty so the caller skips the
/// `println!` entirely (preserving the v0.10.2 "no metadata, no line" behavior).
#[must_use]
fn format_metadata_lines(meta: &HashMap<String, String>) -> Vec<String> {
    /// Threshold above which the renderer switches to a tabular block.
    /// Picked so that typical safetensors `__metadata__` (which carries 3–6
    /// quantization keys) keeps the v0.10.2 inline form, while typical GGUF
    /// metadata (30+ scalar keys) gets the tabular form.
    const TABULAR_THRESHOLD: usize = 6;

    if meta.is_empty() {
        return Vec::new();
    }

    // BORROW: explicit .as_str() — sorted projection onto &str borrows
    let mut keys: Vec<&str> = meta.keys().map(String::as_str).collect();
    keys.sort_unstable();

    if keys.len() <= TABULAR_THRESHOLD {
        let entries: Vec<String> = keys
            .iter()
            // BORROW: explicit .map_or("", String::as_str) for Option<&String> → &str
            .map(|k| format!("{k}={}", meta.get(*k).map_or("", String::as_str)))
            .collect();
        return vec![format!("Metadata: {}", entries.join(", "))];
    }

    let mut lines: Vec<String> = Vec::with_capacity(keys.len() + 1);
    // BORROW: explicit .to_owned() for &str → owned String
    lines.push("Metadata:".to_owned());
    for k in &keys {
        // BORROW: explicit .map_or("", String::as_str) for Option<&String> → &str
        let v = meta.get(*k).map_or("", String::as_str);
        if v.contains('\n') {
            lines.push(format!("  {k}="));
            for value_line in v.lines() {
                lines.push(format!("    {value_line}"));
            }
        } else {
            lines.push(format!("  {k}={v}"));
        }
    }
    lines
}

/// Renders the optional `Format:` + `Size:` block in the `inspect` summary
/// for quantized safetensors files. Returns `Vec::new()` (and the caller
/// skips the `println!`s) when `quant_info` is `None` — i.e. for
/// unquantized safetensors, all `GGUF` / `NPZ` / `PTH` paths, and the
/// remote safetensors path (until v0.11.1 lifts it to anamnesis).
///
/// The two-line `Format: …` + `Size: <stored> stored -> <deq> (BF16)` form
/// matches `anamnesis::InspectInfo`'s `Display` idiom while preserving the
/// explicit `Format:` label called out in the v0.10.3 roadmap.
#[must_use]
fn format_quant_lines(quant_info: Option<&inspect::QuantInfo>) -> Vec<String> {
    let Some(q) = quant_info else {
        return Vec::new();
    };
    vec![
        format!("Format:    {}", q.scheme),
        format!(
            "Size:      {} stored -> {} (BF16)",
            format_size(q.stored_bytes),
            format_size(q.dequantized_bytes),
        ),
    ]
}

fn run_status_all(json: bool) -> Result<(), FetchError> {
    let cache_dir = cache::hf_cache_dir()?;
    let summaries = cache::cache_summary()?;

    if json {
        return print_status_all_json(&summaries, &cache_dir);
    }

    if summaries.is_empty() {
        println!("No models found in local cache.");
        return Ok(());
    }

    println!("Cache: {}\n", cache_dir.display());
    let rw = summaries
        .iter()
        .map(|s| s.repo_id.len())
        .max()
        .unwrap_or(10)
        .max(10); // BORROW: "Repository".len()
    println!(
        "  {:<rw$} {:>5}  {:>10}  Status",
        "Repository", "Files", "Size"
    );
    println!("  {:-<rw$} {:-<5}  {:-<10}  {:-<8}", "", "", "", "");

    for s in &summaries {
        let status_label = if s.has_partial { "PARTIAL" } else { "ok" };
        println!(
            "  {:<rw$} {:>5}  {:>10}  {}",
            s.repo_id,
            s.file_count,
            format_size(s.total_size),
            status_label
        );
    }

    println!(
        "\n{} {} cached",
        summaries.len(),
        pluralize(summaries.len(), "model", "models")
    );

    Ok(())
}

/// Resolves a `du` argument to a repo ID.
///
/// If the argument contains `/`, it is treated as a repo ID. If it parses
/// as a number, it is treated as a 1-based index into the size-sorted cache
/// summary. Otherwise, returns an error.
///
/// # Errors
///
/// Returns [`FetchError::InvalidArgument`] if the index is out of range
/// or the argument is not a valid repo ID or numeric index.
fn resolve_du_arg(arg: &str) -> Result<String, FetchError> {
    // Repo ID: contains '/' (e.g., "google/gemma-2-2b-it").
    if arg.contains('/') {
        // BORROW: explicit .to_owned() for &str → owned String
        return Ok(arg.to_owned());
    }

    // Numeric index: resolve against the size-sorted cache summary.
    if let Ok(n) = arg.parse::<usize>() {
        let mut summaries = cache::cache_summary()?;
        summaries.sort_by_key(|s| std::cmp::Reverse(s.total_size));

        if n == 0 || n > summaries.len() {
            return Err(FetchError::InvalidArgument(format!(
                "index {n} is out of range (cache has {} {} — use 1..{})",
                summaries.len(),
                pluralize(summaries.len(), "repo", "repos"),
                summaries.len()
            )));
        }

        // INDEX: n is bounded by 1..=summaries.len() checked above
        // BORROW: explicit .clone() for owned String
        #[allow(clippy::indexing_slicing)]
        return Ok(summaries[n - 1].repo_id.clone());
    }

    Err(FetchError::InvalidArgument(format!(
        "\"{arg}\" is not a valid repo ID (expected \"org/model\") or numeric index"
    )))
}

/// Shows disk usage summary for all cached repos, sorted by size descending.
fn run_du(age: bool, json: bool) -> Result<(), FetchError> {
    let cache_dir = cache::hf_cache_dir()?;

    let mut summaries = cache::cache_summary()?;
    summaries.sort_by_key(|s| std::cmp::Reverse(s.total_size));

    if json {
        return print_du_json(&summaries, &cache_dir);
    }

    println!("Cache: {}\n", cache_dir.display());

    if summaries.is_empty() {
        println!("No models found in local cache.");
        return Ok(());
    }

    // Compute REPO column width from the longest repo ID (minimum 48).
    let repo_width = summaries
        .iter()
        .map(|s| s.repo_id.len())
        .max()
        .unwrap_or(0)
        .max(48);

    if age {
        println!(
            "  {:>3}  {:>10}  {:<repo_width$} {:>5}  {:<15}",
            "#", "SIZE", "REPO", "FILES", "AGE"
        );
    } else {
        println!(
            "  {:>3}  {:>10}  {:<repo_width$} {:>5}",
            "#", "SIZE", "REPO", "FILES"
        );
    }

    let mut total_size: u64 = 0;
    let mut total_files: usize = 0;
    let mut any_partial = false;

    for (i, s) in summaries.iter().enumerate() {
        total_size = total_size.saturating_add(s.total_size);
        total_files = total_files.saturating_add(s.file_count);

        let partial_marker = if s.has_partial {
            any_partial = true;
            "  \u{25cf}"
        } else {
            ""
        };

        if age {
            let age_str = s
                .last_modified
                .map_or_else(|| "\u{2014}".to_owned(), format_age);
            println!(
                "  {:>3}  {:>10}  {:<repo_width$} {:>5}  {:<15}{}",
                i + 1,
                format_size(s.total_size),
                s.repo_id,
                s.file_count,
                age_str,
                partial_marker,
            );
        } else {
            println!(
                "  {:>3}  {:>10}  {:<repo_width$} {:>5}{}",
                i + 1,
                format_size(s.total_size),
                s.repo_id,
                s.file_count,
                partial_marker,
            );
        }
    }

    // 3 (pad) + 2 + 3 (#) + 2 + 10 (SIZE) + 2 + repo_width + 2 + 5 (FILES) = repo_width + 29
    // When --age is active, add 2 (gap) + 15 (AGE column) = 17 extra.
    let rule_width = if age {
        repo_width + 46
    } else {
        repo_width + 29
    };
    println!("  {}", "\u{2500}".repeat(rule_width));
    println!(
        "  {:>10}  total ({} {}, {} {})",
        format_size(total_size),
        summaries.len(),
        pluralize(summaries.len(), "repo", "repos"),
        total_files,
        pluralize(total_files, "file", "files"),
    );
    if any_partial {
        println!("  \u{25cf} = partial downloads");
    }

    Ok(())
}

/// Shows per-file disk usage for a specific cached repo, sorted by size descending.
fn run_du_repo(repo_id: &str, json: bool) -> Result<(), FetchError> {
    let cache_dir = cache::hf_cache_dir()?;
    let files = cache::cache_repo_usage(repo_id)?;
    let has_partial = cache::repo_has_partial(repo_id)?;

    if json {
        return print_du_repo_json(repo_id, &files, has_partial, &cache_dir);
    }

    println!("Cache: {}\n", cache_dir.display());

    if files.is_empty() {
        println!("No cached files found for {repo_id}.");
        return Ok(());
    }

    println!("  {repo_id}:\n");
    let fw = files
        .iter()
        .map(|f| f.filename.len())
        .max()
        .unwrap_or(4)
        .max(4); // BORROW: "FILE".len()
    let row_width = 3 + 2 + 10 + 2 + fw;
    println!("  {:>3}  {:>10}  FILE", "#", "SIZE");

    let mut total_size: u64 = 0;

    for (i, f) in files.iter().enumerate() {
        total_size = total_size.saturating_add(f.size);
        println!(
            "  {:>3}  {:>10}  {}",
            i + 1,
            format_size(f.size),
            f.filename
        );
    }

    println!("  {}", "\u{2500}".repeat(row_width));
    println!(
        "  {:>10}  total ({} {})",
        format_size(total_size),
        files.len(),
        pluralize(files.len(), "file", "files"),
    );

    // Hint the user when this repo has partial downloads (computed above).
    if has_partial {
        println!("\n  \u{25cf} partial downloads — run `hf-fm status {repo_id}` for details");
    }

    Ok(())
}

/// Serializes `value` as pretty JSON to stdout.
///
/// The shared tail of every `--json` printer added in the v0.10.7 parity work.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if serialization fails.
fn emit_json<T: serde::Serialize>(value: &T) -> Result<(), FetchError> {
    let output = serde_json::to_string_pretty(value)
        .map_err(|e| FetchError::Http(format!("failed to serialize JSON: {e}")))?;
    println!("{output}");
    Ok(())
}

/// Converts an optional [`SystemTime`](std::time::SystemTime) to Unix epoch
/// seconds, or `None` when absent or before the epoch.
fn system_time_to_unix(t: Option<std::time::SystemTime>) -> Option<u64> {
    t.and_then(|st| st.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
}

/// Per-file leaf for `du --json`.
#[derive(serde::Serialize)]
struct DuFileJson {
    /// Filename relative to the snapshot root.
    filename: String,
    /// File size in bytes.
    size: u64,
}

/// Per-repo entry for `du --json` (flat) and `du --tree --json` (with `files`).
#[derive(serde::Serialize)]
struct DuRepoJson {
    /// Repository identifier.
    repo_id: String,
    /// Total size on disk in bytes.
    size: u64,
    /// Number of files in the snapshot directory.
    file_count: usize,
    /// Whether the repo has incomplete `.chunked.part` downloads.
    has_partial: bool,
    /// Most recent file mtime as Unix epoch seconds (`null` if unknown).
    last_modified: Option<u64>,
    /// Per-file leaves; present only under `--tree`.
    #[serde(skip_serializing_if = "Option::is_none")]
    files: Option<Vec<DuFileJson>>,
}

/// Top-level shape for `du --json` (all repos) and `du --tree --json`.
#[derive(serde::Serialize)]
struct DuJson {
    /// HF cache directory.
    cache_dir: String,
    /// Per-repo entries, sorted by size descending.
    repos: Vec<DuRepoJson>,
    /// Total bytes across all cached repos.
    total_bytes: u64,
    /// Total file count across all cached repos.
    total_files: usize,
    /// Number of cached repos.
    repo_count: usize,
}

/// Top-level shape for `du <repo> --json` (per-file drill-down).
#[derive(serde::Serialize)]
struct DuRepoDetailJson {
    /// HF cache directory.
    cache_dir: String,
    /// Repository identifier.
    repo_id: String,
    /// Per-file entries, sorted by size descending.
    files: Vec<DuFileJson>,
    /// Total bytes across the repo's files.
    total_bytes: u64,
    /// Number of files.
    file_count: usize,
    /// Whether the repo has incomplete `.chunked.part` downloads.
    has_partial: bool,
}

/// Prints the `du` all-repos summary as JSON (flat — no per-file leaves).
fn print_du_json(
    summaries: &[cache::CachedModelSummary],
    cache_dir: &std::path::Path,
) -> Result<(), FetchError> {
    let mut total_bytes: u64 = 0;
    let mut total_files: usize = 0;
    let mut repos: Vec<DuRepoJson> = Vec::with_capacity(summaries.len());
    // EXPLICIT: accumulates totals alongside entry construction.
    for s in summaries {
        total_bytes = total_bytes.saturating_add(s.total_size);
        total_files = total_files.saturating_add(s.file_count);
        repos.push(DuRepoJson {
            // BORROW: explicit .clone() for owned String field
            repo_id: s.repo_id.clone(),
            size: s.total_size,
            file_count: s.file_count,
            has_partial: s.has_partial,
            last_modified: system_time_to_unix(s.last_modified),
            files: None,
        });
    }
    let result = DuJson {
        // BORROW: explicit .to_string() for Path → String
        cache_dir: cache_dir.display().to_string(),
        repo_count: repos.len(),
        repos,
        total_bytes,
        total_files,
    };
    emit_json(&result)
}

/// Prints the `du --tree` view as JSON (nested — each repo carries its files).
fn print_du_tree_json(
    repos: &[CacheTreeRepo],
    cache_dir: &std::path::Path,
) -> Result<(), FetchError> {
    let mut total_bytes: u64 = 0;
    let mut total_files: usize = 0;
    let mut out: Vec<DuRepoJson> = Vec::with_capacity(repos.len());
    // EXPLICIT: accumulates totals alongside entry construction.
    for r in repos {
        total_bytes = total_bytes.saturating_add(r.total_size);
        total_files = total_files.saturating_add(r.file_count);
        let files = r
            .files
            .iter()
            .map(|f| DuFileJson {
                // BORROW: explicit .clone() for owned String field
                filename: f.filename.clone(),
                size: f.size,
            })
            .collect();
        out.push(DuRepoJson {
            // BORROW: explicit .clone() for owned String field
            repo_id: r.repo_id.clone(),
            size: r.total_size,
            file_count: r.file_count,
            has_partial: r.has_partial,
            last_modified: system_time_to_unix(r.last_modified),
            files: Some(files),
        });
    }
    let result = DuJson {
        // BORROW: explicit .to_string() for Path → String
        cache_dir: cache_dir.display().to_string(),
        repo_count: out.len(),
        repos: out,
        total_bytes,
        total_files,
    };
    emit_json(&result)
}

/// Prints a single repo's per-file disk usage as JSON.
fn print_du_repo_json(
    repo_id: &str,
    files: &[cache::CacheFileUsage],
    has_partial: bool,
    cache_dir: &std::path::Path,
) -> Result<(), FetchError> {
    let mut total_bytes: u64 = 0;
    let mut entries: Vec<DuFileJson> = Vec::with_capacity(files.len());
    // EXPLICIT: accumulates total alongside entry construction.
    for f in files {
        total_bytes = total_bytes.saturating_add(f.size);
        entries.push(DuFileJson {
            // BORROW: explicit .clone() for owned String field
            filename: f.filename.clone(),
            size: f.size,
        });
    }
    let result = DuRepoDetailJson {
        // BORROW: explicit .to_string()/.to_owned() for owned fields
        cache_dir: cache_dir.display().to_string(),
        repo_id: repo_id.to_owned(),
        file_count: entries.len(),
        files: entries,
        total_bytes,
        has_partial,
    };
    emit_json(&result)
}

// ============================================================================
// du --tree: hierarchical cache view (repos as branches, files as leaves)
// ============================================================================

/// Returns `singular` when `n == 1`, otherwise `plural`.
///
/// Grammar helper for runtime-counted nouns:
/// `pluralize(n, "file", "files")`.
const fn pluralize<'a>(n: usize, singular: &'a str, plural: &'a str) -> &'a str {
    if n == 1 {
        singular
    } else {
        plural
    }
}

/// Formats a file-count parenthetical with correct singular/plural form.
///
/// Returns `"(1 file)"` for `n == 1` and `"(N files)"` otherwise.
fn format_file_count(n: usize) -> String {
    format!("({n} {})", pluralize(n, "file", "files"))
}

/// Returns `true` when `name` contains `pattern` as a **case-insensitive**
/// substring.
///
/// The single match predicate behind every `--filter` call site in `inspect`
/// and `diff`, so they all agree with the case-insensitive substring contract
/// `inspect --pick` adopted in v0.10.5 — e.g. `--filter "Layers.0"` now
/// matches `model.layers.0.*`. Both operands are lowercased per call; tensor
/// name lists are bounded (hundreds–thousands of entries), so the per-name
/// allocation is not a hot-path concern.
fn matches_filter(name: &str, pattern: &str) -> bool {
    // BORROW: explicit .as_str() for the lowercased String → &str
    name.to_lowercase()
        .contains(pattern.to_lowercase().as_str())
}

/// Builds a table-of-contents-style dotted filler of the exact requested `width`.
///
/// Pattern: `"  .  .  .  .  ."` — two leading spaces, dots separated by
/// two spaces, and the rightmost dot flush against the trailing edge.
/// When `width` isn't a clean multiple of the 3-char period, the slack
/// is absorbed as extra leading spaces.
fn dot_filler(width: usize) -> String {
    if width < 3 {
        return " ".repeat(width);
    }
    let dots = width / 3;
    let pad = width - 3 * dots;
    let mut s = String::with_capacity(width);
    for _ in 0..(pad + 2) {
        s.push(' ');
    }
    for i in 0..dots {
        if i > 0 {
            s.push_str("  ");
        }
        s.push('.');
    }
    s
}

/// Repo branch in the `du --tree` view.
///
/// Carries the per-repo aggregates already computed by
/// [`cache::cache_summary`] plus the per-file leaves produced by
/// [`cache::cache_repo_usage`]. The tree has a fixed two-level shape
/// (repo → file), so a flat pair of structs is sufficient — no recursive
/// node enum is needed (cf. `inspect --tree`'s `TreeNode`, which is
/// arbitrarily deep).
struct CacheTreeRepo {
    /// Repository identifier (e.g., `"google/gemma-2-2b-it"`).
    repo_id: String,
    /// Total size on disk across all snapshot files, in bytes.
    total_size: u64,
    /// Number of files counted in the snapshot directory.
    file_count: usize,
    /// Whether the repo has any `.chunked.part` partial downloads.
    has_partial: bool,
    /// Most recent file modification time, if available.
    last_modified: Option<std::time::SystemTime>,
    /// Per-file leaves, sorted by size descending.
    files: Vec<CacheTreeFile>,
}

/// File leaf in the `du --tree` view.
struct CacheTreeFile {
    /// Filename relative to the snapshot root (e.g., `"tokenizer/vocab.json"`).
    filename: String,
    /// File size in bytes.
    size: u64,
}

/// Builds a `du --tree` forest from the local cache, sorted by total size descending.
///
/// One [`CacheTreeRepo`] per cached `models--*` directory; its [`CacheTreeFile`]
/// leaves come pre-sorted by size descending from [`cache::cache_repo_usage`].
fn build_cache_tree() -> Result<Vec<CacheTreeRepo>, FetchError> {
    let mut summaries = cache::cache_summary()?;
    summaries.sort_by_key(|s| std::cmp::Reverse(s.total_size));

    let mut repos: Vec<CacheTreeRepo> = Vec::with_capacity(summaries.len());
    for s in summaries {
        // BORROW: explicit .as_str() instead of Deref coercion
        let usage = cache::cache_repo_usage(s.repo_id.as_str())?;
        let files: Vec<CacheTreeFile> = usage
            .into_iter()
            .map(|f| CacheTreeFile {
                filename: f.filename,
                size: f.size,
            })
            .collect();
        repos.push(CacheTreeRepo {
            repo_id: s.repo_id,
            total_size: s.total_size,
            file_count: s.file_count,
            has_partial: s.has_partial,
            last_modified: s.last_modified,
            files,
        });
    }

    Ok(repos)
}

/// Cap on the file-leaf name column, in characters.
///
/// Names ≤ this width are padded to it so the file-leaf size column lines
/// up vertically across every repo in the tree. Names longer than the cap
/// overflow into the size column locally — preserving full info on outlier
/// paths (e.g. the deeply-nested `gemma_2b_blocks.0.…/sae_weights.safetensors`
/// SAE filenames) without dragging every other repo's column to that width.
const FILE_NAME_WIDTH_CAP: usize = 60;

/// Column widths for the cache tree, derived from the actual data.
///
/// Computed once in [`run_du_tree`] and reused by [`render_cache_tree`],
/// [`render_file_leaves`], and the footer rule via
/// [`CacheTreeWidths::rule_width`] — keeping the repo-total size column
/// (on branch lines) and the file size column (on leaf lines) sharing a
/// single vertical column so all sizes right-align across the whole tree,
/// and the rule visually flush with the widest repo branch line.
struct CacheTreeWidths {
    /// Repo-id column on the repo branch line. Padded up beyond the
    /// natural max repo-id length when needed so the size column on the
    /// branch line lines up with the size column on leaf lines.
    repo: usize,
    /// Shared size column width — used by both the repo-total on branch
    /// lines and the file size on leaf lines. Right-aligned in both, so
    /// every size string in the tree shares a common right edge.
    size: usize,
    /// `(N files)` column on the repo branch line.
    files: usize,
    /// Age column on the repo branch line (zero when `--age` is off).
    age: usize,
    /// File-leaf name column. Naturally capped at [`FILE_NAME_WIDTH_CAP`],
    /// but padded up beyond that cap when needed so the leaf size column
    /// lines up with the branch size column.
    file_name: usize,
    /// File-leaf size column. Equal to [`Self::size`] — kept as a separate
    /// field so the renderer reads the right intent at the leaf call site.
    file_size: usize,
}

impl CacheTreeWidths {
    /// Computes column widths from a slice of repo branches.
    ///
    /// `age` toggles whether the age column is sized; when off, the field
    /// is left at zero so the footer-rule formula can ignore it without a
    /// branch. File-leaf widths are computed *globally* across every file
    /// in every repo (so a leaf's size column lines up with all other
    /// leaves in the tree, regardless of which repo it belongs to).
    ///
    /// To align the repo-total size column (on branch lines) with the file
    /// size column (on leaf lines), the repo-id column and file-name
    /// column are padded up so both size columns share the same start
    /// position — with branch lines, the size column starts at
    /// `2 + 4 + repo + 2` and on leaf lines at `2 + 4 + 4 + file_name + 2`,
    /// so `repo = file_name + 4` keeps them flush.
    fn compute(repos: &[CacheTreeRepo], age: bool) -> Self {
        // Floors are tuned to match the visual minimums used by the flat
        // `du` view, so the eye picks up the same shape across both.
        let natural_repo = repos
            .iter()
            .map(|r| r.repo_id.len())
            .max()
            .unwrap_or(0)
            .max(10); // floor: "Repository".len()

        let natural_size = repos
            .iter()
            .map(|r| format_size(r.total_size).len())
            .max()
            .unwrap_or(0)
            .max(8); // floor: typical "x.xx GiB" length

        let files = repos
            .iter()
            .map(|r| format_file_count(r.file_count).len())
            .max()
            .unwrap_or(0);

        let age = if age {
            repos
                .iter()
                // `1` is the em-dash fallback width, used when a repo's
                // mtime cannot be read.
                .map(|r| r.last_modified.map_or(1, |t| format_age(t).len()))
                .max()
                .unwrap_or(0)
                .max(15) // floor: matches `du --age` AGE column
        } else {
            0
        };

        let natural_file_name = repos
            .iter()
            .flat_map(|r| r.files.iter())
            .map(|f| f.filename.len())
            .max()
            .unwrap_or(0)
            .min(FILE_NAME_WIDTH_CAP);

        let natural_file_size = repos
            .iter()
            .flat_map(|r| r.files.iter())
            .map(|f| format_size(f.size).len())
            .max()
            .unwrap_or(0);

        // Align the size column across branch lines and leaf lines.
        // Branch size starts at: 2 + 4 + repo + 2          = repo + 8
        // Leaf   size starts at: 2 + 4 + 4 + file_name + 2 = file_name + 12
        // Pad the shorter side so both columns share the same start, then
        // use a single width for the size column so right-edges also align.
        let size_start = (natural_repo + 8).max(natural_file_name + 12);
        let repo = size_start - 8;
        let file_name = size_start - 12;
        let size = natural_size.max(natural_file_size);

        Self {
            repo,
            size,
            files,
            age,
            file_name,
            file_size: size,
        }
    }

    /// Width of the visual rule under the tree, sized to the widest repo branch.
    ///
    /// Counts the leading 4-char connector (`├── ` or `└── `) plus each
    /// rendered column and the 2-char gaps between them. When `--age` is
    /// off the age column contributes zero.
    fn rule_width(&self) -> usize {
        // 4 (connector) + repo + 2 + size + 2 + files + (2 + age, when active).
        let mut w = 4 + self.repo + 2 + self.size + 2 + self.files;
        if self.age > 0 {
            w += 2 + self.age;
        }
        w
    }
}

/// Renders the cache tree to stdout using Unicode box-drawing connectors.
///
/// `age` controls the optional last-modified column on repo branch lines.
/// File leaves never show age (mirroring `du --age`'s repo-only column).
fn render_cache_tree(repos: &[CacheTreeRepo], widths: &CacheTreeWidths, age: bool) {
    for (i, repo) in repos.iter().enumerate() {
        let is_last = i + 1 == repos.len();
        render_repo_node(repo, is_last, widths, age);
    }
}

/// Renders a single repo branch and its file leaves.
///
/// The gap between the repo id and the size column is rendered as a
/// dotted filler ([`dot_filler`]) — `"  .  .  .  ."` — so the eye can
/// travel from a short repo name to the (possibly far) shared size
/// column. The filler width is `widths.repo + 2 - repo_id.len()`, sized
/// so the size column lands at the column [`CacheTreeWidths::compute`]
/// reserved for it on both branch and leaf lines.
fn render_repo_node(repo: &CacheTreeRepo, is_last: bool, widths: &CacheTreeWidths, age: bool) {
    let connector = if is_last { "└── " } else { "├── " };
    let indent = if is_last { "    " } else { "│   " };

    let size_str = format_size(repo.total_size);
    let files_str = format_file_count(repo.file_count);
    let partial_marker = if repo.has_partial { "  \u{25cf}" } else { "" };

    // Width of the gap between end of repo_id and start of size column.
    // `widths.repo >= repo_id.len()` by construction, so the +2 keeps
    // the saturating_sub a defensive no-op rather than load-bearing.
    let filler = dot_filler((widths.repo + 2).saturating_sub(repo.repo_id.len()));

    if age {
        let age_str = repo
            .last_modified
            .map_or_else(|| "\u{2014}".to_owned(), format_age);
        println!(
            "  {connector}{repo}{filler}{size:>sw$}  {files:<fw$}  {age_str:<aw$}{partial_marker}",
            repo = repo.repo_id,
            size = size_str,
            files = files_str,
            sw = widths.size,
            fw = widths.files,
            aw = widths.age,
        );
    } else {
        println!(
            "  {connector}{repo}{filler}{size:>sw$}  {files}{partial_marker}",
            repo = repo.repo_id,
            size = size_str,
            files = files_str,
            sw = widths.size,
        );
    }

    render_file_leaves(&repo.files, indent, widths);
}

/// Renders the file leaves under a repo branch using the tree-wide name/size widths.
///
/// `widths.file_name` is the global, capped name column (see
/// [`FILE_NAME_WIDTH_CAP`]) — names shorter than it are padded so size
/// columns line up vertically across every repo; names longer than it
/// overflow locally without dragging the other repos' columns out.
fn render_file_leaves(files: &[CacheTreeFile], indent: &str, widths: &CacheTreeWidths) {
    if files.is_empty() {
        return;
    }

    for (i, file) in files.iter().enumerate() {
        let is_last = i + 1 == files.len();
        let connector = if is_last { "└── " } else { "├── " };
        println!(
            "  {indent}{connector}{name:<nw$}  {size:>sw$}",
            name = file.filename,
            size = format_size(file.size),
            nw = widths.file_name,
            sw = widths.file_size,
        );
    }
}

/// Hierarchical cache view: every cached repo and its files in one box-drawing tree.
///
/// Composes with `--age` (adds a last-modified column on repo branches).
fn run_du_tree(age: bool, json: bool) -> Result<(), FetchError> {
    let cache_dir = cache::hf_cache_dir()?;
    let repos = build_cache_tree()?;

    if json {
        return print_du_tree_json(&repos, &cache_dir);
    }

    println!("Cache: {}\n", cache_dir.display());

    if repos.is_empty() {
        println!("No models found in local cache.");
        return Ok(());
    }

    let widths = CacheTreeWidths::compute(&repos, age);
    render_cache_tree(&repos, &widths, age);

    let total_size: u64 = repos
        .iter()
        .map(|r| r.total_size)
        .fold(0_u64, u64::saturating_add);
    let total_files: usize = repos
        .iter()
        .map(|r| r.file_count)
        .fold(0_usize, usize::saturating_add);
    let any_partial = repos.iter().any(|r| r.has_partial);

    println!("\n  {}", "\u{2500}".repeat(widths.rule_width()));
    println!(
        "  {:>10}  total ({} {}, {} {})",
        format_size(total_size),
        repos.len(),
        pluralize(repos.len(), "repo", "repos"),
        total_files,
        pluralize(total_files, "file", "files"),
    );
    if any_partial {
        println!("  \u{25cf} = partial downloads");
    }

    Ok(())
}

/// Removes `.chunked.part` temp files from the `HuggingFace` cache.
///
/// When `repo_filter` is `Some`, only that repo is scanned.
/// With `--dry-run`, prints what would be removed without deleting.
/// With `--yes`, skips the confirmation prompt.
fn run_cache_clean_partial(
    repo_filter: Option<&str>,
    yes: bool,
    dry_run: bool,
) -> Result<(), FetchError> {
    let cache_dir = cache::hf_cache_dir()?;

    if !cache_dir.exists() {
        println!("No HuggingFace cache found at {}", cache_dir.display());
        return Ok(());
    }

    println!("Cache: {}\n", cache_dir.display());

    let partials = cache::find_partial_files(repo_filter)?;

    if partials.is_empty() {
        println!("No partial downloads found.");
        return Ok(());
    }

    let total_size: u64 = partials.iter().map(|p| p.size).sum();

    if dry_run {
        println!(
            "Would remove {} {} ({}):",
            partials.len(),
            pluralize(partials.len(), "file", "files"),
            format_size(total_size)
        );
        for p in &partials {
            println!("  {}: {}  ({})", p.repo_id, p.filename, format_size(p.size));
        }
        return Ok(());
    }

    println!(
        "Found {} partial {}:",
        partials.len(),
        pluralize(partials.len(), "download", "downloads")
    );
    for p in &partials {
        println!("  {}: {}  ({})", p.repo_id, p.filename, format_size(p.size));
    }

    if !yes {
        let prompt = format!(
            "Clean {} {} ({})? [y/N]",
            partials.len(),
            pluralize(partials.len(), "file", "files"),
            format_size(total_size)
        );
        // BORROW: explicit .as_str() instead of Deref coercion
        if !confirm_prompt(prompt.as_str()) {
            println!("Aborted.");
            return Ok(());
        }
    }

    for p in &partials {
        std::fs::remove_file(&p.path).map_err(|e| FetchError::Io {
            // BORROW: explicit .clone() for owned PathBuf
            path: p.path.clone(),
            source: e,
        })?;
        // Sidecars (`.chunked.part.state`, `.chunked.part.state.tmp`) are
        // best-effort companions: they may not exist for every partial
        // (older interrupted downloads pre-date the sidecar) and they're
        // kilobyte-sized, so a removal failure here is not worth aborting
        // the whole sweep. Errors are silently ignored.
        for sidecar in p.sidecar_paths() {
            let _ = std::fs::remove_file(&sidecar);
        }
    }

    println!(
        "Removed {} {}. Freed {}.",
        partials.len(),
        pluralize(partials.len(), "file", "files"),
        format_size(total_size)
    );
    Ok(())
}

/// Deletes a cached model by removing its `models--org--name/` directory.
///
/// Shows a size preview and prompts for confirmation unless `--yes` is passed.
/// Removes the `models--org--name/` directory for `repo_id`.
///
/// Low-level helper shared by `run_cache_delete` and `run_cache_gc`. Caller
/// owns any preview/prompt rendering. The function refuses to follow
/// symlinks (HF cache repos are always real directories) so a malicious or
/// accidental symlink can't redirect deletion outside the cache.
///
/// # Errors
///
/// Returns [`FetchError::InvalidArgument`] when `repo_id` is not cached or
/// when its `models--…` entry is a symlink. Returns [`FetchError::Io`] on
/// filesystem failure during removal.
fn delete_repo_dir(repo_id: &str) -> Result<(), FetchError> {
    let cache_dir = cache::hf_cache_dir()?;
    let repo_dir = hf_fetch_model::cache_layout::repo_dir(&cache_dir, repo_id);

    if !repo_dir.exists() {
        return Err(FetchError::InvalidArgument(format!(
            "{repo_id} is not cached"
        )));
    }

    // Defensive: refuse to follow symlinks. `remove_dir_all` on Unix follows
    // symlinks and would delete the target, not the link — dangerous if the
    // cache dir was tampered with.
    let meta = std::fs::symlink_metadata(&repo_dir).map_err(|e| FetchError::Io {
        // BORROW: explicit .clone() for owned PathBuf
        path: repo_dir.clone(),
        source: e,
    })?;
    if meta.file_type().is_symlink() {
        return Err(FetchError::InvalidArgument(format!(
            "{repo_id} cache entry is a symlink; refusing to delete"
        )));
    }

    std::fs::remove_dir_all(&repo_dir).map_err(|e| FetchError::Io {
        // BORROW: explicit .clone() for owned PathBuf
        path: repo_dir.clone(),
        source: e,
    })?;
    Ok(())
}

fn run_cache_delete(repo_id: &str, yes: bool) -> Result<(), FetchError> {
    let cache_dir = cache::hf_cache_dir()?;

    if !cache_dir.exists() {
        println!("No HuggingFace cache found at {}", cache_dir.display());
        return Ok(());
    }

    let repo_dir = hf_fetch_model::cache_layout::repo_dir(&cache_dir, repo_id);

    if !repo_dir.exists() {
        return Err(FetchError::InvalidArgument(format!(
            "{repo_id} is not cached"
        )));
    }

    // Get size and file count for the preview (targeted scan, not full cache).
    let (file_count, size) = cache::repo_disk_usage(repo_id)?;

    println!(
        "  {repo_id}  ({}, {} {})",
        format_size(size),
        file_count,
        pluralize(file_count, "file", "files")
    );

    if !yes && !confirm_prompt("  Delete? [y/N]") {
        println!("  Aborted.");
        return Ok(());
    }

    delete_repo_dir(repo_id)?;
    println!("  Deleted. Freed {}.", format_size(size));
    Ok(())
}

/// Prints the snapshot directory path for a cached model.
///
/// Resolves the named ref (default: `main`) to a commit hash and constructs
/// the snapshot path. Output is a bare path with no decoration, intended for
/// shell substitution: `cd $(hf-fm cache path org/model)`.
fn run_cache_path(repo_id: &str, revision: Option<&str>) -> Result<(), FetchError> {
    let revision = revision.unwrap_or("main");
    let cache_dir = cache::hf_cache_dir()?;
    let repo_dir = hf_fetch_model::cache_layout::repo_dir(&cache_dir, repo_id);

    if !repo_dir.exists() {
        return Err(FetchError::InvalidArgument(format!(
            "{repo_id} is not cached"
        )));
    }

    let commit_hash = cache::read_ref(&repo_dir, revision).ok_or_else(|| {
        FetchError::InvalidArgument(format!(
            "{repo_id} is cached but has no ref for \"{revision}\""
        ))
    })?;

    // BORROW: explicit .as_str() instead of Deref coercion
    let snapshot_dir = hf_fetch_model::cache_layout::snapshot_dir(&repo_dir, commit_hash.as_str());

    if !snapshot_dir.exists() {
        return Err(FetchError::InvalidArgument(format!(
            "snapshot directory for {repo_id} at revision \"{revision}\" does not exist"
        )));
    }

    // Print bare path (no labels) for shell substitution.
    println!("{}", snapshot_dir.display());
    Ok(())
}

/// Renders one streamed result line for [`run_cache_verify`].
///
/// Returns the formatted line(s) for a [`cache::VerifyStatus`], padded to
/// the global filename column width `fw` so the size column lines up.
/// Mismatches expand to three lines (status + expected + actual digests).
fn format_verify_line(
    filename: &str,
    size: u64,
    status: &cache::VerifyStatus,
    fw: usize,
) -> String {
    match status {
        cache::VerifyStatus::Ok => format!(
            "  \u{2713} {filename:<fw$} {:>10}  SHA256 OK",
            format_size(size)
        ),
        cache::VerifyStatus::Mismatch { expected, actual } => format!(
            "  \u{2717} {filename:<fw$} {:>10}  SHA256 MISMATCH\n      expected {expected}\n      actual   {actual}",
            format_size(size)
        ),
        cache::VerifyStatus::Skipped => format!(
            "  \u{2014} {filename:<fw$} {:>10}  no LFS hash",
            format_size(size)
        ),
        cache::VerifyStatus::Missing => format!(
            "  ! {filename:<fw$} {:>10}  MISSING",
            format_size(size)
        ),
        // EXPLICIT: future VerifyStatus variants display as UNKNOWN
        _ => format!(
            "  ? {filename:<fw$} {:>10}  UNKNOWN",
            format_size(size)
        ),
    }
}

/// Re-verifies SHA256 digests of cached files against `HuggingFace` LFS metadata.
///
/// Drives [`cache::verify_cache_with_progress`] with an `indicatif` spinner
/// per file so the user sees liveness even when a single SHA256 takes 30+
/// seconds on a multi-GiB safetensors. Result lines are streamed (printed
/// above the spinner via `bar.println`) as each file completes — the user
/// gets progressive output rather than a wall of text at the end.
///
/// Mismatches print expected/actual digests inline; the function returns
/// [`FetchError::InvalidArgument`] so the process exits non-zero.
// EXPLICIT: linear pipeline of token resolution, runtime construction,
// header rendering, streamed verification with spinner, and footer counts.
// Splitting would obscure the streamed-output flow.
#[allow(clippy::too_many_lines)]
fn run_cache_verify(
    repo_id: &str,
    revision: Option<&str>,
    token: Option<&str>,
) -> Result<(), FetchError> {
    use indicatif::{ProgressBar, ProgressStyle};
    use std::cell::Cell;

    let cache_dir = cache::hf_cache_dir()?;
    let repo_dir = hf_fetch_model::cache_layout::repo_dir(&cache_dir, repo_id);

    if !repo_dir.exists() {
        return Err(FetchError::InvalidArgument(format!(
            "{repo_id} is not cached"
        )));
    }

    // Resolve token from arg or HF_TOKEN env (mirrors run_status / run_list_files).
    // BORROW: explicit String::from for Option<&str> → Option<String>
    let resolved_token = token
        .map(String::from)
        .or_else(|| std::env::var("HF_TOKEN").ok());

    let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
        path: PathBuf::from("<runtime>"),
        source: e,
    })?;

    // Header — printed BEFORE the spinner so the user sees what's being verified.
    let rev_display = revision.unwrap_or("main");
    let commit_hash = cache::read_ref(&repo_dir, rev_display);
    match commit_hash.as_deref() {
        Some(hash) => println!("{repo_id} ({rev_display} @ {hash})"),
        None => println!("{repo_id} ({rev_display}, ref not resolved)"),
    }
    println!("Cache: {}\n", cache_dir.display());

    // One spinner for the whole run. `set_message` updates per file;
    // `println` emits result lines above the spinner without disturbing
    // its position. `tick_chars("|/-\\ ")` gives the classic ASCII
    // rotating-bar animation the user asked for.
    let bar = ProgressBar::new_spinner();
    let style = ProgressStyle::with_template("  {spinner} {msg}")
        .unwrap_or_else(|_| ProgressStyle::default_spinner())
        .tick_chars("|/-\\ ");
    bar.set_style(style);
    bar.enable_steady_tick(Duration::from_millis(100));

    // Filename column width is computed from `Started` and reused for every
    // streamed line so they stay aligned. `Cell` keeps the closure `Fn`.
    let fw_cell: Cell<usize> = Cell::new(4);

    let bar_ref = &bar;
    let on_event = |event: cache::VerifyEvent<'_>| match event {
        cache::VerifyEvent::Started {
            total: _,
            max_filename_len,
        } => {
            // Minimum 4 ("File".len()), grow to fit the longest name.
            fw_cell.set(max_filename_len.max(4));
        }
        cache::VerifyEvent::FileStart {
            index,
            total,
            filename,
            size,
            has_lfs,
        } => {
            // "Verifying" for LFS files (real SHA256 work); "Reading" for
            // git-stored files (instant skip — the spinner barely shows).
            let label = if has_lfs { "Verifying" } else { "Reading" };
            bar_ref.set_message(format!(
                "[{index}/{total}] {label} {filename} ({})",
                format_size(size)
            ));
        }
        cache::VerifyEvent::FileComplete {
            index: _,
            total: _,
            filename,
            size,
            status,
        } => {
            // `bar.suspend` clears the spinner, runs the closure, then
            // redraws — works in TTY (no visual interference with the
            // spinner) and in non-TTY (the closure just prints normally
            // because the bar's draw target is hidden anyway). This is
            // robust where `bar.println` would silently drop the line in
            // non-TTY contexts (CI, redirected stderr).
            let line = format_verify_line(filename, size, status, fw_cell.get());
            bar_ref.suspend(|| println!("{line}"));
        }
        // EXPLICIT: future VerifyEvent variants are silently ignored —
        // the spinner keeps ticking, the streamed table keeps printing.
        _ => {}
    };

    // BORROW: explicit .as_deref() for Option<String> → Option<&str>
    let results = rt.block_on(cache::verify_cache_with_progress(
        repo_id,
        resolved_token.as_deref(),
        revision,
        on_event,
    ))?;

    bar.finish_and_clear();

    if results.is_empty() {
        println!("  (no files found in remote repository)");
        return Ok(());
    }

    // Footer counts, derived from the final results so we don't carry
    // mutable state through the closure.
    let mut ok_count: usize = 0;
    let mut mismatch_count: usize = 0;
    let mut skipped_count: usize = 0;
    let mut missing_count: usize = 0;
    for r in &results {
        match &r.status {
            cache::VerifyStatus::Ok => ok_count += 1,
            cache::VerifyStatus::Mismatch { .. } => mismatch_count += 1,
            cache::VerifyStatus::Skipped => skipped_count += 1,
            cache::VerifyStatus::Missing => missing_count += 1,
            // EXPLICIT: future VerifyStatus variants do not contribute to
            // any counted bucket; the per-file `?` line already reflected them.
            _ => {}
        }
    }

    let total = results.len();
    println!();
    println!(
        "{total} {}: {ok_count} SHA256 OK, {mismatch_count} mismatch, \
         {skipped_count} skipped, {missing_count} missing",
        pluralize(total, "file", "files")
    );

    if mismatch_count > 0 {
        return Err(FetchError::InvalidArgument(format!(
            "{mismatch_count} {} failed SHA256 verification",
            pluralize(mismatch_count, "file", "files")
        )));
    }

    Ok(())
}

// ============================================================================
// cache gc: age- and budget-based eviction
// ============================================================================

/// A cached repo `has_partial == true` whose mtime falls within this window
/// is treated as actively downloading and skipped from gc to avoid racing
/// with `hf-fm download`. Stale partials (older than the window) remain
/// eligible — `cache clean-partial` is the right tool for those.
const FRESH_PARTIAL_WINDOW_SECS: u64 = 60 * 60;

/// Snapshot of a single cached repo for gc planning and preview rendering.
#[derive(Debug, Clone)]
struct EvictionEntry {
    /// Repository identifier (e.g., `"google/gemma-2-2b-it"`).
    repo_id: String,
    /// Total size on disk in bytes.
    size: u64,
    /// Most recent file mtime in the snapshot dir, if available.
    last_modified: Option<std::time::SystemTime>,
}

impl From<&cache::CachedModelSummary> for EvictionEntry {
    fn from(s: &cache::CachedModelSummary) -> Self {
        Self {
            // BORROW: explicit .clone() for owned String
            repo_id: s.repo_id.clone(),
            size: s.total_size,
            last_modified: s.last_modified,
        }
    }
}

/// User-supplied gc criteria, normalised from CLI flags.
struct GcCriteria {
    /// Maximum allowed age in seconds; repos older than this are eligible.
    /// `None` disables age eviction.
    older_than_secs: Option<u64>,
    /// Hard cap on total cache size in bytes after gc. `None` disables
    /// budget eviction.
    max_size: Option<u64>,
    /// Repository identifiers that must never be evicted (deduped).
    except: HashSet<String>,
}

/// Outcome of [`compute_gc_plan`] — what would be deleted, kept, skipped,
/// and the cache-size delta.
struct GcPlan {
    /// Repos selected for deletion, sorted by mtime ascending (oldest
    /// first), `repo_id` ascending as a deterministic tiebreaker.
    evict: Vec<EvictionEntry>,
    /// Repos protected by `--except` (only populated when `--except` was passed).
    protected: Vec<EvictionEntry>,
    /// Eligible repos kept by gc (only populated when `list_kept == true`).
    kept: Vec<EvictionEntry>,
    /// Repos skipped because of an active partial download (see
    /// [`FRESH_PARTIAL_WINDOW_SECS`]).
    skipped_partials: Vec<EvictionEntry>,
    /// Total cache size before gc, in bytes.
    size_before: u64,
    /// Total cache size after gc, in bytes (computed).
    size_after: u64,
    /// `true` when `--max-size` was requested but the budget can't be met
    /// because protected repos exceed it.
    budget_shortfall: bool,
}

/// Returns the set of repo IDs from `summaries` that are older than
/// `threshold_secs` relative to `now`.
///
/// Repos with `last_modified == None` are not selected (we don't age-evict
/// repos whose age we can't determine). Repos with a future-dated mtime
/// (clock skew, NFS) are also not selected — `now.duration_since(mtime)`
/// returns `Err` and we treat them as age 0.
#[must_use]
fn select_age_evictions(
    summaries: &[cache::CachedModelSummary],
    threshold_secs: u64,
    now: std::time::SystemTime,
) -> HashSet<&str> {
    // BORROW: explicit .as_str() to expose owned `repo_id: String` as the borrowed key.
    summaries
        .iter()
        .filter_map(|s| {
            let mtime = s.last_modified?;
            let elapsed = now.duration_since(mtime).ok()?;
            (elapsed.as_secs() >= threshold_secs).then_some(s.repo_id.as_str())
        })
        .collect()
}

/// Builds a [`GcPlan`] from cache state and user criteria.
///
/// Algorithm:
/// 1. Partition `summaries` into `protected` (in `criteria.except`),
///    `skipped_partials` (active in-flight downloads), and `eligible`.
/// 2. Apply age eviction over `eligible` → preliminary evict set.
/// 3. If `criteria.max_size` is set and the post-step-2 cache is still
///    over budget, sort the remaining eligible by `(last_modified,
///    repo_id)` ascending and greedily add until under budget.
/// 4. `budget_shortfall` is set when step 3 ran but the final size still
///    exceeds the budget (protected repos pinned more than `max_size`).
///
/// `last_modified == None` sorts before any `Some(_)` (so unknown-age
/// repos are evicted first by the budget step), and `repo_id` ascending
/// is a deterministic tiebreaker so the preview is stable across runs.
#[must_use]
fn compute_gc_plan(
    summaries: &[cache::CachedModelSummary],
    criteria: &GcCriteria,
    now: std::time::SystemTime,
    list_kept: bool,
) -> GcPlan {
    let size_before: u64 = summaries
        .iter()
        .map(|s| s.total_size)
        .fold(0_u64, u64::saturating_add);

    // BORROW: this function uses .as_str() throughout for HashSet lookups
    // and tuple sort keys — all owned `repo_id: String` → `&str` borrows
    // valid for the duration of the call.

    // Step 1: partition.
    let mut protected: Vec<EvictionEntry> = Vec::new();
    let mut skipped_partials: Vec<EvictionEntry> = Vec::new();
    let mut eligible: Vec<&cache::CachedModelSummary> = Vec::new();
    for s in summaries {
        if criteria.except.contains(s.repo_id.as_str()) {
            protected.push(EvictionEntry::from(s));
            continue;
        }
        let is_fresh_partial = s.has_partial
            && s.last_modified.is_some_and(|m| {
                now.duration_since(m)
                    .is_ok_and(|d| d.as_secs() < FRESH_PARTIAL_WINDOW_SECS)
            });
        if is_fresh_partial {
            skipped_partials.push(EvictionEntry::from(s));
        } else {
            eligible.push(s);
        }
    }

    // Step 2: age eviction. Compute the global age set across all summaries,
    // then partition `eligible` against it — protected and skipped_partials
    // repos are not in `eligible`, so they're already excluded.
    let age_set: HashSet<&str> = match criteria.older_than_secs {
        Some(threshold) => select_age_evictions(summaries, threshold, now),
        None => HashSet::new(),
    };
    let (mut to_evict, mut still_eligible): (
        Vec<&cache::CachedModelSummary>,
        Vec<&cache::CachedModelSummary>,
    ) = eligible
        .into_iter()
        .partition(|s| age_set.contains(s.repo_id.as_str()));

    // Step 3: budget eviction.
    let mut budget_shortfall = false;
    if let Some(max_size) = criteria.max_size {
        let already_evicting: u64 = to_evict
            .iter()
            .map(|s| s.total_size)
            .fold(0_u64, u64::saturating_add);
        let mut remaining_after = size_before.saturating_sub(already_evicting);
        if remaining_after > max_size {
            // Sort by (mtime, repo_id) ascending. `Option<SystemTime>`
            // orders `None` before any `Some(_)`, so unknown-age repos
            // are picked first — they're the riskiest to keep.
            still_eligible.sort_by(|a, b| {
                (a.last_modified, a.repo_id.as_str()).cmp(&(b.last_modified, b.repo_id.as_str()))
            });
            let mut split_idx = 0_usize;
            for s in &still_eligible {
                if remaining_after <= max_size {
                    break;
                }
                to_evict.push(s);
                remaining_after = remaining_after.saturating_sub(s.total_size);
                split_idx = split_idx.saturating_add(1);
            }
            budget_shortfall = remaining_after > max_size;
            still_eligible.drain(0..split_idx);
        }
    }

    // Final: sort the eviction list deterministically, materialise entries.
    to_evict.sort_by(|a, b| {
        (a.last_modified, a.repo_id.as_str()).cmp(&(b.last_modified, b.repo_id.as_str()))
    });
    let evict: Vec<EvictionEntry> = to_evict.iter().map(|s| EvictionEntry::from(*s)).collect();
    let evicted_size: u64 = evict
        .iter()
        .map(|e| e.size)
        .fold(0_u64, u64::saturating_add);
    let size_after = size_before.saturating_sub(evicted_size);

    let kept = if list_kept {
        still_eligible
            .iter()
            .map(|s| EvictionEntry::from(*s))
            .collect()
    } else {
        Vec::new()
    };

    GcPlan {
        evict,
        protected,
        kept,
        skipped_partials,
        size_before,
        size_after,
        budget_shortfall,
    }
}

/// Renders an [`EvictionEntry`] table (repo / size / age) to stdout.
///
/// Used by the gc preview for the `Will remove` and (with `--list-kept`)
/// `Keep` sections so they share the same alignment.
fn render_eviction_table(entries: &[EvictionEntry]) {
    let name_width = entries.iter().map(|e| e.repo_id.len()).max().unwrap_or(0);
    let size_width = entries
        .iter()
        .map(|e| format_size(e.size).len())
        .max()
        .unwrap_or(0);
    for entry in entries {
        let age_str = entry
            .last_modified
            .map_or_else(|| "\u{2014}".to_owned(), format_age);
        println!(
            "  {:<nw$}  {:>sw$}  {age_str}",
            entry.repo_id,
            format_size(entry.size),
            nw = name_width,
            sw = size_width,
        );
    }
}

/// Renders the human-readable gc preview to stdout, in this order:
/// `Will remove`, `Skipped (active partial downloads)`, `Protected by --except`,
/// optional `Keep`, then a summary line `Cache: X → Y (free Z)`.
fn render_gc_plan_preview(plan: &GcPlan) {
    if !plan.evict.is_empty() {
        println!("Will remove:");
        render_eviction_table(&plan.evict);
        println!();
    }

    if !plan.skipped_partials.is_empty() {
        println!("Skipped (active partial downloads):");
        for entry in &plan.skipped_partials {
            println!("  {}", entry.repo_id);
        }
        println!();
    }

    if !plan.protected.is_empty() {
        println!("Protected by --except:");
        for entry in &plan.protected {
            println!("  {}", entry.repo_id);
        }
        println!();
    }

    if !plan.kept.is_empty() {
        println!("Keep:");
        render_eviction_table(&plan.kept);
        println!();
    }

    let freed = plan.size_before.saturating_sub(plan.size_after);
    println!(
        "Cache: {} \u{2192} {} (free {})",
        format_size(plan.size_before),
        format_size(plan.size_after),
        format_size(freed)
    );

    if plan.budget_shortfall {
        eprintln!("warning: --max-size budget cannot be reached; protected repos exceed the cap");
    }
}

/// Garbage-collects cached models per the supplied criteria.
///
/// Wires the CLI flags through `compute_gc_plan`, renders a preview, and
/// (unless `dry_run` is set) prompts for confirmation before deleting via
/// [`delete_repo_dir`]. Deletion failures are collected per-repo so a
/// single permission error doesn't abort the whole run; the function
/// exits with an error if any deletion failed.
///
/// `older_than_days` is converted to seconds (1 day = `86_400` s);
/// `max_size` is already in bytes (parsed by [`parse_size_arg`]).
///
/// # Errors
///
/// Returns [`FetchError::Io`] when reading the cache directory fails, or
/// [`FetchError::InvalidArgument`] when one or more repo deletions fail.
fn run_cache_gc(
    older_than_days: Option<u64>,
    max_size: Option<u64>,
    except: Vec<String>,
    dry_run: bool,
    yes: bool,
    list_kept: bool,
) -> Result<(), FetchError> {
    let cache_dir = cache::hf_cache_dir()?;
    if !cache_dir.exists() {
        println!("No HuggingFace cache found at {}", cache_dir.display());
        return Ok(());
    }

    println!("Cache: {}\n", cache_dir.display());

    let summaries = cache::cache_summary()?;
    if summaries.is_empty() {
        println!("No models in cache.");
        return Ok(());
    }

    // BORROW: explicit .as_str() to expose owned `repo_id: String` and the
    // `--except` `String` argument as borrowed keys for the lookup set.
    let known_ids: HashSet<&str> = summaries.iter().map(|s| s.repo_id.as_str()).collect();
    // Validate --except against known repos; warn (don't error) on misses.
    for repo in &except {
        if !known_ids.contains(repo.as_str()) {
            eprintln!("warning: --except {repo:?} is not a cached repo; ignoring");
        }
    }

    let criteria = GcCriteria {
        older_than_secs: older_than_days.map(|d| d.saturating_mul(86_400)),
        max_size,
        except: except.into_iter().collect(),
    };

    let plan = compute_gc_plan(
        &summaries,
        &criteria,
        std::time::SystemTime::now(),
        list_kept,
    );

    if !plan.skipped_partials.is_empty() {
        let n = plan.skipped_partials.len();
        eprintln!(
            "note: {n} {} skipped (active partial {}); run `hf-fm cache clean-partial` first",
            pluralize(n, "repo", "repos"),
            pluralize(n, "download", "downloads")
        );
    }

    if plan.evict.is_empty() {
        println!("No repos matched eviction criteria.");
        return Ok(());
    }

    render_gc_plan_preview(&plan);

    if dry_run {
        return Ok(());
    }

    if !yes && !confirm_prompt("\nProceed? [y/N]") {
        println!("Aborted.");
        return Ok(());
    }

    let mut failures: Vec<(String, FetchError)> = Vec::new();
    let mut freed: u64 = 0;
    for entry in &plan.evict {
        // BORROW: explicit .as_str() for owned String → &str argument.
        match delete_repo_dir(entry.repo_id.as_str()) {
            Ok(()) => freed = freed.saturating_add(entry.size),
            Err(e) => {
                eprintln!("error: failed to delete {}: {e}", entry.repo_id);
                // BORROW: explicit .clone() for owned String
                failures.push((entry.repo_id.clone(), e));
            }
        }
    }

    let success_count = plan.evict.len().saturating_sub(failures.len());
    println!(
        "Removed {success_count} {}. Freed {}.",
        pluralize(success_count, "repo", "repos"),
        format_size(freed)
    );

    if !failures.is_empty() {
        let n = failures.len();
        return Err(FetchError::InvalidArgument(format!(
            "{n} {} failed to delete during gc",
            pluralize(n, "repo", "repos")
        )));
    }

    Ok(())
}

/// Prompts the user for confirmation via stdin.
///
/// Returns `true` if the user enters `y` or `Y`, `false` otherwise.
fn confirm_prompt(message: &str) -> bool {
    eprint!("{message} ");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).is_ok() && input.trim().eq_ignore_ascii_case("y")
}

/// Collects all tensors from a repo into a name-keyed map.
///
/// Inspects all `.safetensors` files (cached or remote) and flattens
/// tensors across shards into a single `HashMap`.
fn collect_repo_tensors(
    repo_id: &str,
    revision: Option<&str>,
    token: Option<&str>,
    cached: bool,
) -> Result<HashMap<String, inspect::TensorInfo>, FetchError> {
    let results: Vec<(String, inspect::SafetensorsHeaderInfo)> = if cached {
        inspect::inspect_repo_safetensors_cached(repo_id, revision)?
    } else {
        // BORROW: explicit String::from for Option<&str> → Option<String>
        let token = token
            .map(String::from)
            .or_else(|| std::env::var("HF_TOKEN").ok());

        let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
            path: PathBuf::from("<runtime>"),
            source: e,
        })?;
        // BORROW: explicit .as_deref() for Option<String> → Option<&str>
        let remote_results = rt.block_on(inspect::inspect_repo_safetensors(
            repo_id,
            token.as_deref(),
            revision,
        ))?;
        remote_results
            .into_iter()
            .map(|(name, info, _source)| (name, info))
            .collect()
    };

    let mut tensors = HashMap::new();
    for (_filename, info) in results {
        for t in info.tensors {
            // BORROW: explicit .clone() for owned String key
            tensors.insert(t.name.clone(), t);
        }
    }

    Ok(tensors)
}

/// Compares tensor layouts between two model repositories.
// EXPLICIT: orchestrates two-side metadata fetch, tensor classification (only-A,
// only-B, dtype/shape diffs, matching), and output (table or JSON).
// Sequential pipeline; splitting hides the comparison flow.
#[allow(
    clippy::too_many_arguments,
    clippy::fn_params_excessive_bools,
    clippy::too_many_lines
)]
fn run_diff(
    repo_a: &str,
    repo_b: &str,
    revision_a: Option<&str>,
    revision_b: Option<&str>,
    token: Option<&str>,
    cached: bool,
    filter: Option<&str>,
    summary: bool,
    dtypes: bool,
    limit: Option<usize>,
    json: bool,
) -> Result<(), FetchError> {
    let tensors_a = collect_repo_tensors(repo_a, revision_a, token, cached)
        .map_err(|e| enrich_gated_content_error(e, repo_a, token))?;
    let tensors_b = collect_repo_tensors(repo_b, revision_b, token, cached)
        .map_err(|e| enrich_gated_content_error(e, repo_b, token))?;

    if tensors_a.is_empty() {
        println!("No .safetensors files found in {repo_a}.");
        println!("Hint: use `hf-fm list-files {repo_a}` to see available file types");
        return Ok(());
    }
    if tensors_b.is_empty() {
        println!("No .safetensors files found in {repo_b}.");
        println!("Hint: use `hf-fm list-files {repo_b}` to see available file types");
        return Ok(());
    }

    // Collect all tensor names from both repos (BTreeSet deduplicates and sorts).
    let mut all_names: Vec<&str> = tensors_a
        .keys()
        .chain(tensors_b.keys())
        // BORROW: explicit .as_str() instead of Deref coercion
        .map(String::as_str)
        .collect::<BTreeSet<&str>>()
        .into_iter()
        .collect();

    // Apply filter.
    if let Some(pattern) = filter {
        all_names.retain(|name| matches_filter(name, pattern));
    }

    // Classify into four buckets.
    let mut only_a: Vec<&str> = Vec::new();
    let mut only_b: Vec<&str> = Vec::new();
    let mut differ: Vec<&str> = Vec::new();
    let mut matching: Vec<&str> = Vec::new();

    for name in &all_names {
        match (tensors_a.get(*name), tensors_b.get(*name)) {
            (Some(_), None) => only_a.push(name),
            (None, Some(_)) => only_b.push(name),
            (Some(a), Some(b)) => {
                if a.dtype == b.dtype && a.shape == b.shape {
                    matching.push(name);
                } else {
                    differ.push(name);
                }
            }
            (None, None) => {} // EXPLICIT: impossible — name comes from one of the two maps
        }
    }

    // Compute totals for the summary.
    let total_a = if filter.is_some() {
        only_a.len() + differ.len() + matching.len()
    } else {
        tensors_a.len()
    };
    let total_b = if filter.is_some() {
        only_b.len() + differ.len() + matching.len()
    } else {
        tensors_b.len()
    };

    // JSON output mode.
    if json {
        return print_diff_json(
            repo_a, repo_b, &tensors_a, &tensors_b, &only_a, &only_b, &differ, &matching, filter,
            dtypes, limit,
        );
    }

    // --dtypes mode replaces the per-tensor body + standard summary with a
    // side-by-side per-dtype histogram (header, table, own footer).
    if dtypes {
        print_diff_dtypes(repo_a, repo_b, &tensors_a, &tensors_b, filter);
        return Ok(());
    }

    // Print header.
    println!("  A: {repo_a}");
    println!("  B: {repo_b}");

    if !summary {
        // Compute name width across only-A and only-B sections for consistent columns.
        let nw = only_a
            .iter()
            .chain(only_b.iter())
            .map(|n| n.len())
            .max()
            .unwrap_or(0);

        // Per-section cap from `--limit` (applied after `--filter`); the full
        // vecs are left intact so the summary line keeps true counts.
        let cap = limit.unwrap_or(usize::MAX);
        // Prints a truncation note when `--limit` cut a section short.
        let print_limit_note = |total: usize| {
            if let Some(n) = limit {
                if total > n {
                    println!("    \u{2026} showing {n} of {total} (limit {n})");
                }
            }
        };

        println!();

        // Print only-in-A.
        if !only_a.is_empty() {
            let label = if only_a.len() == 1 {
                "tensor"
            } else {
                "tensors"
            };
            println!("  Only in A ({} {label}):", only_a.len());
            for name in only_a.iter().take(cap) {
                if let Some(t) = tensors_a.get(*name) {
                    let shape_str = format!("{:?}", t.shape);
                    println!("    {name:<nw$} {:<8} {shape_str}", t.dtype);
                }
            }
            print_limit_note(only_a.len());
            println!();
        }

        // Print only-in-B.
        if !only_b.is_empty() {
            let label = if only_b.len() == 1 {
                "tensor"
            } else {
                "tensors"
            };
            println!("  Only in B ({} {label}):", only_b.len());
            for name in only_b.iter().take(cap) {
                if let Some(t) = tensors_b.get(*name) {
                    let shape_str = format!("{:?}", t.shape);
                    println!("    {name:<nw$} {:<8} {shape_str}", t.dtype);
                }
            }
            print_limit_note(only_b.len());
            println!();
        }

        // Print dtype/shape differences.
        if !differ.is_empty() {
            let label = if differ.len() == 1 {
                "tensor"
            } else {
                "tensors"
            };
            println!("  Dtype/shape differences ({} {label}):", differ.len());
            for name in differ.iter().take(cap) {
                if let Some((a, b)) = tensors_a.get(*name).zip(tensors_b.get(*name)) {
                    let shape_a = format!("{:?}", a.shape);
                    let shape_b = format!("{:?}", b.shape);
                    println!("    {name}");
                    println!("      A: {:<8} {shape_a}", a.dtype);
                    println!("      B: {:<8} {shape_b}", b.dtype);
                }
            }
            print_limit_note(differ.len());
            println!();
        }

        // Print matching count.
        let match_label = if matching.len() == 1 {
            "tensor"
        } else {
            "tensors"
        };
        println!("  Matching: {} {match_label} identical", matching.len());
    }

    // Summary line.
    println!("  {}", "\u{2500}".repeat(70));
    print!(
        "  A: {} {} | B: {} {} | only-A: {} | only-B: {} | differ: {} | match: {}",
        total_a,
        pluralize(total_a, "tensor", "tensors"),
        total_b,
        pluralize(total_b, "tensor", "tensors"),
        only_a.len(),
        only_b.len(),
        differ.len(),
        matching.len(),
    );
    if let Some(pattern) = filter {
        println!(" (filter: {pattern:?})");
    } else {
        println!();
    }

    Ok(())
}

/// Serializable diff entry for `--json` output.
#[derive(serde::Serialize)]
struct DiffTensorEntry {
    /// Tensor name.
    name: String,
    /// Tensor info from model A, if present.
    #[serde(skip_serializing_if = "Option::is_none")]
    a: Option<DiffTensorSide>,
    /// Tensor info from model B, if present.
    #[serde(skip_serializing_if = "Option::is_none")]
    b: Option<DiffTensorSide>,
}

/// One side of a diff entry (dtype + shape + byte count).
#[derive(serde::Serialize)]
struct DiffTensorSide {
    /// Element dtype string.
    dtype: String,
    /// Tensor shape.
    shape: Vec<usize>,
    /// On-disk byte count from the safetensors header; powers downstream `jq` recipes.
    byte_count: u64,
}

/// One per-dtype row in a `--dtypes` histogram (owned, ready for serialization).
///
/// Owned mirror of [`DtypeGroup`] that doesn't borrow its dtype — required
/// because [`DiffResult`] is constructed as an owned value before being
/// serialized, and threading a lifetime through it for one borrowed field
/// would propagate complexity for no gain.
#[derive(serde::Serialize)]
struct DiffDtypeGroup {
    /// Element dtype string.
    dtype: String,
    /// Number of tensors of this dtype on this side.
    tensors: usize,
    /// Sum of element counts (`num_elements()`) across tensors of this dtype.
    params: u64,
    /// Sum of on-disk byte counts across tensors of this dtype.
    bytes: u64,
}

/// Per-side dtype histograms emitted under `--dtypes --json`.
#[derive(serde::Serialize)]
struct DtypeHistograms {
    /// Histogram for repo A, sorted by `bytes` descending.
    a: Vec<DiffDtypeGroup>,
    /// Histogram for repo B, sorted by `bytes` descending.
    b: Vec<DiffDtypeGroup>,
}

/// Per-section truncation metadata for `diff --json`, emitted only when
/// `--limit` capped at least one section.
///
/// All three sections are always present for a stable schema; an untruncated
/// section reports `shown == total`. Reuses [`TruncationInfo`] per section so
/// the `{shown, total}` shape matches `inspect --json`.
#[derive(serde::Serialize)]
struct DiffTruncation {
    /// Truncation of the only-in-A section.
    only_a: TruncationInfo,
    /// Truncation of the only-in-B section.
    only_b: TruncationInfo,
    /// Truncation of the dtype/shape-differences section.
    differ: TruncationInfo,
}

/// Serializable diff result for `--json` output.
#[derive(serde::Serialize)]
struct DiffResult {
    /// Model A repository identifier.
    repo_a: String,
    /// Model B repository identifier.
    repo_b: String,
    /// Tensors only in model A.
    only_a: Vec<DiffTensorEntry>,
    /// Tensors only in model B.
    only_b: Vec<DiffTensorEntry>,
    /// Tensors with different dtypes or shapes.
    differ: Vec<DiffTensorEntry>,
    /// Number of tensors that match exactly.
    matching_count: usize,
    /// Filter pattern applied, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    filter: Option<String>,
    /// Per-side dtype histograms when `--dtypes` is set; absent otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    dtype_histograms: Option<DtypeHistograms>,
    /// Per-section truncation when `--limit` capped output; absent when complete.
    #[serde(skip_serializing_if = "Option::is_none")]
    truncated: Option<DiffTruncation>,
}

/// Prints diff results as JSON.
#[allow(clippy::too_many_arguments)]
fn print_diff_json(
    repo_a: &str,
    repo_b: &str,
    tensors_a: &HashMap<String, inspect::TensorInfo>,
    tensors_b: &HashMap<String, inspect::TensorInfo>,
    only_a: &[&str],
    only_b: &[&str],
    differ: &[&str],
    matching: &[&str],
    filter: Option<&str>,
    dtypes: bool,
    limit: Option<usize>,
) -> Result<(), FetchError> {
    let make_entry = |name: &str,
                      a: Option<&inspect::TensorInfo>,
                      b: Option<&inspect::TensorInfo>|
     -> DiffTensorEntry {
        DiffTensorEntry {
            name: name.to_owned(),
            a: a.map(|t| DiffTensorSide {
                // BORROW: explicit .clone() for owned String
                dtype: t.dtype.clone(),
                shape: t.shape.clone(),
                byte_count: t.byte_len(),
            }),
            b: b.map(|t| DiffTensorSide {
                // BORROW: explicit .clone() for owned String
                dtype: t.dtype.clone(),
                shape: t.shape.clone(),
                byte_count: t.byte_len(),
            }),
        }
    };

    // EXPLICIT: --dtypes populates dtype_histograms; otherwise the field is omitted via skip_serializing_if.
    let dtype_histograms = if dtypes {
        let (rows_a, rows_b) = aggregate_diff_dtypes(tensors_a, tensors_b, filter);
        Some(DtypeHistograms {
            a: rows_a,
            b: rows_b,
        })
    } else {
        None
    };

    // Per-section cap (after `--filter`); the full slices stay intact so
    // `truncated` can report each section's true total.
    let cap = limit.unwrap_or(usize::MAX);
    let truncated = limit
        .is_some_and(|n| only_a.len() > n || only_b.len() > n || differ.len() > n)
        .then(|| DiffTruncation {
            only_a: TruncationInfo {
                shown: only_a.len().min(cap),
                total: only_a.len(),
            },
            only_b: TruncationInfo {
                shown: only_b.len().min(cap),
                total: only_b.len(),
            },
            differ: TruncationInfo {
                shown: differ.len().min(cap),
                total: differ.len(),
            },
        });

    let result = DiffResult {
        repo_a: repo_a.to_owned(),
        repo_b: repo_b.to_owned(),
        only_a: only_a
            .iter()
            .take(cap)
            .map(|n| make_entry(n, tensors_a.get(*n), None))
            .collect(),
        only_b: only_b
            .iter()
            .take(cap)
            .map(|n| make_entry(n, None, tensors_b.get(*n)))
            .collect(),
        differ: differ
            .iter()
            .take(cap)
            .map(|n| make_entry(n, tensors_a.get(*n), tensors_b.get(*n)))
            .collect(),
        matching_count: matching.len(),
        filter: filter.map(str::to_owned),
        dtype_histograms,
        truncated,
    };

    emit_json(&result)
}

/// Design target for the `--dtypes` histogram width on a default terminal.
///
/// Informational only — the renderer uses dynamic widths and the table is
/// allowed to grow wider when dtype names or large byte values demand it.
/// Guarded by `diff_dtypes_canonical_case_fits_design_width` so a future
/// column-format change that blows past the budget on the canonical scaled-
/// sibling case is caught at test time. Natural breakpoint for a future
/// `--compact` mode or terminal-width-aware rendering.
#[allow(dead_code)] // EXPLICIT: only the test target references this today; the const is also a documentation anchor for a future `--compact` mode.
const DIFF_DTYPES_DESIGN_WIDTH: usize = 75;

/// Column widths for the `diff --dtypes` histogram, computed from the data.
///
/// Returned by [`diff_dtypes_column_widths`]; consumed by [`print_diff_dtypes`]
/// for `println!` width specs, and by the budget test to compare
/// [`Self::total_width`] against [`DIFF_DTYPES_DESIGN_WIDTH`].
struct DiffDtypesColumnWidths {
    /// Width of the dtype-name column.
    dtype: usize,
    /// Width of the A-tensors count column.
    a_tensors: usize,
    /// Width of the A-size column (formatted byte string).
    a_size: usize,
    /// Width of the B-tensors count column.
    b_tensors: usize,
    /// Width of the B-size column (formatted byte string).
    b_size: usize,
    /// Width of the signed Δ-size column.
    delta: usize,
}

impl DiffDtypesColumnWidths {
    /// Total characters consumed by one full row: indent + columns + gaps.
    ///
    /// Layout has six 2-char gaps (one leading indent, five between the six
    /// columns) plus the six column widths themselves.
    fn total_width(&self) -> usize {
        2 + self.dtype
            + 2
            + self.a_tensors
            + 2
            + self.a_size
            + 2
            + self.b_tensors
            + 2
            + self.b_size
            + 2
            + self.delta
    }
}

/// Formats a signed byte delta with a leading `+` / `-` sign.
///
/// Zero renders as `format_size(0)` without a sign. Used for the `Δ Size`
/// column in the `diff --dtypes` histogram footer.
fn format_size_delta(delta: i64) -> String {
    match delta.cmp(&0) {
        std::cmp::Ordering::Less => format!("-{}", format_size(delta.unsigned_abs())),
        std::cmp::Ordering::Greater => format!("+{}", format_size(delta.unsigned_abs())),
        std::cmp::Ordering::Equal => format_size(0),
    }
}

/// Formats a signed tensor-count delta with a leading sign.
///
/// Zero renders as `"0"` without a sign. Used in the histogram footer.
fn format_count_delta(delta: i64) -> String {
    if delta == 0 {
        "0".to_owned()
    } else {
        format!("{delta:+}")
    }
}

/// Aggregates per-dtype histograms for both repos, optionally filtered by name.
///
/// Returns `(rows_a, rows_b)` where each `Vec<DiffDtypeGroup>` is sorted by
/// byte count descending. When `filter` is `Some(p)`, only tensors whose name
/// contains `p` contribute to either side's histogram. Pure helper — no I/O.
fn aggregate_diff_dtypes(
    tensors_a: &HashMap<String, inspect::TensorInfo>,
    tensors_b: &HashMap<String, inspect::TensorInfo>,
    filter: Option<&str>,
) -> (Vec<DiffDtypeGroup>, Vec<DiffDtypeGroup>) {
    let collect = |map: &HashMap<String, inspect::TensorInfo>| -> Vec<DiffDtypeGroup> {
        let filtered = map
            .iter()
            // `is_none_or` keeps the tensor when filter is None, or when its
            // name matches the filter pattern. Stable since Rust 1.82.
            .filter(|(name, _)| filter.is_none_or(|p| matches_filter(name, p)))
            .map(|(_, t)| t);
        let mut rows: Vec<DiffDtypeGroup> = compute_dtype_groups(filtered)
            .into_iter()
            .map(|(dtype, tensors, params, bytes)| DiffDtypeGroup {
                dtype: dtype.to_owned(),
                tensors,
                params,
                bytes,
            })
            .collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.bytes));
        rows
    };
    (collect(tensors_a), collect(tensors_b))
}

/// Computes dynamic column widths for the `diff --dtypes` histogram.
///
/// Each column floors at its header width (e.g. `"A Tensors".chars().count()`)
/// and grows to fit the widest data cell. Em-dash placeholders for missing
/// dtypes count as 1 char (single codepoint).
fn diff_dtypes_column_widths(
    rows_a: &[DiffDtypeGroup],
    rows_b: &[DiffDtypeGroup],
) -> DiffDtypesColumnWidths {
    let mut dtype_w = "Dtype".chars().count();
    let mut a_tensors_w = "A Tensors".chars().count();
    let mut a_size_w = "A Size".chars().count();
    let mut b_tensors_w = "B Tensors".chars().count();
    let mut b_size_w = "B Size".chars().count();
    let mut delta_w = "\u{0394} Size".chars().count(); // "Δ Size"

    // Union of dtype names present in either side.
    let mut all_dtypes: BTreeSet<&str> = BTreeSet::new();
    for r in rows_a {
        all_dtypes.insert(r.dtype.as_str());
    }
    for r in rows_b {
        all_dtypes.insert(r.dtype.as_str());
    }

    for dtype in &all_dtypes {
        dtype_w = dtype_w.max(dtype.chars().count());
        let a = rows_a.iter().find(|r| r.dtype == *dtype);
        let b = rows_b.iter().find(|r| r.dtype == *dtype);
        let a_tensors_str = a.map_or_else(|| "\u{2014}".to_owned(), |r| r.tensors.to_string());
        let b_tensors_str = b.map_or_else(|| "\u{2014}".to_owned(), |r| r.tensors.to_string());
        let a_size_str = a.map_or_else(|| "\u{2014}".to_owned(), |r| format_size(r.bytes));
        let b_size_str = b.map_or_else(|| "\u{2014}".to_owned(), |r| format_size(r.bytes));
        a_tensors_w = a_tensors_w.max(a_tensors_str.chars().count());
        b_tensors_w = b_tensors_w.max(b_tensors_str.chars().count());
        a_size_w = a_size_w.max(a_size_str.chars().count());
        b_size_w = b_size_w.max(b_size_str.chars().count());
        // CAST: u64 → i64 for delta computation; clipping to i64::MAX is safe — tensor
        // byte counts never approach 9.2 EiB in practice.
        let a_bytes_i = a.map_or(0_i64, |r| i64::try_from(r.bytes).unwrap_or(i64::MAX));
        let b_bytes_i = b.map_or(0_i64, |r| i64::try_from(r.bytes).unwrap_or(i64::MAX));
        let delta = b_bytes_i.saturating_sub(a_bytes_i);
        delta_w = delta_w.max(format_size_delta(delta).chars().count());
    }

    DiffDtypesColumnWidths {
        dtype: dtype_w,
        a_tensors: a_tensors_w,
        a_size: a_size_w,
        b_tensors: b_tensors_w,
        b_size: b_size_w,
        delta: delta_w,
    }
}

/// Renders the side-by-side per-dtype histogram for `diff --dtypes`.
///
/// Output shape (data-driven widths):
///
/// ```text
///   A: org/model-80B
///   B: org/model-35B
///
///   Dtype  A Tensors   A Size     B Tensors   B Size     Δ Size
///   BF16         630   6.72 GiB         126   1.34 GiB   -5.38 GiB
///   U8           192  18.91 GiB         192  14.30 GiB   -4.61 GiB
///   ───────────────────────────────────────────────────────────────
///   A: 822 tensors, 25.63 GiB | B: 318 tensors, 15.64 GiB | Δ: -504 tensors, -9.99 GiB
/// ```
// EXPLICIT: clippy::similar_names — diff-shaped function inherently has A-side / B-side
// mirrored bindings (rows_a / rows_b, total_a_tensors / total_b_tensors, ...). The
// pairing is the function's *purpose*; renaming would obscure intent.
#[allow(clippy::similar_names)]
fn print_diff_dtypes(
    repo_a: &str,
    repo_b: &str,
    tensors_a: &HashMap<String, inspect::TensorInfo>,
    tensors_b: &HashMap<String, inspect::TensorInfo>,
    filter: Option<&str>,
) {
    let (rows_a, rows_b) = aggregate_diff_dtypes(tensors_a, tensors_b, filter);
    let w = diff_dtypes_column_widths(&rows_a, &rows_b);

    let dw = w.dtype;
    let atw = w.a_tensors;
    let asw = w.a_size;
    let btw = w.b_tensors;
    let bsw = w.b_size;
    let dlw = w.delta;

    println!("  A: {repo_a}");
    println!("  B: {repo_b}");
    println!();

    println!(
        "  {:<dw$}  {:>atw$}  {:>asw$}  {:>btw$}  {:>bsw$}  {:>dlw$}",
        "Dtype", "A Tensors", "A Size", "B Tensors", "B Size", "\u{0394} Size",
    );

    // Unified dtype order: union, sorted by max(A bytes, B bytes) descending.
    let mut all_dtypes: Vec<&str> = rows_a
        .iter()
        .chain(rows_b.iter())
        .map(|r| r.dtype.as_str())
        .collect::<BTreeSet<&str>>()
        .into_iter()
        .collect();
    all_dtypes.sort_by_key(|dtype| {
        let a_bytes = rows_a
            .iter()
            .find(|r| r.dtype == *dtype)
            .map_or(0, |r| r.bytes);
        let b_bytes = rows_b
            .iter()
            .find(|r| r.dtype == *dtype)
            .map_or(0, |r| r.bytes);
        std::cmp::Reverse(a_bytes.max(b_bytes))
    });

    for dtype in &all_dtypes {
        let a = rows_a.iter().find(|r| r.dtype == *dtype);
        let b = rows_b.iter().find(|r| r.dtype == *dtype);
        let a_tensors = a.map_or_else(|| "\u{2014}".to_owned(), |r| r.tensors.to_string());
        let b_tensors = b.map_or_else(|| "\u{2014}".to_owned(), |r| r.tensors.to_string());
        let a_size = a.map_or_else(|| "\u{2014}".to_owned(), |r| format_size(r.bytes));
        let b_size = b.map_or_else(|| "\u{2014}".to_owned(), |r| format_size(r.bytes));
        // CAST: u64 → i64 clipping at i64::MAX; tensor bytes never approach 9.2 EiB.
        let a_bytes_i = a.map_or(0_i64, |r| i64::try_from(r.bytes).unwrap_or(i64::MAX));
        let b_bytes_i = b.map_or(0_i64, |r| i64::try_from(r.bytes).unwrap_or(i64::MAX));
        let delta_str = format_size_delta(b_bytes_i.saturating_sub(a_bytes_i));
        println!(
            "  {dtype:<dw$}  {a_tensors:>atw$}  {a_size:>asw$}  {b_tensors:>btw$}  {b_size:>bsw$}  {delta_str:>dlw$}",
        );
    }

    // Footer separator: row width minus the 2-char leading indent (which the
    // `"  "` literal in the println below contributes).
    println!("  {}", "\u{2500}".repeat(w.total_width().saturating_sub(2)));

    // Footer totals.
    let total_a_tensors: usize = rows_a.iter().map(|r| r.tensors).sum();
    let total_b_tensors: usize = rows_b.iter().map(|r| r.tensors).sum();
    let total_a_bytes: u64 = rows_a.iter().map(|r| r.bytes).sum();
    let total_b_bytes: u64 = rows_b.iter().map(|r| r.bytes).sum();
    // CAST: usize → i64 for tensor-count delta; clipping at i64::MAX is safe.
    let delta_tensors = i64::try_from(total_b_tensors)
        .unwrap_or(i64::MAX)
        .saturating_sub(i64::try_from(total_a_tensors).unwrap_or(i64::MAX));
    // CAST: u64 → i64 for byte delta; clipping at i64::MAX is safe.
    let delta_bytes = i64::try_from(total_b_bytes)
        .unwrap_or(i64::MAX)
        .saturating_sub(i64::try_from(total_a_bytes).unwrap_or(i64::MAX));

    let footer_core = format!(
        "  A: {} {}, {} | B: {} {}, {} | \u{0394}: {} tensors, {}",
        total_a_tensors,
        pluralize(total_a_tensors, "tensor", "tensors"),
        format_size(total_a_bytes),
        total_b_tensors,
        pluralize(total_b_tensors, "tensor", "tensors"),
        format_size(total_b_bytes),
        format_count_delta(delta_tensors),
        format_size_delta(delta_bytes),
    );
    if let Some(p) = filter {
        println!("{footer_core} (filter: {p:?})");
    } else {
        println!("{footer_core}");
    }
}

/// Inspects tensor file headers (`.safetensors` / `.gguf` / `.npz` / `.pth`)
/// for tensor metadata, upgrading raw 401/403 content errors into a
/// gated-repo diagnosis on the way out.
#[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments)]
fn run_inspect(
    repo_id: &str,
    filename: Option<&str>,
    revision: Option<&str>,
    token: Option<&str>,
    cached: bool,
    list: bool,
    pick: bool,
    no_metadata: bool,
    json: bool,
    filter: Option<&str>,
    dtypes: bool,
    limit: Option<usize>,
    tree: bool,
    check_gpu: Option<u32>,
    context: Option<u32>,
) -> Result<(), FetchError> {
    if list {
        return run_inspect_list(repo_id, revision, token, cached);
    }
    if pick {
        let picked = pick_inspect_file(repo_id, filename, revision, token, cached)?;
        // BORROW: explicit .as_str() for String → &str argument
        return run_inspect_single(
            repo_id,
            picked.as_str(),
            revision,
            token,
            cached,
            no_metadata,
            json,
            filter,
            dtypes,
            limit,
            tree,
            check_gpu,
            context,
        )
        .map_err(|e| enrich_gated_content_error(e, repo_id, token));
    }
    match filename {
        Some(f) => {
            let resolved = resolve_inspect_filename_arg(f, repo_id, revision, token, cached)?;
            // BORROW: explicit .as_str() for String → &str argument
            run_inspect_single(
                repo_id,
                resolved.as_str(),
                revision,
                token,
                cached,
                no_metadata,
                json,
                filter,
                dtypes,
                limit,
                tree,
                check_gpu,
                context,
            )
            .map_err(|e| enrich_gated_content_error(e, repo_id, token))
        }
        None => run_inspect_repo(
            repo_id, revision, token, cached, json, filter, dtypes, limit, tree, check_gpu, context,
        )
        .map_err(|e| enrich_gated_content_error(e, repo_id, token)),
    }
}

/// Returns `true` when an inspect-path error is an HTTP `401` / `403`
/// status failure — the signature of gated-content rejection (the Hub
/// serves a gated repo's metadata publicly; only content requests carry
/// the gate, so the failure surfaces here rather than at listing time).
fn is_auth_status_error(err: &FetchError) -> bool {
    matches!(err, FetchError::Http(msg)
        if msg.contains("returned status 401") || msg.contains("returned status 403"))
}

/// Upgrades a raw 401/403 content error into a gated-repo diagnosis.
///
/// `download` has had a gated-model pre-flight since v0.9.3; the `inspect`
/// and `diff` Range paths historically surfaced the raw `403 Forbidden`
/// instead. On a status-401/403 error this makes one best-effort metadata
/// probe and, when the repo is confirmed gated, replaces the raw HTTP
/// error with the same actionable wording the download pre-flight uses
/// (license link + token guidance). Any probe failure — network error,
/// private repo, no runtime — returns the original error untouched.
/// `diff` wraps each side's fetch separately, so the failing repo is
/// always named precisely.
fn enrich_gated_content_error(err: FetchError, repo_id: &str, token: Option<&str>) -> FetchError {
    if !is_auth_status_error(&err) {
        return err;
    }

    let Ok(rt) = tokio::runtime::Runtime::new() else {
        return err;
    };
    let Ok(metadata) = rt.block_on(discover::fetch_model_card(repo_id)) else {
        return err;
    };
    if !metadata.gated.is_gated() {
        return err;
    }

    // BORROW: explicit .map(ToOwned::to_owned) for Option<&str> → Option<String>
    let effective_token = token
        .map(ToOwned::to_owned)
        .or_else(|| std::env::var("HF_TOKEN").ok());

    let reason = if effective_token.is_none() {
        format!(
            "{repo_id} is a gated model — its file listing is public but content \
             requires access: accept the license at https://huggingface.co/{repo_id} \
             and set HF_TOKEN or pass --token"
        )
    } else {
        format!(
            "{repo_id} is a gated model and your token was rejected — accept the \
             license at https://huggingface.co/{repo_id} (each gated family is \
             licensed separately) and check that the token grants gated-repo read access"
        )
    };
    FetchError::Auth { reason }
}

/// Resolves an inspect filename argument to a concrete filename.
///
/// If `arg` parses as a positive `usize`, treats it as a 1-based index into
/// the repository's alphabetically-sorted list of supported tensor files
/// (`.safetensors` / `.gguf` / `.npz` / `.pth`, same universe as `--list`)
/// and returns the corresponding filename. Otherwise returns `arg` unchanged.
///
/// When an index is resolved, a one-line `Resolving index N → <name>` note
/// is printed to stderr so the user can confirm the pick before the inspect
/// proceeds.
fn resolve_inspect_filename_arg(
    arg: &str,
    repo_id: &str,
    revision: Option<&str>,
    token: Option<&str>,
    cached: bool,
) -> Result<String, FetchError> {
    let Ok(n) = arg.parse::<usize>() else {
        // Not a number — treat as a literal filename.
        // BORROW: explicit .to_owned() for &str → owned String
        return Ok(arg.to_owned());
    };

    let (entries, commit_sha) = gather_tensor_listing(repo_id, revision, token, cached)?;

    if entries.is_empty() {
        return Err(FetchError::InvalidArgument(format!(
            "index {n} cannot be resolved: no supported tensor files \
             (.safetensors / .gguf / .npz / .pth) in repository {repo_id} \
             (run `hf-fm inspect {repo_id} --list` to confirm)"
        )));
    }

    if n == 0 || n > entries.len() {
        return Err(FetchError::InvalidArgument(format!(
            "index {n} is out of range (repository has {count} tensor files — \
             use 1..{count}; run `hf-fm inspect {repo_id} --list` to see them)",
            count = entries.len()
        )));
    }

    // INDEX: n is bounded by 1..=entries.len() checked above
    #[allow(clippy::indexing_slicing)]
    let (filename, _size) = &entries[n - 1];

    // Transparency: show what the index resolved to before proceeding.
    let rev_note = match &commit_sha {
        Some(sha) => format!(" (repo rev: {})", short_sha(sha)),
        None => String::new(),
    };
    eprintln!("Resolving index {n} → {filename}{rev_note}");

    // BORROW: explicit .clone() for owned String result
    Ok(filename.clone())
}

/// Returns a short (12-char) prefix of a commit SHA for display.
fn short_sha(sha: &str) -> String {
    // BORROW: explicit .to_owned() for &str → owned String fallback
    sha.get(..12).map_or_else(|| sha.to_owned(), str::to_owned)
}

/// Fetches the `(filename, size_bytes)` list of supported tensor files
/// (`.safetensors` / `.gguf` / `.npz` / `.pth`) for a repo, from either the
/// local cache or the `HuggingFace` API, sorted alphabetically.
///
/// Also returns the commit SHA of the resolved revision when available.
fn gather_tensor_listing(
    repo_id: &str,
    revision: Option<&str>,
    token: Option<&str>,
    cached: bool,
) -> Result<inspect::TensorFileListing, FetchError> {
    if cached {
        return inspect::list_cached_tensor_files(repo_id, revision);
    }

    // BORROW: explicit .to_owned() for Option<&str> → Option<String>
    let resolved_token = token
        .map(ToOwned::to_owned)
        .or_else(|| std::env::var("HF_TOKEN").ok());

    let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
        path: PathBuf::from("<runtime>"),
        source: e,
    })?;
    let client = hf_fetch_model::build_client(resolved_token.as_deref())?;
    let (files, commit_sha) = rt.block_on(repo::list_repo_files_with_commit(
        repo_id,
        resolved_token.as_deref(),
        revision,
        &client,
    ))?;

    let mut entries: Vec<(String, u64)> = files
        .into_iter()
        // BORROW: explicit .as_str() — predicate takes &str
        .filter(|f| inspect::is_supported_tensor_file(f.filename.as_str()))
        .map(|f| (f.filename, f.size.unwrap_or(0)))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    Ok((entries, commit_sha))
}

/// Resolves the `--pick` file choice for `inspect`.
///
/// Narrows the repo's tensor-file listing by an optional case-insensitive
/// substring (`needle`): a unique match auto-resolves with a transparency
/// note on stderr; several matches print a numbered table and prompt on
/// stderr, reading the selection from stdin. The prompt channel is stderr
/// throughout, so `--json` output on stdout survives redirection.
fn pick_inspect_file(
    repo_id: &str,
    needle: Option<&str>,
    revision: Option<&str>,
    token: Option<&str>,
    cached: bool,
) -> Result<String, FetchError> {
    if !(std::io::stdin().is_terminal() && std::io::stderr().is_terminal()) {
        return Err(FetchError::InvalidArgument(
            "--pick requires an interactive terminal (stdin + stderr attached): \
             run `hf-fm inspect <repo> --list`, then `hf-fm inspect <repo> <n>`"
                .to_owned(),
        ));
    }

    let (entries, commit_sha) = gather_tensor_listing(repo_id, revision, token, cached)?;
    let candidates = narrow_pick_candidates(&entries, needle);

    // Same transparency suffix as the numeric-index path: show which commit
    // the listing was resolved against when known.
    let rev_note = match &commit_sha {
        Some(sha) => format!(" (repo rev: {})", short_sha(sha)),
        None => String::new(),
    };

    match candidates.as_slice() {
        [] => Err(FetchError::InvalidArgument(match needle {
            Some(n) => format!(
                "no tensor files match {n:?} in {repo_id} \
                 (run `hf-fm inspect {repo_id} --list` to see all)"
            ),
            None => format!(
                "no supported tensor files (.safetensors / .gguf / .npz / .pth) \
                 in repository {repo_id}"
            ),
        })),
        [(filename, _size)] => {
            eprintln!("Resolving to {filename}{rev_note}");
            // BORROW: explicit .clone() for owned String result
            Ok((*filename).clone())
        }
        // EXPLICIT: two or more candidates — hand off to the prompt loop
        _ => prompt_pick_selection(repo_id, needle, &candidates, &rev_note),
    }
}

/// Prints the numbered candidate table on stderr and loops the `Pick` prompt
/// until a valid selection, returning the chosen filename.
///
/// Empty input / EOF cancels with a non-zero exit (`cancelled — no file
/// picked`); unparseable or out-of-range input re-prompts without crashing.
fn prompt_pick_selection(
    repo_id: &str,
    needle: Option<&str>,
    candidates: &[&(String, u64)],
    rev_note: &str,
) -> Result<String, FetchError> {
    match needle {
        Some(n) => eprintln!("Multiple tensor files match {n:?} in {repo_id}:"),
        None => eprintln!("Tensor files in {repo_id}:"),
    }

    // Same dynamic column-width idiom as `run_inspect_list`, rendered on
    // stderr (the prompt channel) instead of stdout.
    let count = candidates.len();
    let index_width = count.to_string().len();
    let file_width = candidates.iter().map(|(f, _)| f.len()).max().unwrap_or(0);
    let size_strings: Vec<String> = candidates.iter().map(|(_, s)| format_size(*s)).collect();
    let size_width = size_strings.iter().map(String::len).max().unwrap_or(0);

    for (i, ((filename, _), size_str)) in candidates.iter().zip(size_strings.iter()).enumerate() {
        let n = i + 1;
        eprintln!("  {n:>index_width$}  {filename:<file_width$}  {size_str:>size_width$}");
    }

    // EXPLICIT: imperative retry loop — re-reads stdin until a valid pick
    // or a cancellation; an iterator idiom cannot express the re-prompt.
    loop {
        eprint!("Pick [1..{count}]: ");
        let mut input = String::new();
        let bytes_read = std::io::stdin()
            .read_line(&mut input)
            .map_err(|e| FetchError::Io {
                path: PathBuf::from("<stdin>"),
                source: e,
            })?;
        let trimmed = input.trim();
        if bytes_read == 0 || trimmed.is_empty() {
            // Ctrl-D / Ctrl-Z+Enter / bare Enter: bail rather than loop.
            return Err(FetchError::InvalidArgument(
                "cancelled — no file picked".to_owned(),
            ));
        }
        if let Some(n) = parse_pick_input(trimmed, count) {
            // INDEX: n is bounded by 1..=count via parse_pick_input
            #[allow(clippy::indexing_slicing)]
            let (filename, _size) = candidates[n - 1];
            eprintln!("Resolving to {filename}{rev_note}");
            // BORROW: explicit .clone() for owned String result
            return Ok(filename.clone());
        }
        eprintln!(
            "invalid choice {trimmed:?} — enter a number between 1 and {count}, \
             or press Enter to cancel"
        );
    }
}

/// Narrows a tensor-file listing to the entries whose filename contains
/// `needle` case-insensitively; `None` keeps the full listing.
fn narrow_pick_candidates<'a>(
    entries: &'a [(String, u64)],
    needle: Option<&str>,
) -> Vec<&'a (String, u64)> {
    match needle {
        None => entries.iter().collect(),
        Some(raw) => {
            let needle_lc = raw.to_lowercase();
            entries
                .iter()
                .filter(|(filename, _)| filename.to_lowercase().contains(&needle_lc))
                .collect()
        }
    }
}

/// Parses one line of picker input into a 1-based selection index.
///
/// Returns `None` for anything that should re-prompt: non-integer text,
/// zero, or an index above `count`. Empty / EOF input is the caller's
/// cancellation path, not this function's concern.
fn parse_pick_input(line: &str, count: usize) -> Option<usize> {
    line.trim()
        .parse::<usize>()
        .ok()
        .filter(|n| (1..=count).contains(n))
}

/// Prints the numbered list of supported tensor files in `repo_id`.
///
/// Used for discovery: tells the user what filenames / indices they can pass
/// to a follow-up `hf-fm inspect <repo> <n>` run. Does not read file headers.
fn run_inspect_list(
    repo_id: &str,
    revision: Option<&str>,
    token: Option<&str>,
    cached: bool,
) -> Result<(), FetchError> {
    let (entries, commit_sha) = gather_tensor_listing(repo_id, revision, token, cached)?;

    println!("Repo: {repo_id}");
    let rev_label = revision.unwrap_or("main");
    match &commit_sha {
        Some(sha) => println!("Rev:  {sha} ({rev_label})"),
        None => println!("Rev:  (unknown) ({rev_label})"),
    }
    println!();

    if entries.is_empty() {
        println!(
            "No supported tensor files (.safetensors / .gguf / .npz / .pth) in this repository."
        );
        if cached {
            println!();
            println!("Hint: the repo may not be cached locally. Try without --cached.");
        }
        return Ok(());
    }

    let count = entries.len();
    // Column widths: index gutter matches the highest number; filename/size scale to data.
    let index_width = count.to_string().len();
    let file_width = entries
        .iter()
        .map(|(f, _)| f.len())
        .max()
        .unwrap_or(4)
        .max(4); // BORROW: "File".len()
    let size_strings: Vec<String> = entries.iter().map(|(_, s)| format_size(*s)).collect();
    let size_width = size_strings
        .iter()
        .map(String::len)
        .max()
        .unwrap_or(4)
        .max(4); // BORROW: "Size".len()

    println!(
        "{:>index_width$}  {:<file_width$}  {:>size_width$}",
        "#", "File", "Size"
    );
    println!(
        "{:->index_width$}  {:-<file_width$}  {:->size_width$}",
        "", "", ""
    );
    let mut total: u64 = 0;
    for (i, ((filename, size), size_str)) in entries.iter().zip(size_strings.iter()).enumerate() {
        let n = i + 1;
        println!("{n:>index_width$}  {filename:<file_width$}  {size_str:>size_width$}");
        total = total.saturating_add(*size);
    }
    println!();
    println!(
        "{count} {}, {} total",
        pluralize(count, "file", "files"),
        format_size(total)
    );

    // Reproducibility hints: only show when the user did not pin a revision.
    if revision.is_none() {
        if let Some(sha) = commit_sha.as_deref() {
            println!();
            println!(
                "Tip: run `hf-fm inspect {repo_id} <n>` to inspect file #n.\n     \
                 Pass `--revision {sha}` on both sides to lock against this view."
            );
        } else {
            println!();
            println!("Tip: run `hf-fm inspect {repo_id} <n>` to inspect file #n.");
        }
    }

    Ok(())
}

// ============================================================================
// --tree: hierarchical tensor-name view with auto-collapsing of numeric ranges
// ============================================================================

/// One node in a displayable tensor-name tree.
#[allow(clippy::exhaustive_enums)] // EXHAUSTIVE: internal view type; crate owns all render paths
#[derive(Debug, Clone)]
enum TreeNode {
    /// A single tensor (terminal leaf).
    Leaf(LeafNode),
    /// An internal node with named children.
    Branch(BranchNode),
    /// A collapsed numeric range like `layers.[0..27]` with shared sub-structure.
    Ranged(RangedNode),
}

/// Leaf node: one tensor with its display-relevant metadata.
#[derive(Debug, Clone)]
struct LeafNode {
    /// Segment(s) of the tensor name, collapsed from any single-child ancestor chain.
    name: String,
    /// Dtype string (`"BF16"`, `"F32"`, `"F8_E4M3"`, ...).
    dtype: String,
    /// Tensor shape.
    shape: Vec<usize>,
    /// Number of elements (product of shape).
    params: u64,
    /// Byte length of the tensor data.
    bytes: u64,
}

/// Internal branch: children under a common dotted prefix.
#[derive(Debug, Clone)]
struct BranchNode {
    /// Segment(s) for this level, collapsed from single-child ancestor chains.
    segment: String,
    /// Child nodes, sorted by segment (from `BTreeMap` iteration).
    children: Vec<TreeNode>,
    /// Aggregate tensor count across the subtree.
    total_tensors: usize,
    /// Aggregate parameter count across the subtree.
    total_params: u64,
    /// Aggregate byte count across the subtree.
    total_bytes: u64,
}

/// Collapsed numeric range: `layers.[0..N]` with an N+1-instance identical sub-structure.
#[derive(Debug, Clone)]
struct RangedNode {
    /// Segment name (e.g., `"layers"`).
    segment: String,
    /// Inclusive start index.
    range_start: usize,
    /// Inclusive end index.
    range_end: usize,
    /// Sub-structure appearing once; multiplied by `count` instances at display time.
    template: Vec<TreeNode>,
    /// Aggregate tensor count across all instances.
    total_tensors: usize,
    /// Aggregate parameter count across all instances.
    total_params: u64,
    /// Aggregate byte count across all instances.
    total_bytes: u64,
}

impl TreeNode {
    /// Aggregate tensor count at this subtree (including all instances for `Ranged`).
    fn total_tensors(&self) -> usize {
        match self {
            Self::Leaf(_) => 1,
            Self::Branch(b) => b.total_tensors,
            Self::Ranged(r) => r.total_tensors,
        }
    }

    /// Aggregate parameter count at this subtree.
    fn total_params(&self) -> u64 {
        match self {
            Self::Leaf(l) => l.params,
            Self::Branch(b) => b.total_params,
            Self::Ranged(r) => r.total_params,
        }
    }

    /// Aggregate byte count at this subtree.
    fn total_bytes(&self) -> u64 {
        match self {
            Self::Leaf(l) => l.bytes,
            Self::Branch(b) => b.total_bytes,
            Self::Ranged(r) => r.total_bytes,
        }
    }
}

/// Intermediate trie used while building a `TreeNode` tree.
///
/// Each node may terminate a tensor (if `tensor` is `Some`) AND/OR have children
/// (in the `BTreeMap`). Using a `BTreeMap` gives deterministic, sorted iteration.
#[derive(Debug, Default)]
struct TrieNode {
    /// Tensor whose full name terminates at this node, if any.
    tensor: Option<inspect::TensorInfo>,
    /// Child nodes keyed by segment, sorted alphabetically.
    children: BTreeMap<String, TrieNode>,
}

/// Builds a `TreeNode` forest from a slice of tensors.
///
/// Splits each tensor name on `.`, inserts into a trie, then converts to
/// `TreeNode`s, collapsing single-child chains into dotted paths.
fn build_tree(tensors: &[inspect::TensorInfo]) -> Vec<TreeNode> {
    let mut root = TrieNode::default();
    for t in tensors {
        // BORROW: explicit .as_str() instead of Deref coercion
        let segments: Vec<&str> = t.name.as_str().split('.').collect();
        insert_trie(&mut root, &segments, t.clone());
    }
    // Top-level nodes come from root's children (root itself has no segment).
    root.children
        .into_iter()
        .map(|(seg, child)| trie_to_tree(seg, child))
        .collect()
}

/// Inserts a tensor into the trie along a path of segments.
fn insert_trie(node: &mut TrieNode, segments: &[&str], tensor: inspect::TensorInfo) {
    // INDEX: slice split — first element and tail used; empty segments unreachable
    //        because .split('.') on a non-empty string always yields at least one item
    let Some((head, rest)) = segments.split_first() else {
        node.tensor = Some(tensor);
        return;
    };
    if rest.is_empty() {
        // BORROW: (*head).to_owned() for &&str → String key
        let child = node.children.entry((*head).to_owned()).or_default();
        child.tensor = Some(tensor);
    } else {
        // BORROW: (*head).to_owned() for &&str → String key
        let child = node.children.entry((*head).to_owned()).or_default();
        insert_trie(child, rest, tensor);
    }
}

/// Converts a trie node into a `TreeNode`, collapsing single-child chains.
fn trie_to_tree(segment: String, mut node: TrieNode) -> TreeNode {
    // No children: must be a leaf (or a degenerate empty node — treat as empty branch).
    if node.children.is_empty() {
        if let Some(tensor) = node.tensor {
            // Compute stats before moving out dtype/shape.
            let params = tensor.num_elements();
            let bytes = tensor.byte_len();
            return TreeNode::Leaf(LeafNode {
                name: segment,
                dtype: tensor.dtype,
                shape: tensor.shape,
                params,
                bytes,
            });
        }
        // EXPLICIT: unreachable in well-formed input; fall through to empty branch for safety
        return TreeNode::Branch(BranchNode {
            segment,
            children: Vec::new(),
            total_tensors: 0,
            total_params: 0,
            total_bytes: 0,
        });
    }

    // Single-child collapse: if no own tensor and exactly one child, merge segments.
    if node.tensor.is_none() && node.children.len() == 1 {
        if let Some((child_segment, child)) = node.children.pop_first() {
            let merged = format!("{segment}.{child_segment}");
            return trie_to_tree(merged, child);
        }
    }

    // Multi-child branch: recurse into each, compute aggregates.
    let mut children: Vec<TreeNode> = node
        .children
        .into_iter()
        .map(|(seg, child)| trie_to_tree(seg, child))
        .collect();

    // If this node also carries its own tensor alongside children, surface it as
    // a pseudo-leaf with an empty name (rare in safetensors; kept for correctness).
    if let Some(tensor) = node.tensor {
        // Compute stats before moving out dtype/shape.
        let params = tensor.num_elements();
        let bytes = tensor.byte_len();
        children.insert(
            0,
            TreeNode::Leaf(LeafNode {
                name: String::new(),
                dtype: tensor.dtype,
                shape: tensor.shape,
                params,
                bytes,
            }),
        );
    }

    let total_tensors: usize = children.iter().map(TreeNode::total_tensors).sum();
    let total_params: u64 = children
        .iter()
        .map(TreeNode::total_params)
        .fold(0u64, u64::saturating_add);
    let total_bytes: u64 = children
        .iter()
        .map(TreeNode::total_bytes)
        .fold(0u64, u64::saturating_add);

    TreeNode::Branch(BranchNode {
        segment,
        children,
        total_tensors,
        total_params,
        total_bytes,
    })
}

/// Post-processes a tree in place, collapsing numeric-indexed sibling branches
/// into `Ranged` nodes when their sub-structures match.
fn collapse_ranges(nodes: Vec<TreeNode>) -> Vec<TreeNode> {
    nodes.into_iter().map(collapse_node).collect()
}

fn collapse_node(node: TreeNode) -> TreeNode {
    match node {
        TreeNode::Leaf(_) => node,
        TreeNode::Branch(mut branch) => {
            // Recurse first: child-level collapses before parent-level check.
            branch.children = collapse_ranges(branch.children);
            try_collapse_range(&branch).map_or(TreeNode::Branch(branch), TreeNode::Ranged)
        }
        TreeNode::Ranged(mut ranged) => {
            // Already ranged — recurse into template anyway for nested structure.
            ranged.template = collapse_ranges(ranged.template);
            TreeNode::Ranged(ranged)
        }
    }
}

/// Checks whether a branch's children form a collapsible contiguous numeric range
/// `0..N` with structurally identical sub-trees. Returns the collapsed `RangedNode`
/// if so, or `None` if any requirement fails.
fn try_collapse_range(branch: &BranchNode) -> Option<RangedNode> {
    // Require at least 2 children; a single numeric child isn't a range.
    if branch.children.len() < 2 {
        return None;
    }

    // All children must be Branches with purely numeric segments.
    let mut indexed: Vec<(usize, &BranchNode)> = Vec::with_capacity(branch.children.len());
    for child in &branch.children {
        let TreeNode::Branch(sub) = child else {
            return None;
        };
        // BORROW: explicit .as_str() instead of Deref coercion
        let idx: usize = sub.segment.as_str().parse().ok()?;
        indexed.push((idx, sub));
    }

    // Indices must already be sorted ascending (BTreeMap order) — verify contiguous 0..N.
    indexed.sort_by_key(|(i, _)| *i);
    for (expected, (actual, _)) in indexed.iter().enumerate() {
        if expected != *actual {
            return None;
        }
    }

    // Structurally compare every branch's children against the first.
    // INDEX: indexed.len() >= 2 checked above, so indexed[0] is valid
    #[allow(clippy::indexing_slicing)]
    let (_, first_branch) = &indexed[0];
    #[allow(clippy::indexing_slicing)]
    for (_, other) in &indexed[1..] {
        if !branches_structurally_equal(first_branch, other) {
            return None;
        }
    }

    // Collapse: use the first branch's children as the template.
    let count = indexed.len();
    // CAST: usize → usize, no cast needed; range_end is last index
    let range_end = count.saturating_sub(1);

    Some(RangedNode {
        segment: branch.segment.clone(),
        range_start: 0,
        range_end,
        template: first_branch.children.clone(),
        total_tensors: branch.total_tensors,
        total_params: branch.total_params,
        total_bytes: branch.total_bytes,
    })
}

/// Compares two branches' children for structural equivalence. Ignores the top-level
/// segment (which is the numeric index — different by construction in a range).
fn branches_structurally_equal(a: &BranchNode, b: &BranchNode) -> bool {
    if a.children.len() != b.children.len() {
        return false;
    }
    a.children
        .iter()
        .zip(b.children.iter())
        .all(|(c1, c2)| nodes_structurally_equal(c1, c2))
}

/// Full structural equality including segments and leaf dtype/shape.
fn nodes_structurally_equal(a: &TreeNode, b: &TreeNode) -> bool {
    match (a, b) {
        (TreeNode::Leaf(l1), TreeNode::Leaf(l2)) => {
            l1.name == l2.name && l1.dtype == l2.dtype && l1.shape == l2.shape
        }
        (TreeNode::Branch(b1), TreeNode::Branch(b2)) => {
            b1.segment == b2.segment && branches_structurally_equal(b1, b2)
        }
        (TreeNode::Ranged(r1), TreeNode::Ranged(r2)) => {
            r1.segment == r2.segment
                && r1.range_end == r2.range_end
                && r1.template.len() == r2.template.len()
                && r1
                    .template
                    .iter()
                    .zip(r2.template.iter())
                    .all(|(c1, c2)| nodes_structurally_equal(c1, c2))
        }
        // EXPLICIT: mismatched variants cannot be structurally equal
        _ => false,
    }
}

// --- Human-readable rendering ---

/// Renders a tree forest to stdout using Unicode box-drawing connectors.
fn render_tree(nodes: &[TreeNode]) {
    render_children(nodes, "");
}

/// Renders children of a node, computing per-level leaf alignment.
fn render_children(children: &[TreeNode], prefix: &str) {
    // Compute max leaf-name width at THIS level for column alignment.
    let leaf_name_width: usize = children
        .iter()
        .filter_map(|c| match c {
            TreeNode::Leaf(l) => Some(l.name.len()),
            // EXPLICIT: branch/ranged children don't contribute to leaf-name width
            TreeNode::Branch(_) | TreeNode::Ranged(_) => None,
        })
        .max()
        .unwrap_or(0);

    // Max dtype width for aligned leaf rows.
    let dtype_width: usize = children
        .iter()
        .filter_map(|c| match c {
            TreeNode::Leaf(l) => Some(l.dtype.len()),
            // EXPLICIT: branch/ranged children don't contribute to dtype width
            TreeNode::Branch(_) | TreeNode::Ranged(_) => None,
        })
        .max()
        .unwrap_or(0);

    for (i, child) in children.iter().enumerate() {
        let is_last = i + 1 == children.len();
        render_node(child, prefix, is_last, leaf_name_width, dtype_width);
    }
}

fn render_node(
    node: &TreeNode,
    prefix: &str,
    is_last: bool,
    leaf_name_width: usize,
    dtype_width: usize,
) {
    let connector = if is_last { "└── " } else { "├── " };
    let indent = if is_last { "    " } else { "│   " };

    match node {
        TreeNode::Leaf(leaf) => {
            let shape_str = format!("{:?}", leaf.shape);
            let size_str = format_size(leaf.bytes);
            // Leaf columns: name (padded) | dtype (padded) | shape | size
            println!(
                "  {prefix}{connector}{name:<nw$}  {dtype:<dw$}  {shape_str}  {size_str}",
                name = leaf.name,
                dtype = leaf.dtype,
                nw = leaf_name_width,
                dw = dtype_width,
            );
        }
        TreeNode::Branch(branch) => {
            println!(
                "  {prefix}{connector}{seg}.",
                seg = branch.segment.as_str(), // BORROW: explicit .as_str()
            );
            let new_prefix = format!("{prefix}{indent}");
            render_children(&branch.children, new_prefix.as_str()); // BORROW: explicit .as_str()
        }
        TreeNode::Ranged(ranged) => {
            let count = ranged.range_end - ranged.range_start + 1;
            println!(
                "  {prefix}{connector}{seg}.[{start}..{end}].   (\u{00d7}{count})",
                seg = ranged.segment.as_str(), // BORROW: explicit .as_str()
                start = ranged.range_start,
                end = ranged.range_end,
            );
            let new_prefix = format!("{prefix}{indent}");
            render_children(&ranged.template, new_prefix.as_str()); // BORROW: explicit .as_str()
        }
    }
}

// --- JSON rendering ---

/// JSON node: tagged enum mirroring `TreeNode` for serialization.
#[derive(serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TreeJsonNode<'a> {
    Leaf {
        name: &'a str,
        dtype: &'a str,
        shape: &'a [usize],
        params: u64,
        bytes: u64,
    },
    Branch {
        name: &'a str,
        tensors: usize,
        params: u64,
        bytes: u64,
        children: Vec<TreeJsonNode<'a>>,
    },
    Ranged {
        name: &'a str,
        range_start: usize,
        range_end: usize,
        count: usize,
        tensors: usize,
        params: u64,
        bytes: u64,
        template: Vec<TreeJsonNode<'a>>,
    },
}

/// Top-level JSON wrapper for `--tree --json`.
#[derive(serde::Serialize)]
struct TreeJsonOutput<'a> {
    repo_id: &'a str,
    filename: &'a str,
    total_tensors: usize,
    total_params: u64,
    tree: Vec<TreeJsonNode<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_check: Option<serde_json::Value>,
}

fn tree_to_json(nodes: &[TreeNode]) -> Vec<TreeJsonNode<'_>> {
    nodes.iter().map(node_to_json).collect()
}

/// Builds, collapses, and prints a tensor-name tree to stdout.
///
/// `total_tensor_count` and `total_params` are used only for the footer line
/// (show `X/Y tensors` when `filter` is active, or `N tensors` otherwise).
fn print_tree_summary(
    tensors: &[inspect::TensorInfo],
    filter: Option<&str>,
    total_tensor_count: usize,
    total_params: u64,
) {
    let forest = collapse_ranges(build_tree(tensors));
    println!();
    render_tree(&forest);

    // Footer: mirror the regular inspect footer conventions.
    let shown: usize = forest.iter().map(TreeNode::total_tensors).sum();
    let shown_params: u64 = forest
        .iter()
        .map(TreeNode::total_params)
        .fold(0u64, u64::saturating_add);
    let tensor_label = if shown == 1 { "tensor" } else { "tensors" };
    if let Some(pattern) = filter {
        println!(
            "  {shown}/{total_tensor_count} {tensor_label}, {}/{} params (filter: {pattern:?})",
            inspect::format_params(shown_params),
            inspect::format_params(total_params),
        );
    } else {
        println!(
            "  {shown} {tensor_label}, {} params",
            inspect::format_params(shown_params),
        );
    }
}

/// Builds, collapses, and emits the tree as JSON to stdout.
fn print_tree_json(
    repo_id: &str,
    filename: &str,
    tensors: &[inspect::TensorInfo],
    total_tensor_count: usize,
    total_params: u64,
    gpu_check: Option<serde_json::Value>,
) -> Result<(), FetchError> {
    let forest = collapse_ranges(build_tree(tensors));
    let output = TreeJsonOutput {
        repo_id,
        filename,
        total_tensors: total_tensor_count,
        total_params,
        tree: tree_to_json(&forest),
        gpu_check,
    };
    let serialized = serde_json::to_string_pretty(&output)
        .map_err(|e| FetchError::Http(format!("failed to serialize JSON: {e}")))?;
    println!("{serialized}");
    Ok(())
}

fn node_to_json(node: &TreeNode) -> TreeJsonNode<'_> {
    match node {
        TreeNode::Leaf(l) => TreeJsonNode::Leaf {
            name: l.name.as_str(),     // BORROW: explicit .as_str()
            dtype: l.dtype.as_str(),   // BORROW: explicit .as_str()
            shape: l.shape.as_slice(), // BORROW: explicit .as_slice()
            params: l.params,
            bytes: l.bytes,
        },
        TreeNode::Branch(b) => TreeJsonNode::Branch {
            name: b.segment.as_str(), // BORROW: explicit .as_str()
            tensors: b.total_tensors,
            params: b.total_params,
            bytes: b.total_bytes,
            children: tree_to_json(&b.children),
        },
        TreeNode::Ranged(r) => {
            let count = r.range_end - r.range_start + 1;
            TreeJsonNode::Ranged {
                name: r.segment.as_str(), // BORROW: explicit .as_str()
                range_start: r.range_start,
                range_end: r.range_end,
                count,
                tensors: r.total_tensors,
                params: r.total_params,
                bytes: r.total_bytes,
                template: tree_to_json(&r.template),
            }
        }
    }
}

// ============================================================================

/// Truncation metadata added to `--json` output when `--limit` cuts the tensor list short.
#[derive(serde::Serialize)]
struct TruncationInfo {
    /// Number of tensors in the `tensors` array (after filter and limit).
    shown: usize,
    /// Total tensors in the file (before any filter or limit).
    total: usize,
}

/// JSON wrapper that adds a top-level `truncated` field when the output was capped by `--limit`.
///
/// The field is omitted entirely when the tensor list is complete, preserving
/// the plain `SafetensorsHeaderInfo` schema for non-truncated output.
///
/// The `gpu_check` field follows the same pattern: present only when
/// `--check-gpu` was passed, omitted otherwise so v0.10.0 consumers see
/// byte-identical output for non-`--check-gpu` invocations.
#[derive(serde::Serialize)]
struct InspectJsonOutput<'a> {
    #[serde(flatten)]
    header: &'a inspect::SafetensorsHeaderInfo,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncated: Option<TruncationInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_check: Option<serde_json::Value>,
}

/// Inspects a single `.safetensors` file and prints the result.
// EXPLICIT: composes header fetch, filter/tree/dtypes/limit branching, and
// JSON-vs-table output formatting. Splitting would obscure the inspect mode
// matrix.
#[allow(
    clippy::fn_params_excessive_bools,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
fn run_inspect_single(
    repo_id: &str,
    filename: &str,
    revision: Option<&str>,
    token: Option<&str>,
    cached: bool,
    no_metadata: bool,
    json: bool,
    filter: Option<&str>,
    dtypes: bool,
    limit: Option<usize>,
    tree: bool,
    check_gpu: Option<u32>,
    context: Option<u32>,
) -> Result<(), FetchError> {
    // Classify extension once; v0.10.3 dispatches across .safetensors (remote
    // or cached) and .gguf / .npz / .pth (cached only — remote inspect for
    // these three formats arrives in v0.11 via the planned `HttpRangeReader`
    // adapter).
    let ext_lc = Path::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    let is_safetensors = ext_lc.as_deref() == Some("safetensors");
    let is_gguf = ext_lc.as_deref() == Some("gguf");
    let is_npz = ext_lc.as_deref() == Some("npz");
    let is_pth = ext_lc.as_deref() == Some("pth");

    if !is_safetensors && !is_gguf && !is_npz && !is_pth {
        // BORROW: owned String for the error variant field
        let extension = ext_lc.unwrap_or_else(|| "unknown".to_owned());
        return Err(FetchError::UnsupportedInspectFormat {
            filename: filename.to_owned(),
            extension,
        });
    }

    if (is_gguf || is_npz || is_pth) && !cached {
        let format_label = if is_pth {
            "PTH"
        } else if is_npz {
            "NPZ"
        } else {
            "GGUF"
        };
        return Err(FetchError::InvalidArgument(format!(
            "remote {format_label} inspect not yet supported (planned for v0.11): \
             pass --cached after downloading {filename} with `hf-fm download`"
        )));
    }

    let (mut info, source) = if cached {
        let info = if is_gguf {
            inspect::inspect_gguf_cached(repo_id, filename, revision)?
        } else if is_npz {
            inspect::inspect_npz_cached(repo_id, filename, revision)?
        } else if is_pth {
            inspect::inspect_pth_cached(repo_id, filename, revision)?
        } else {
            inspect::inspect_safetensors_cached(repo_id, filename, revision)?
        };
        (info, inspect::InspectSource::Cached)
    } else {
        // Reachable only when is_safetensors == true (the .gguf/.npz/.pth
        // cached-only branch returned earlier; the unclassified branch
        // returned earlier).
        // BORROW: explicit String::from for Option<&str> → Option<String>
        let token = token
            .map(String::from)
            .or_else(|| std::env::var("HF_TOKEN").ok());

        let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
            path: PathBuf::from("<runtime>"),
            source: e,
        })?;
        // BORROW: explicit .as_deref() for Option<String> → Option<&str>
        rt.block_on(inspect::inspect_safetensors(
            repo_id,
            filename,
            token.as_deref(),
            revision,
        ))?
    };

    // `--check-gpu` uses the unfiltered model totals — fit is a whole-model
    // question, not a per-filtered-tensor question. Capture the model-wide
    // figures BEFORE the filter / limit pass below.
    let gpu_inputs = check_gpu.map(|idx| GpuCheckInputs {
        device_index: idx,
        weight_bytes: gpu_check::sum_tensor_bytes(&info.tensors),
        dtype_label: gpu_check::dominant_dtype_label(&info.tensors),
        total_params: info.total_params(),
        kv: compute_kv_inputs(repo_id, revision, token, cached, context),
    });

    // Apply tensor name filter.
    let total_tensor_count = info.tensors.len();
    let total_params = info.total_params();
    if let Some(pattern) = filter {
        // BORROW: explicit .as_str() instead of Deref coercion
        info.tensors
            .retain(|t| matches_filter(t.name.as_str(), pattern));
    }

    // Apply limit after filter. Track matched counts to report truncation.
    let matched_count = info.tensors.len();
    let matched_params = info.total_params();
    let truncated_by_limit = limit.is_some_and(|n| matched_count > n);
    if let Some(n) = limit {
        info.tensors.truncate(n);
    }

    // Probe the GPU once (after the filter/limit pass so the JSON value built
    // here can be handed directly to the early-return branches below). The
    // probe is a few-millisecond NVML / DXGI call; no need to skip it for
    // any output mode.
    let gpu_result = gpu_inputs
        .as_ref()
        .map(|i| gpu_check::query_gpu(i.device_index));
    let gpu_check_value = gpu_inputs.as_ref().zip(gpu_result.as_ref()).map(|(i, r)| {
        gpu_check::gpu_check_json(
            r,
            i.weight_bytes,
            i.dtype_label.as_str(),
            i.total_params,
            i.kv.as_ref(),
        )
    });

    // `--tree --json`: hierarchical tree as JSON (distinct schema from plain --json).
    if tree && json {
        return print_tree_json(
            repo_id,
            filename,
            &info.tensors,
            total_tensor_count,
            total_params,
            gpu_check_value,
        );
    }

    // `--dtypes --json`: compact dtype breakdown as JSON (distinct schema from plain --json).
    if dtypes && json {
        return print_dtype_summary_json(
            &info.tensors,
            total_tensor_count,
            total_params,
            gpu_check_value,
        );
    }

    if json {
        // `truncated` is `None` when the list is complete, which `skip_serializing_if`
        // suppresses — so non-truncated output is schema-identical to v0.9.5.
        let wrapped = InspectJsonOutput {
            header: &info,
            truncated: truncated_by_limit.then_some(TruncationInfo {
                shown: info.tensors.len(),
                total: total_tensor_count,
            }),
            gpu_check: gpu_check_value,
        };
        let output = serde_json::to_string_pretty(&wrapped)
            .map_err(|e| FetchError::Http(format!("failed to serialize JSON: {e}")))?;
        println!("{output}");
        return Ok(());
    }

    // Human-readable output.
    let source_label = match source {
        inspect::InspectSource::Cached => "cached",
        inspect::InspectSource::Remote => "remote (2 HTTP requests)",
        _ => "unknown",
    };
    println!("  Repo:     {repo_id}");
    println!("  File:     {filename}");
    println!("  Source:   {source_label}");

    println!(
        "  {}",
        format_header_line(info.header_size, info.file_size, is_gguf)
    );

    // v0.10.3 Phase C: surface quantization scheme + dequantised-size
    // estimate when present. Empty for unquantized safetensors and all
    // non-safetensors formats; the absence communicates full precision.
    for line in format_quant_lines(info.quant_info.as_ref()) {
        println!("  {line}");
    }

    if !no_metadata {
        if let Some(ref meta) = info.metadata {
            for line in format_metadata_lines(meta) {
                println!("  {line}");
            }
        }
    }

    // Hierarchical tree mode.
    if tree {
        print_tree_summary(&info.tensors, filter, total_tensor_count, total_params);
        maybe_print_gpu_check(gpu_inputs.as_ref(), gpu_result.as_ref());
        return Ok(());
    }

    // Per-dtype summary mode.
    if dtypes {
        print_dtype_summary(&info.tensors, filter, total_tensor_count, total_params);
        maybe_print_gpu_check(gpu_inputs.as_ref(), gpu_result.as_ref());
        return Ok(());
    }

    // Compute dynamic column widths from the actual data.
    let nw = info
        .tensors
        .iter()
        .map(|t| t.name.len())
        .max()
        .unwrap_or(6)
        .max(6); // BORROW: "Tensor".len()
    let shape_strs: Vec<String> = info
        .tensors
        .iter()
        .map(|t| format!("{:?}", t.shape))
        .collect();
    let sw = shape_strs.iter().map(String::len).max().unwrap_or(5).max(5); // BORROW: "Shape".len()
    let row_width = nw + 2 + 8 + sw + 2 + 10 + 2 + 10;

    println!();
    println!(
        "  {:<nw$} {:<8} {:<sw$} {:>10} {:>10}",
        "Tensor", "Dtype", "Shape", "Size", "Params",
    );

    for (t, shape_str) in info.tensors.iter().zip(shape_strs.iter()) {
        let size_str = format_size(t.byte_len());
        let params_str = inspect::format_params(t.num_elements());
        println!(
            "  {:<nw$} {:<8} {:<sw$} {:>10} {:>10}",
            t.name, t.dtype, shape_str, size_str, params_str,
        );
    }

    println!("  {}", "\u{2500}".repeat(row_width));
    let shown_count = info.tensors.len();
    let shown_params = info.total_params();
    let tensor_label = if shown_count == 1 {
        "tensor"
    } else {
        "tensors"
    };

    match (filter.is_some(), truncated_by_limit) {
        (false, false) => {
            println!(
                "  {shown_count} {tensor_label}, {} params",
                inspect::format_params(shown_params)
            );
        }
        (true, false) => {
            let filter_str = filter.unwrap_or_default();
            println!(
                "  Showing {shown_count} of {total_tensor_count} tensors matching filter {filter_str:?}."
            );
            println!(
                "  Param counts: {} matching filter, {} total.",
                inspect::format_params(shown_params),
                inspect::format_params(total_params),
            );
        }
        (false, true) => {
            let limit_val = limit.unwrap_or(0);
            println!(
                "  Showing {shown_count} of {total_tensor_count} tensors (limit: {limit_val})."
            );
            println!(
                "  Param counts: {} shown, {} total.",
                inspect::format_params(shown_params),
                inspect::format_params(total_params),
            );
        }
        (true, true) => {
            let filter_str = filter.unwrap_or_default();
            let limit_val = limit.unwrap_or(0);
            println!(
                "  Showing {shown_count} of {matched_count} tensors matching filter {filter_str:?} ({total_tensor_count} tensors total, limit: {limit_val})."
            );
            println!(
                "  Param counts: {} shown, {} matching filter, {} total.",
                inspect::format_params(shown_params),
                inspect::format_params(matched_params),
                inspect::format_params(total_params),
            );
        }
    }

    maybe_print_gpu_check(gpu_inputs.as_ref(), gpu_result.as_ref());
    Ok(())
}

/// Model-side inputs for one `--check-gpu` invocation, captured **before** the
/// `--filter` / `--limit` pass so the verdict reflects the whole-model totals.
///
/// Travels in lockstep with a single [`gpu_check::GpuCheckResult`] from
/// [`gpu_check::query_gpu`]; the pair drives both the text renderer
/// ([`maybe_print_gpu_check`]) and the JSON renderer ([`gpu_check::gpu_check_json`]).
struct GpuCheckInputs {
    /// Zero-based device index the user passed via `--check-gpu N` (default 0).
    device_index: u32,
    /// Sum of tensor byte-lens across every tensor in the model (unfiltered).
    weight_bytes: u64,
    /// Display label for the `Model weights:` line — single dtype or
    /// `"<dominant> + others"`. See [`gpu_check::dominant_dtype_label`].
    dtype_label: String,
    /// Total parameter count across every tensor in the model (unfiltered).
    total_params: u64,
    /// KV-cache budget from `--context N`, or `None` when not requested.
    kv: Option<gpu_check::KvComputed>,
}

/// Renders the `--check-gpu` verdict block when both inputs are present.
///
/// `inputs` and `probe` are paired at construction in the callers — they are
/// always either both `Some` or both `None`. The two-`Option` signature is
/// preserved (rather than collapsed into one) so the caller can hold each
/// half by reference without an extra borrow dance through a wrapper struct.
fn maybe_print_gpu_check(
    inputs: Option<&GpuCheckInputs>,
    probe: Option<&gpu_check::GpuCheckResult>,
) {
    if let (Some(i), Some(result)) = (inputs, probe) {
        gpu_check::print_gpu_check(
            result,
            i.weight_bytes,
            i.dtype_label.as_str(),
            i.total_params,
            i.kv.as_ref(),
        );
    }
}

/// Display label for the KV-cache element dtype shown on the `KV cache` line.
///
/// Derived from the model's `config.json` `torch_dtype`; defaults to `"BF16"`
/// for unknown / absent dtypes, matching [`inspect::torch_dtype_bytes`]'s
/// 2-byte default.
fn torch_dtype_label(torch_dtype: Option<&str>) -> &'static str {
    match torch_dtype {
        Some("float16") => "FP16",
        Some("float32" | "float") => "FP32",
        Some("float8_e4m3fn" | "float8_e5m2") => "FP8",
        // bfloat16 and any unknown or absent dtype.
        _ => "BF16",
    }
}

/// Fetches a model's `config.json` for KV budgeting — cache-only under
/// `--cached`, else cache-first with an HTTP fallback.
///
/// Best-effort: returns `None` on any miss or error (the caller renders an
/// "unavailable" KV line). KV budgeting never fails the inspect command.
fn fetch_model_config_for_inspect(
    repo_id: &str,
    revision: Option<&str>,
    token: Option<&str>,
    cached: bool,
) -> Option<inspect::ModelConfig> {
    if cached {
        return inspect::fetch_model_config_cached(repo_id, revision)
            .ok()
            .flatten();
    }
    let rt = tokio::runtime::Runtime::new().ok()?;
    rt.block_on(inspect::fetch_model_config(repo_id, token, revision))
        .ok()
        .flatten()
}

/// Builds the KV-cache verdict bundle for `--check-gpu --context N`.
///
/// Returns `None` when `--context` was not requested. When it was, always
/// returns `Some`: an unreadable or dimension-less `config.json` yields an
/// [`gpu_check::KvCachePath::Unavailable`] bundle so the verdict can say so
/// without gating the command.
fn compute_kv_inputs(
    repo_id: &str,
    revision: Option<&str>,
    token: Option<&str>,
    cached: bool,
    context: Option<u32>,
) -> Option<gpu_check::KvComputed> {
    let ctx = context?;

    let Some(cfg) = fetch_model_config_for_inspect(repo_id, revision, token, cached) else {
        return Some(gpu_check::KvComputed {
            context: ctx,
            elem_bytes: 0,
            dtype_label: String::new(),
            bytes: None,
            path: gpu_check::KvCachePath::Unavailable {
                reason: "no config.json in repo",
            },
        });
    };

    // BORROW: explicit .as_deref() for Option<String> → Option<&str>
    let elem_bytes = inspect::torch_dtype_bytes(cfg.torch_dtype.as_deref());
    // BORROW: explicit .as_deref()/.to_owned() for the dtype display label
    let dtype_label = torch_dtype_label(cfg.torch_dtype.as_deref()).to_owned();
    let (bytes, path) = gpu_check::kv_cache_bytes(&cfg, u64::from(ctx), elem_bytes);

    Some(gpu_check::KvComputed {
        context: ctx,
        elem_bytes,
        dtype_label,
        bytes,
        path,
    })
}

/// Inspects all `.safetensors` files in a repository (summary or per-file).
///
/// When any of `dtypes`, `limit`, or `tree` is set, the shard-index fast path
/// is bypassed and every shard's header is fetched so the per-tensor data
/// can be flattened across shards and rolled up by the existing
/// `--dtypes` / `--tree` / per-tensor renderers — see
/// [`run_inspect_repo_aggregated`].
// EXPLICIT: linear cache-vs-network branching and shard-index fast path,
// followed by a flatten-and-render fallthrough. Splitting hides the
// aggregation gate.
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
fn run_inspect_repo(
    repo_id: &str,
    revision: Option<&str>,
    token: Option<&str>,
    cached: bool,
    json: bool,
    filter: Option<&str>,
    dtypes: bool,
    limit: Option<usize>,
    tree: bool,
    check_gpu: Option<u32>,
    context: Option<u32>,
) -> Result<(), FetchError> {
    // When the user asked for tensor-level aggregation, every shard's header
    // must be read; the shard-index file alone has no dtype or shape data.
    // `--check-gpu` also needs per-tensor data to sum weight bytes precisely
    // across shards, so it forces the aggregation path too.
    let needs_aggregation = dtypes || tree || limit.is_some() || check_gpu.is_some();

    // The shard-index fast path renders a per-file rollup straight from
    // `model.safetensors.index.json` with no header reads, but it is
    // human-only — `print_shard_index_summary` takes no `json` flag. So
    // `--json` must bypass it and fall through to the header-reading
    // `print_multi_file_json`, which emits real per-tensor JSON. (v0.10.6
    // Symptom 2: `--json` was previously ignored on this path for sharded
    // repos, silently printing the human rollup instead.)
    let use_shard_fast_path = !needs_aggregation && !json;

    // KV bundle for `--context` (None when not requested, so no I/O on the
    // non-aggregation fast paths). Computed once and shared by both the cached
    // and network aggregation paths below.
    let kv = compute_kv_inputs(repo_id, revision, token, cached, context);

    if cached {
        // Cache-only: try shard index first, then walk snapshot.
        if use_shard_fast_path {
            if let Some(index) = inspect::fetch_shard_index_cached(repo_id, revision)? {
                print_shard_index_summary(repo_id, &index, filter);
                print_adapter_config_if_present(repo_id, revision, None, true, json);
                return Ok(());
            }
        }

        let results = inspect::inspect_repo_safetensors_cached(repo_id, revision)?;
        if results.is_empty() {
            println!("No cached .safetensors files found for {repo_id}.");
            println!("Hint: use `hf-fm list-files {repo_id}` to see available file types");
            return Ok(());
        }

        if needs_aggregation {
            run_inspect_repo_aggregated(
                repo_id,
                &results,
                json,
                filter,
                dtypes,
                limit,
                tree,
                check_gpu,
                kv.as_ref(),
            )?;
            print_adapter_config_if_present(repo_id, revision, None, true, json);
            return Ok(());
        }

        if json {
            print_multi_file_json(&results, filter)?;
            print_adapter_config_if_present(repo_id, revision, None, true, true);
            return Ok(());
        }

        print_multi_file_summary(repo_id, "cached", &results, filter);
        print_adapter_config_if_present(repo_id, revision, None, true, false);
        return Ok(());
    }

    // Network-enabled: try shard index first, then full inspection.
    // BORROW: explicit String::from for Option<&str> → Option<String>
    let token = token
        .map(String::from)
        .or_else(|| std::env::var("HF_TOKEN").ok());

    let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
        path: PathBuf::from("<runtime>"),
        source: e,
    })?;

    if use_shard_fast_path {
        // BORROW: explicit .as_deref() for Option<String> → Option<&str>
        let shard_index = rt.block_on(inspect::fetch_shard_index(
            repo_id,
            token.as_deref(),
            revision,
        ))?;

        if let Some(index) = shard_index {
            print_shard_index_summary(repo_id, &index, filter);
            print_adapter_config_if_present(repo_id, revision, token.as_deref(), false, json);
            return Ok(());
        }
    }

    // BORROW: explicit .as_deref() for Option<String> → Option<&str>
    let results = rt.block_on(inspect::inspect_repo_safetensors(
        repo_id,
        token.as_deref(),
        revision,
    ))?;

    if results.is_empty() {
        println!("No .safetensors files found in {repo_id}.");
        println!("Hint: use `hf-fm list-files {repo_id}` to see available file types");
        return Ok(());
    }

    // Compute the repo-level Source label from the per-file provenance before
    // `results` is consumed by the map below. The network path resolves each
    // file cache-first, so a partially-cached repo genuinely mixes the two.
    let sources: Vec<inspect::InspectSource> = results.iter().map(|(_, _, s)| *s).collect();
    let source_label = multi_file_source_label(&sources);

    let mapped: Vec<(String, inspect::SafetensorsHeaderInfo)> = results
        .into_iter()
        .map(|(name, info, _source)| (name, info))
        .collect();

    if needs_aggregation {
        run_inspect_repo_aggregated(
            repo_id,
            &mapped,
            json,
            filter,
            dtypes,
            limit,
            tree,
            check_gpu,
            kv.as_ref(),
        )?;
        print_adapter_config_if_present(repo_id, revision, token.as_deref(), false, json);
        return Ok(());
    }

    if json {
        print_multi_file_json(&mapped, filter)?;
        print_adapter_config_if_present(repo_id, revision, token.as_deref(), false, true);
        return Ok(());
    }

    // BORROW: explicit .as_str() instead of Deref coercion
    print_multi_file_summary(repo_id, source_label.as_str(), &mapped, filter);
    print_adapter_config_if_present(repo_id, revision, token.as_deref(), false, false);
    Ok(())
}

/// Flattens tensors across every shard of a sharded repo and dispatches to
/// the existing `--dtypes` / `--tree` / per-tensor renderers.
///
/// Reused by both the cached and the network paths in [`run_inspect_repo`].
/// `--filter` is applied first (per-tensor name match), then `--limit`
/// truncates the resulting flat list. `total_*` counters reflect the
/// pre-filter, pre-limit totals so footers can show
/// `shown/total` ratios consistent with the single-file inspect.
// EXPLICIT: branches mirror `run_inspect_single`'s mode matrix
// (`tree+json`, `dtypes+json`, plain `--json`, `tree`, `dtypes`,
// per-tensor table). Splitting would obscure the parity.
#[allow(
    clippy::fn_params_excessive_bools,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
fn run_inspect_repo_aggregated(
    repo_id: &str,
    results: &[(String, inspect::SafetensorsHeaderInfo)],
    json: bool,
    filter: Option<&str>,
    dtypes: bool,
    limit: Option<usize>,
    tree: bool,
    check_gpu: Option<u32>,
    kv: Option<&gpu_check::KvComputed>,
) -> Result<(), FetchError> {
    // Flatten tensors across all shards. `(filename, &TensorInfo)` lets the
    // table renderer attribute each tensor to its source shard.
    let mut flat: Vec<(&str, &inspect::TensorInfo)> = Vec::new();
    for (file, info) in results {
        for t in &info.tensors {
            // BORROW: explicit .as_str() for owned String → &str
            flat.push((file.as_str(), t));
        }
    }

    let total_tensor_count = flat.len();
    let total_params: u64 = flat
        .iter()
        .map(|(_, t)| t.num_elements())
        .fold(0u64, u64::saturating_add);

    // Capture `--check-gpu` inputs BEFORE the filter / limit pass so the
    // verdict reflects the whole model. `gpu_check::{sum_tensor_bytes,
    // dominant_dtype_label}` take `&[TensorInfo]`, so we materialize an
    // owned vec once from the unfiltered `flat` slice; the heavier
    // post-filter table path below builds its own owned vec separately,
    // which we accept rather than deduplicate because `--check-gpu` is
    // opt-in and only costs a single linear clone when it is set.
    let gpu_inputs = check_gpu.map(|idx| {
        let owned_unfiltered: Vec<inspect::TensorInfo> =
            flat.iter().map(|(_, t)| (*t).clone()).collect();
        GpuCheckInputs {
            device_index: idx,
            weight_bytes: gpu_check::sum_tensor_bytes(&owned_unfiltered),
            dtype_label: gpu_check::dominant_dtype_label(&owned_unfiltered),
            total_params,
            kv: kv.cloned(),
        }
    });

    if let Some(pattern) = filter {
        // BORROW: explicit .as_str() instead of Deref coercion
        flat.retain(|(_, t)| matches_filter(t.name.as_str(), pattern));
    }

    let matched_count = flat.len();
    let matched_params: u64 = flat
        .iter()
        .map(|(_, t)| t.num_elements())
        .fold(0u64, u64::saturating_add);

    let truncated_by_limit = limit.is_some_and(|n| matched_count > n);
    if let Some(n) = limit {
        flat.truncate(n);
    }

    // BORROW: explicit .clone() to materialize owned TensorInfo for
    // renderers that take `&[TensorInfo]` rather than `&[&TensorInfo]`.
    let tensors_owned: Vec<inspect::TensorInfo> = flat.iter().map(|(_, t)| (*t).clone()).collect();

    let gpu_result = gpu_inputs
        .as_ref()
        .map(|i| gpu_check::query_gpu(i.device_index));
    let gpu_check_value = gpu_inputs.as_ref().zip(gpu_result.as_ref()).map(|(i, r)| {
        gpu_check::gpu_check_json(
            r,
            i.weight_bytes,
            i.dtype_label.as_str(),
            i.total_params,
            i.kv.as_ref(),
        )
    });

    // `--tree --json`: tagged-enum tree schema (matches run_inspect_single).
    if tree && json {
        return print_tree_json(
            repo_id,
            "<all shards>",
            &tensors_owned,
            total_tensor_count,
            total_params,
            gpu_check_value,
        );
    }

    // `--dtypes --json`: compact dtype breakdown.
    if dtypes && json {
        return print_dtype_summary_json(
            &tensors_owned,
            total_tensor_count,
            total_params,
            gpu_check_value,
        );
    }

    // Plain `--json` at the repo level on the aggregation path (forced by
    // `--check-gpu` when neither `--tree` nor `--dtypes` is set). Emits a
    // wrapped object so the verdict can ride along — distinct from the
    // unwrapped `Vec<(name, info)>` schema used by the non-aggregation
    // fast-path. The schema variation is gated by `--check-gpu`: without
    // it, `run_inspect_repo` takes the non-aggregation path and returns
    // the historical array.
    if json {
        return print_multi_file_json_with_gpu_check(results, filter, gpu_check_value);
    }

    // Header — same shape as the per-file inspect.
    println!("  Repo:   {repo_id}");
    let n_shards = results.len();
    let shard_label = if n_shards == 1 { "shard" } else { "shards" };
    println!("  Source: aggregated across {n_shards} {shard_label}");

    // `--tree`: hierarchical view, aggregated across shards.
    if tree {
        print_tree_summary(&tensors_owned, filter, total_tensor_count, total_params);
        maybe_print_gpu_check(gpu_inputs.as_ref(), gpu_result.as_ref());
        return Ok(());
    }

    // `--dtypes`: per-dtype histogram, aggregated across shards.
    if dtypes {
        print_dtype_summary(&tensors_owned, filter, total_tensor_count, total_params);
        maybe_print_gpu_check(gpu_inputs.as_ref(), gpu_result.as_ref());
        return Ok(());
    }

    // Bare `--limit` (and any combination of `--filter` + `--limit`):
    // flat tensor table with a `Shard` column so the user knows which
    // file each tensor came from.
    print_multi_shard_table(
        &flat,
        filter,
        limit,
        truncated_by_limit,
        total_tensor_count,
        matched_count,
        total_params,
        matched_params,
    );

    maybe_print_gpu_check(gpu_inputs.as_ref(), gpu_result.as_ref());
    Ok(())
}

/// Prints a flat tensor table for multi-shard inspect when `--limit` is set
/// (and neither `--dtypes` nor `--tree` was selected). Adds a `Shard` column
/// vs. the single-file table so the user can see provenance.
#[allow(clippy::too_many_arguments)]
fn print_multi_shard_table(
    flat: &[(&str, &inspect::TensorInfo)],
    filter: Option<&str>,
    limit: Option<usize>,
    truncated_by_limit: bool,
    total_tensor_count: usize,
    matched_count: usize,
    total_params: u64,
    matched_params: u64,
) {
    // Dynamic column widths.
    let nw = flat
        .iter()
        .map(|(_, t)| t.name.len())
        .max()
        .unwrap_or(6)
        .max(6); // BORROW: "Tensor".len()
    let shape_strs: Vec<String> = flat.iter().map(|(_, t)| format!("{:?}", t.shape)).collect();
    let sw = shape_strs.iter().map(String::len).max().unwrap_or(5).max(5); // BORROW: "Shape".len()
    let fw = flat
        .iter()
        .map(|(file, _)| file.len())
        .max()
        .unwrap_or(5)
        .max(5); // BORROW: "Shard".len()
    let row_width = nw + 2 + 8 + 2 + sw + 2 + 10 + 2 + 10 + 2 + fw;

    println!();
    println!(
        "  {:<nw$} {:<8} {:<sw$} {:>10} {:>10}  {:<fw$}",
        "Tensor", "Dtype", "Shape", "Size", "Params", "Shard",
    );

    for ((file, t), shape_str) in flat.iter().zip(shape_strs.iter()) {
        let size_str = format_size(t.byte_len());
        let params_str = inspect::format_params(t.num_elements());
        println!(
            "  {:<nw$} {:<8} {:<sw$} {:>10} {:>10}  {:<fw$}",
            t.name, t.dtype, shape_str, size_str, params_str, file,
        );
    }

    println!("  {}", "\u{2500}".repeat(row_width));

    let shown_count = flat.len();
    let shown_params: u64 = flat
        .iter()
        .map(|(_, t)| t.num_elements())
        .fold(0u64, u64::saturating_add);
    let tensor_label = if shown_count == 1 {
        "tensor"
    } else {
        "tensors"
    };

    match (filter.is_some(), truncated_by_limit) {
        (false, false) => {
            println!(
                "  {shown_count} {tensor_label}, {} params",
                inspect::format_params(shown_params)
            );
        }
        (true, false) => {
            println!(
                "  {shown_count}/{total_tensor_count} {tensor_label}, {}/{} params (filter: {:?})",
                inspect::format_params(shown_params),
                inspect::format_params(total_params),
                filter.unwrap_or_default(),
            );
        }
        (false, true) => {
            println!(
                "  {shown_count}/{total_tensor_count} {tensor_label} shown, {}/{} params (limit: {})",
                inspect::format_params(shown_params),
                inspect::format_params(total_params),
                limit.unwrap_or(0),
            );
        }
        (true, true) => {
            println!(
                "  {shown_count}/{matched_count}/{total_tensor_count} {tensor_label} shown, {}/{}/{} params (filter: {:?}, limit: {})",
                inspect::format_params(shown_params),
                inspect::format_params(matched_params),
                inspect::format_params(total_params),
                filter.unwrap_or_default(),
                limit.unwrap_or(0),
            );
        }
    }
}

/// Prints adapter configuration if `adapter_config.json` is found in the repository.
///
/// Silently returns if the file does not exist or cannot be fetched.
fn print_adapter_config_if_present(
    repo_id: &str,
    revision: Option<&str>,
    token: Option<&str>,
    cached: bool,
    json: bool,
) {
    let result = if cached {
        inspect::fetch_adapter_config_cached(repo_id, revision)
    } else {
        let Ok(rt) = tokio::runtime::Runtime::new() else {
            return;
        };
        rt.block_on(inspect::fetch_adapter_config(repo_id, token, revision))
    };

    let Ok(Some(config)) = result else { return };

    if json {
        if let Ok(output) = serde_json::to_string_pretty(&config) {
            println!("{output}");
        }
        return;
    }

    println!();
    println!("  Adapter config:");
    if let Some(ref peft_type) = config.peft_type {
        println!("    PEFT type:       {peft_type}");
    }
    if let Some(ref base) = config.base_model_name_or_path {
        println!("    Base model:      {base}");
    }
    if let Some(r) = config.r {
        println!("    Rank (r):        {r}");
    }
    if let Some(alpha) = config.lora_alpha {
        println!("    LoRA alpha:      {alpha}");
    }
    if let Some(ref task) = config.task_type {
        println!("    Task type:       {task}");
    }
    if !config.target_modules.is_empty() {
        println!("    Target modules:  {}", config.target_modules.join(", "));
    }
}

/// One row of a `--dtypes` summary.
#[derive(serde::Serialize)]
struct DtypeGroup<'a> {
    dtype: &'a str,
    tensors: usize,
    params: u64,
    bytes: u64,
}

/// JSON shape emitted by `inspect --dtypes --json`.
///
/// `total_tensors` and `total_params` always reflect the whole file, before any
/// filter. Summing the `dtypes` array gives the filtered totals.
#[derive(serde::Serialize)]
struct DtypeSummaryJson<'a> {
    dtypes: Vec<DtypeGroup<'a>>,
    total_tensors: usize,
    total_params: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_check: Option<serde_json::Value>,
}

/// Groups tensors by dtype and returns rows sorted by tensor count descending.
///
/// Each row is `(dtype, count, params, bytes)`. Generic over any iterator of
/// `&TensorInfo` so both slice callers (`inspect` paths) and HashMap-values
/// callers (`diff` paths) can use it without an intermediate `Vec`.
fn compute_dtype_groups<'a, I>(tensors: I) -> Vec<(&'a str, usize, u64, u64)>
where
    I: IntoIterator<Item = &'a inspect::TensorInfo>,
{
    let mut groups: HashMap<&str, (usize, u64, u64)> = HashMap::new();
    for t in tensors {
        let entry = groups
            .entry(t.dtype.as_str()) // BORROW: explicit .as_str()
            .or_insert((0, 0, 0));
        entry.0 += 1;
        entry.1 = entry.1.saturating_add(t.num_elements());
        entry.2 = entry.2.saturating_add(t.byte_len());
    }
    // BORROW: flatten nested HashMap tuple into (dtype, count, params, bytes)
    let mut rows: Vec<(&str, usize, u64, u64)> = groups
        .into_iter()
        .map(|(dtype, (count, params, bytes))| (dtype, count, params, bytes))
        .collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.1));
    rows
}

/// Emits the `--dtypes` summary as JSON.
fn print_dtype_summary_json(
    tensors: &[inspect::TensorInfo],
    total_tensor_count: usize,
    total_params: u64,
    gpu_check: Option<serde_json::Value>,
) -> Result<(), FetchError> {
    let rows = compute_dtype_groups(tensors);
    let output = DtypeSummaryJson {
        dtypes: rows
            .into_iter()
            .map(|(dtype, tensors, params, bytes)| DtypeGroup {
                dtype,
                tensors,
                params,
                bytes,
            })
            .collect(),
        total_tensors: total_tensor_count,
        total_params,
        gpu_check,
    };
    let serialized = serde_json::to_string_pretty(&output)
        .map_err(|e| FetchError::Http(format!("failed to serialize JSON: {e}")))?;
    println!("{serialized}");
    Ok(())
}

/// Prints a per-dtype summary table (tensor count, param count, byte size per dtype).
fn print_dtype_summary(
    tensors: &[inspect::TensorInfo],
    filter: Option<&str>,
    total_tensor_count: usize,
    total_params: u64,
) {
    let rows = compute_dtype_groups(tensors);

    // Dynamic column widths.
    let dw = rows
        .iter()
        .map(|(d, _, _, _)| d.len())
        .max()
        .unwrap_or(5)
        .max(5); // BORROW: "Dtype".len()
    let row_width = dw + 2 + 8 + 2 + 12 + 2 + 10;

    println!();
    println!(
        "  {:<dw$} {:>8} {:>12} {:>10}",
        "Dtype", "Tensors", "Params", "Size",
    );

    for (dtype, count, params, bytes) in &rows {
        println!(
            "  {:<dw$} {:>8} {:>12} {:>10}",
            dtype,
            count,
            inspect::format_params(*params),
            format_size(*bytes),
        );
    }

    println!("  {}", "\u{2500}".repeat(row_width));

    let filtered_count: usize = rows.iter().map(|(_, count, _, _)| count).sum();
    let filtered_params: u64 = rows.iter().map(|(_, _, params, _)| params).sum();
    let tensor_label = if filtered_count == 1 {
        "tensor"
    } else {
        "tensors"
    };

    if filter.is_some() {
        println!(
            "  {filtered_count}/{total_tensor_count} {tensor_label}, {}/{} params",
            inspect::format_params(filtered_params),
            inspect::format_params(total_params),
        );
    } else {
        println!(
            "  {filtered_count} {tensor_label}, {} params",
            inspect::format_params(filtered_params),
        );
    }
}

/// Prints the shard-index rollup: tensor counts per shard, plus the matched
/// tensor names nested under each shard when a `filter` is active (the names
/// come free from the index's `weight_map` — no header reads).
fn print_shard_index_summary(repo_id: &str, index: &inspect::ShardedIndex, filter: Option<&str>) {
    println!("  Repo:   {repo_id}");
    println!("  Source: shard index (model.safetensors.index.json)");
    println!();

    // Count tensors per shard, optionally filtering by tensor name. When a
    // filter is active we also collect the matched names per shard: they are
    // free here (the shard index maps every name to its shard with no header
    // reads), so the rollup can list them instead of hiding them behind a
    // count (v0.10.6 Symptom 1).
    let total_tensors = index.weight_map.len();
    let mut by_shard: HashMap<&str, usize> = HashMap::new();
    let mut names_by_shard: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut filtered_total: usize = 0;
    for (tensor_name, shard_name) in &index.weight_map {
        if let Some(pattern) = filter {
            if !matches_filter(tensor_name, pattern) {
                continue;
            }
            // BORROW: explicit .as_str() for &String → &str map key + value
            names_by_shard
                .entry(shard_name.as_str())
                .or_default()
                .push(tensor_name.as_str());
        }
        // BORROW: explicit .as_str() for &String → &str map key
        *by_shard.entry(shard_name.as_str()).or_default() += 1;
        filtered_total += 1;
    }

    let fw = index
        .shards
        .iter()
        .map(String::len)
        .max()
        .unwrap_or(4)
        .max(4); // BORROW: "File".len()
    let row_width = fw + 2 + 8;
    println!("  {:<fw$} {:>8}", "File", "Tensors");

    for shard in &index.shards {
        // BORROW: explicit .as_str() for &String → &str map lookup
        let count = by_shard.get(shard.as_str()).copied().unwrap_or(0);
        if filter.is_some() && count == 0 {
            continue;
        }
        println!("  {shard:<fw$} {count:>8}");
        // Filtered: list this shard's matched tensor names beneath its row.
        if filter.is_some() {
            if let Some(names) = names_by_shard.get(shard.as_str()) {
                let mut sorted = names.clone();
                sorted.sort_unstable();
                for tname in sorted {
                    println!("    {tname}");
                }
            }
        }
    }

    println!("  {}", "\u{2500}".repeat(row_width));

    let displayed_shards = if filter.is_some() {
        by_shard.len()
    } else {
        index.shards.len()
    };
    let shard_label = if displayed_shards == 1 {
        "shard"
    } else {
        "shards"
    };
    let tensor_label = if filtered_total == 1 {
        "tensor"
    } else {
        "tensors"
    };

    if filter.is_some() {
        println!(
            "  {displayed_shards} {shard_label}, {filtered_total}/{total_tensors} {tensor_label} (filter: {:?})",
            filter.unwrap_or_default(),
        );
        println!(
            "  Hint: names shown above \u{2014} for shapes/dtypes/sizes run \
             `hf-fm inspect {repo_id} <filename> --tree` (or `--dtypes`), or add \
             `--limit N` to cap a broad match."
        );
    } else {
        println!("  {displayed_shards} {shard_label}, {filtered_total} {tensor_label}");
        println!(
            "  Hint: this rollup hides tensor names \u{2014} run \
             `hf-fm inspect {repo_id} <filename> --tree` (or `--dtypes`) for per-tensor detail."
        );
    }
}

/// Prints multi-file inspection results as JSON, optionally filtering tensors.
fn print_multi_file_json(
    results: &[(String, inspect::SafetensorsHeaderInfo)],
    filter: Option<&str>,
) -> Result<(), FetchError> {
    if let Some(pattern) = filter {
        // Filter tensors before cloning to avoid O(T) clone-then-discard.
        let filtered: Vec<(String, inspect::SafetensorsHeaderInfo)> = results
            .iter()
            .filter_map(|(name, info)| {
                let matching: Vec<inspect::TensorInfo> = info
                    .tensors
                    .iter()
                    .filter(|t| matches_filter(t.name.as_str(), pattern)) // BORROW: explicit .as_str()
                    .cloned()
                    .collect();
                if matching.is_empty() {
                    return None;
                }
                Some((
                    name.clone(), // BORROW: explicit .clone() for owned String
                    // `SafetensorsHeaderInfo` is `#[non_exhaustive]` since
                    // v0.10.3 — use the public `new()` constructor instead
                    // of struct-literal syntax. Quant info is preserved
                    // across the filter (it summarises the whole file,
                    // independent of which tensor subset we're rendering).
                    inspect::SafetensorsHeaderInfo::new(
                        matching,
                        info.metadata.clone(),
                        info.header_size,
                        info.file_size,
                        info.quant_info.clone(),
                    ),
                ))
            })
            .collect();
        let output = serde_json::to_string_pretty(&filtered)
            .map_err(|e| FetchError::Http(format!("failed to serialize JSON: {e}")))?;
        println!("{output}");
    } else {
        let output = serde_json::to_string_pretty(results)
            .map_err(|e| FetchError::Http(format!("failed to serialize JSON: {e}")))?;
        println!("{output}");
    }
    Ok(())
}

/// Same as [`print_multi_file_json`] but wraps the file array in a top-level
/// object so a `gpu_check` field can ride along.
///
/// Schema:
///
/// ```jsonc
/// {
///   "files": [["filename", { /* SafetensorsHeaderInfo */ }], ...],
///   "gpu_check": { /* see gpu_check::gpu_check_json */ }  // omitted when --check-gpu absent
/// }
/// ```
///
/// Only reached from [`run_inspect_repo_aggregated`] when `--check-gpu` was
/// what forced the aggregation path; without `--check-gpu` the unwrapped
/// `Vec` schema from [`print_multi_file_json`] is preserved for backwards
/// compatibility.
fn print_multi_file_json_with_gpu_check(
    results: &[(String, inspect::SafetensorsHeaderInfo)],
    filter: Option<&str>,
    gpu_check: Option<serde_json::Value>,
) -> Result<(), FetchError> {
    let files_payload: Vec<(String, inspect::SafetensorsHeaderInfo)> = if let Some(pattern) = filter
    {
        results
            .iter()
            .filter_map(|(name, info)| {
                let matching: Vec<inspect::TensorInfo> = info
                    .tensors
                    .iter()
                    // BORROW: explicit .as_str() for &String → &str
                    .filter(|t| matches_filter(t.name.as_str(), pattern))
                    .cloned()
                    .collect();
                if matching.is_empty() {
                    return None;
                }
                Some((
                    name.clone(),
                    // `SafetensorsHeaderInfo` is `#[non_exhaustive]` since
                    // v0.10.3 — use the public `new()` constructor.
                    inspect::SafetensorsHeaderInfo::new(
                        matching,
                        info.metadata.clone(),
                        info.header_size,
                        info.file_size,
                        info.quant_info.clone(),
                    ),
                ))
            })
            .collect()
    } else {
        results.to_vec()
    };

    let mut top = serde_json::Map::new();
    top.insert(
        "files".to_owned(),
        serde_json::to_value(&files_payload)
            .map_err(|e| FetchError::Http(format!("failed to serialize JSON: {e}")))?,
    );
    if let Some(gc) = gpu_check {
        top.insert("gpu_check".to_owned(), gc);
    }

    let output = serde_json::to_string_pretty(&serde_json::Value::Object(top))
        .map_err(|e| FetchError::Http(format!("failed to serialize JSON: {e}")))?;
    println!("{output}");
    Ok(())
}

/// Computes the repo-level `Source:` label for the multi-file inspect summary
/// from the per-file [`inspect::InspectSource`] values.
///
/// Returns `"cached"` when every file was read from the local cache, `"remote"`
/// when every file was fetched via HTTP Range, and `"mixed (N cached, M remote)"`
/// for a genuine split — the network inspect path resolves each file
/// cache-first (range request only on a miss), so a partially-cached repo
/// legitimately mixes the two. An empty slice yields `"unknown"`: there are no
/// files to classify.
fn multi_file_source_label(sources: &[inspect::InspectSource]) -> String {
    let cached = sources
        .iter()
        .filter(|s| matches!(s, inspect::InspectSource::Cached))
        .count();
    let remote = sources
        .iter()
        .filter(|s| matches!(s, inspect::InspectSource::Remote))
        .count();
    match (cached, remote) {
        (0, 0) => "unknown".to_owned(),
        (_, 0) => "cached".to_owned(),
        (0, _) => "remote".to_owned(),
        (c, r) => format!("mixed ({c} cached, {r} remote)"),
    }
}

/// Prints the multi-file rollup: per-file tensor counts and params, plus the
/// matched tensor names nested under each file when a `filter` is active.
fn print_multi_file_summary(
    repo_id: &str,
    source: &str,
    results: &[(String, inspect::SafetensorsHeaderInfo)],
    filter: Option<&str>,
) {
    println!("  Repo:   {repo_id}");
    println!("  Source: {source}");
    println!();

    let fw = results
        .iter()
        .map(|(name, _)| name.len())
        .max()
        .unwrap_or(4)
        .max(4); // BORROW: "File".len()
    let row_width = fw + 2 + 8 + 1 + 12;
    println!("  {:<fw$} {:>8} {:>12}", "File", "Tensors", "Params");

    let mut total_tensors_unfiltered: usize = 0;
    let mut total_params_unfiltered: u64 = 0;
    let mut total_tensors_filtered: usize = 0;
    let mut total_params_filtered: u64 = 0;
    let mut files_with_matches: usize = 0;

    for (name, info) in results {
        total_tensors_unfiltered = total_tensors_unfiltered.saturating_add(info.tensors.len());
        total_params_unfiltered = total_params_unfiltered.saturating_add(info.total_params());

        let (tensor_count, params, matched_names) = if let Some(pattern) = filter {
            let matching: Vec<&inspect::TensorInfo> = info
                .tensors
                .iter()
                // BORROW: explicit .as_str() instead of Deref coercion
                .filter(|t| matches_filter(t.name.as_str(), pattern))
                .collect();
            let p: u64 = matching.iter().map(|t| t.num_elements()).sum();
            // BORROW: explicit .as_str() for &String → &str
            let mut names: Vec<&str> = matching.iter().map(|t| t.name.as_str()).collect();
            names.sort_unstable();
            (matching.len(), p, names)
        } else {
            (info.tensors.len(), info.total_params(), Vec::new())
        };

        if filter.is_some() && tensor_count == 0 {
            continue;
        }

        files_with_matches += 1;
        total_tensors_filtered = total_tensors_filtered.saturating_add(tensor_count);
        total_params_filtered = total_params_filtered.saturating_add(params);
        println!(
            "  {name:<fw$} {tensor_count:>8} {:>12}",
            inspect::format_params(params)
        );
        // Filtered: list this file's matched tensor names beneath its row
        // (v0.10.6 Symptom 1). Names only — shapes/dtypes stay in the per-file
        // form the Hint points at, for parity with the shard-index path.
        for tname in &matched_names {
            println!("    {tname}");
        }
    }

    println!("  {}", "\u{2500}".repeat(row_width));
    let file_label = if files_with_matches == 1 {
        "file"
    } else {
        "files"
    };
    let tensor_label = if total_tensors_filtered == 1 {
        "tensor"
    } else {
        "tensors"
    };

    if filter.is_some() {
        println!(
            "  {} {file_label}, {total_tensors_filtered}/{total_tensors_unfiltered} {tensor_label}, {}/{} params (filter: {:?})",
            files_with_matches,
            inspect::format_params(total_params_filtered),
            inspect::format_params(total_params_unfiltered),
            filter.unwrap_or_default(),
        );
    } else {
        println!(
            "  {} {file_label}, {total_tensors_filtered} {tensor_label}, {} params",
            files_with_matches,
            inspect::format_params(total_params_filtered)
        );
    }

    // Discoverability nudge. When filtered, the matched names are listed
    // above, so the hint pivots to the shapes/dtypes the rollup still omits
    // (and to `--limit` for capping a broad match). When unfiltered, names are
    // hidden behind counts; nudge the common single-`.safetensors` repo (least
    // informative — one row, a param count, nothing else) toward the per-tensor
    // views. Multi-file unfiltered rollups keep their per-file breakdown as the
    // useful signal, so they get no hint.
    if filter.is_some() {
        println!(
            "  Hint: names shown above \u{2014} for shapes/dtypes/sizes run \
             `hf-fm inspect {repo_id} <filename> --tree` (or `--dtypes`), or add \
             `--limit N` to cap a broad match."
        );
    } else if let [(name, _)] = results {
        println!(
            "  Hint: this rollup hides tensor names \u{2014} run \
             `hf-fm inspect {repo_id} {name} --tree` (or `--dtypes`) for per-tensor detail."
        );
    }
}

#[allow(clippy::too_many_lines)]
fn run_status(
    repo_id: &str,
    revision: Option<&str>,
    token: Option<&str>,
    preset: Option<&Preset>,
    json: bool,
) -> Result<(), FetchError> {
    // BORROW: explicit String::from (equivalent to .to_owned()) for Option<&str> → Option<String>
    let token = token
        .map(String::from)
        .or_else(|| std::env::var("HF_TOKEN").ok());

    let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
        path: PathBuf::from("<runtime>"),
        source: e,
    })?;

    // Resolve the effective preset: CLI override → sidecar → none.
    // The sidecar lookup needs the repo cache dir; reuse the same helper
    // `cache::repo_status` uses internally so paths stay in lockstep.
    let cache_root = cache::hf_cache_dir()?;
    let repo_dir = hf_fetch_model::cache_layout::repo_dir(&cache_root, repo_id);
    let sidecar = cache::read_snapshot(&repo_dir)?;
    let effective_preset_name: Option<String> = match preset {
        Some(p) => Some(preset_name(p).to_owned()), // BORROW: explicit .to_owned()
        None => sidecar.and_then(|s| s.preset),
    };
    let preset_glob_list: Option<&'static [&'static str]> = effective_preset_name
        .as_deref() // BORROW: explicit .as_deref() for Option<String> → Option<&str>
        .and_then(hf_fetch_model::config::preset_globs);

    // BORROW: explicit .as_deref() for Option<String> → Option<&str>
    let status = rt.block_on(cache::repo_status(
        repo_id,
        token.as_deref(),
        revision,
        preset_glob_list,
    ))?;

    if json {
        return print_status_json(&status, revision.unwrap_or("main"));
    }

    // Header
    let rev_display = revision.unwrap_or("main");
    match &status.commit_hash {
        Some(hash) => println!("{repo_id} ({rev_display} @ {hash})"),
        None => println!("{repo_id} ({rev_display}, not yet cached)"),
    }
    println!("Cache: {}\n", status.cache_path.display());

    if status.files.is_empty() {
        println!("  (no files found in remote repository)");
        return Ok(());
    }

    // File table
    let fw = status
        .files
        .iter()
        .map(|(name, _)| name.len())
        .max()
        .unwrap_or(4)
        .max(4); // BORROW: "File".len()
    for (filename, file_status) in &status.files {
        match file_status {
            cache::FileStatus::Complete { local_size } => {
                println!(
                    "  {:<fw$} {:>10}  complete",
                    filename,
                    format_size(*local_size)
                );
            }
            cache::FileStatus::Partial {
                local_size,
                expected_size,
            } => {
                println!(
                    "  {:<fw$} {:>10} / {:<10}  PARTIAL",
                    filename,
                    format_size(*local_size),
                    format_size(*expected_size)
                );
            }
            cache::FileStatus::Missing { expected_size } => {
                if *expected_size > 0 {
                    println!(
                        "  {:<fw$} {:>10}  MISSING",
                        filename,
                        format_size(*expected_size)
                    );
                } else {
                    println!("  {filename:<fw$} {:>10}  MISSING", "\u{2014}");
                }
            }
            cache::FileStatus::Excluded { expected_size } => {
                if *expected_size > 0 {
                    println!(
                        "  {:<fw$} {:>10}  excluded",
                        filename,
                        format_size(*expected_size)
                    );
                } else {
                    println!("  {filename:<fw$} {:>10}  excluded", "\u{2014}");
                }
            }
            // EXPLICIT: future FileStatus variants display as UNKNOWN
            _ => {
                println!("  {filename:<fw$}              UNKNOWN");
            }
        }
    }

    // Summary
    let total = status.files.len();
    let complete = status.complete_count();
    let partial = status.partial_count();
    let missing = status.missing_count();
    let excluded = status.excluded_count();
    println!();
    if excluded > 0 {
        println!(
            "{complete}/{total} complete, {partial} partial, {missing} missing, {excluded} excluded"
        );
    } else {
        println!("{complete}/{total} complete, {partial} partial, {missing} missing");
    }

    Ok(())
}

/// Per-repo summary entry for `status --json` (all repos).
#[derive(serde::Serialize)]
struct StatusRepoSummaryJson {
    /// Repository identifier.
    repo_id: String,
    /// Number of files in the snapshot directory.
    file_count: usize,
    /// Total size on disk in bytes.
    size: u64,
    /// Whether the repo has incomplete `.chunked.part` downloads.
    has_partial: bool,
}

/// Top-level shape for `status --json` (all repos).
#[derive(serde::Serialize)]
struct StatusAllJson {
    /// HF cache directory.
    cache_dir: String,
    /// Per-repo summaries, in cache-scan order.
    repos: Vec<StatusRepoSummaryJson>,
    /// Number of cached models.
    model_count: usize,
}

/// Per-file entry for `status <repo> --json`.
#[derive(serde::Serialize)]
struct StatusFileJson {
    /// Repo-relative filename.
    filename: String,
    /// One of `complete` / `partial` / `missing` / `excluded` (or `unknown`).
    state: &'static str,
    /// Local size in bytes; present for `complete` / `partial`.
    #[serde(skip_serializing_if = "Option::is_none")]
    local_size: Option<u64>,
    /// Expected size in bytes; present for `partial` / `missing` / `excluded`.
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_size: Option<u64>,
}

/// Category counts for `status <repo> --json`.
#[derive(serde::Serialize)]
struct StatusSummaryJson {
    /// Total files reported.
    total: usize,
    /// Files fully present.
    complete: usize,
    /// Files present but smaller than expected.
    partial: usize,
    /// Files absent from the local snapshot.
    missing: usize,
    /// Files deliberately excluded by the active preset / filter.
    excluded: usize,
}

/// Top-level shape for `status <repo> --json` (single repo).
#[derive(serde::Serialize)]
struct StatusRepoJson {
    /// Repository identifier.
    repo_id: String,
    /// Requested revision (branch / tag / SHA, default `main`).
    revision: String,
    /// Resolved commit hash, if the repo is cached.
    #[serde(skip_serializing_if = "Option::is_none")]
    commit_hash: Option<String>,
    /// The repo's cache snapshot path.
    cache_path: String,
    /// Per-file status entries, sorted by filename.
    files: Vec<StatusFileJson>,
    /// Category counts.
    summary: StatusSummaryJson,
}

/// Prints the `status` all-repos summary as JSON.
fn print_status_all_json(
    summaries: &[cache::CachedModelSummary],
    cache_dir: &std::path::Path,
) -> Result<(), FetchError> {
    let repos: Vec<StatusRepoSummaryJson> = summaries
        .iter()
        .map(|s| StatusRepoSummaryJson {
            // BORROW: explicit .clone() for owned String field
            repo_id: s.repo_id.clone(),
            file_count: s.file_count,
            size: s.total_size,
            has_partial: s.has_partial,
        })
        .collect();
    let result = StatusAllJson {
        // BORROW: explicit .to_string() for Path → String
        cache_dir: cache_dir.display().to_string(),
        model_count: repos.len(),
        repos,
    };
    emit_json(&result)
}

/// Maps a [`FileStatus`](cache::FileStatus) to its JSON
/// `(state, local_size, expected_size)` tuple.
fn file_status_json_fields(status: &cache::FileStatus) -> (&'static str, Option<u64>, Option<u64>) {
    match status {
        cache::FileStatus::Complete { local_size } => ("complete", Some(*local_size), None),
        cache::FileStatus::Partial {
            local_size,
            expected_size,
        } => ("partial", Some(*local_size), Some(*expected_size)),
        cache::FileStatus::Missing { expected_size } => ("missing", None, Some(*expected_size)),
        cache::FileStatus::Excluded { expected_size } => ("excluded", None, Some(*expected_size)),
        // EXPLICIT: FileStatus is #[non_exhaustive]; future variants serialize as "unknown".
        _ => ("unknown", None, None),
    }
}

/// Prints a single repo's cache status as JSON.
fn print_status_json(status: &cache::RepoStatus, revision: &str) -> Result<(), FetchError> {
    let files: Vec<StatusFileJson> = status
        .files
        .iter()
        .map(|(filename, fs)| {
            let (state, local_size, expected_size) = file_status_json_fields(fs);
            StatusFileJson {
                // BORROW: explicit .clone() for owned String field
                filename: filename.clone(),
                state,
                local_size,
                expected_size,
            }
        })
        .collect();
    let summary = StatusSummaryJson {
        total: status.files.len(),
        complete: status.complete_count(),
        partial: status.partial_count(),
        missing: status.missing_count(),
        excluded: status.excluded_count(),
    };
    let result = StatusRepoJson {
        // BORROW: explicit .clone()/.to_owned()/.to_string() for owned fields
        repo_id: status.repo_id.clone(),
        revision: revision.to_owned(),
        commit_hash: status.commit_hash.clone(),
        cache_path: status.cache_path.display().to_string(),
        files,
        summary,
    };
    emit_json(&result)
}

/// Cache state of a listed file relative to the local snapshot.
///
/// Shared by the `list-files` human table (rendered as a glyph) and `--json`
/// output (rendered as a word), so the two never disagree on how a file's cache
/// status is classified.
// EXHAUSTIVE: internal list-files state enum; crate owns and matches all variants
#[derive(Clone, Copy)]
enum FileCacheState {
    /// Local file present and at least the expected size.
    Complete,
    /// Local file present but smaller than the expected size.
    Partial,
    /// No local file present in the snapshot.
    Missing,
}

impl FileCacheState {
    /// Marker shown in the human table's Cached column (check mark for
    /// complete, the word `partial`, cross mark for missing).
    const fn glyph(self) -> &'static str {
        match self {
            Self::Complete => "\u{2713}",
            Self::Partial => "partial",
            Self::Missing => "\u{2717}",
        }
    }

    /// Machine-readable word for `--json` (`complete` / `partial` / `missing`).
    const fn word(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::Partial => "partial",
            Self::Missing => "missing",
        }
    }
}

/// Lists files in a remote `HuggingFace` repository without downloading.
// EXPLICIT: composes filter compilation, file enumeration, optional checksum
// fetch, optional cache cross-reference, and table formatting. Sequential
// pipeline; splitting hides the listing flow.
#[allow(
    clippy::too_many_arguments,
    clippy::fn_params_excessive_bools,
    clippy::too_many_lines
)]
fn run_list_files(
    repo_id: &str,
    revision: Option<&str>,
    token: Option<&str>,
    filter_patterns: &[String],
    exclude_patterns: &[String],
    preset: Option<&Preset>,
    no_checksum: bool,
    show_cached: bool,
    json: bool,
) -> Result<(), FetchError> {
    if !repo_id.contains('/') {
        return Err(FetchError::InvalidArgument(format!(
            "invalid REPO_ID \"{repo_id}\": expected \"org/model\" format \
             (e.g., \"google/gemma-2-2b-it\")"
        )));
    }

    // Build glob filters from preset + explicit patterns.
    let mut include_patterns: Vec<String> = match preset {
        Some(&Preset::Safetensors) => vec![
            "*.safetensors".to_owned(),
            "*.json".to_owned(),
            "*.txt".to_owned(),
        ],
        Some(&Preset::Gguf) => vec!["*.gguf".to_owned(), "*.json".to_owned(), "*.txt".to_owned()],
        Some(&Preset::Npz) => vec![
            "*.npz".to_owned(),
            "*.npy".to_owned(),
            "config.yaml".to_owned(),
            "*.json".to_owned(),
            "*.txt".to_owned(),
        ],
        Some(&Preset::Pth) => vec![
            "pytorch_model*.bin".to_owned(),
            "*.json".to_owned(),
            "*.txt".to_owned(),
        ],
        Some(&Preset::ConfigOnly) => {
            vec!["*.json".to_owned(), "*.txt".to_owned(), "*.md".to_owned()]
        }
        None => Vec::new(),
    };
    for p in filter_patterns {
        // BORROW: explicit .clone() for owned String
        include_patterns.push(p.clone());
    }
    let include = compile_glob_patterns(&include_patterns)?;
    let exclude = compile_glob_patterns(exclude_patterns)?;

    // Resolve token from arg or env.
    // BORROW: explicit .to_owned() for Option<&str> → Option<String>
    let resolved_token = token
        .map(ToOwned::to_owned)
        .or_else(|| std::env::var("HF_TOKEN").ok());

    // Fetch remote file list with metadata.
    let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
        path: PathBuf::from("<runtime>"),
        source: e,
    })?;
    let client = hf_fetch_model::build_client(resolved_token.as_deref())?;
    let files = rt.block_on(repo::list_repo_files_with_metadata(
        repo_id,
        resolved_token.as_deref(),
        revision,
        &client,
    ))?;

    // Apply glob filters.
    let filtered: Vec<_> = files
        .into_iter()
        .filter(|f| {
            // BORROW: explicit .as_str() instead of Deref coercion
            file_matches(f.filename.as_str(), include.as_ref(), exclude.as_ref())
        })
        .collect();

    // Resolve cache state if requested.
    // Three states: "✓" (complete), "partial" (local < expected), "✗" (missing).
    // Uses the same size-comparison logic as `status` (cache.rs).
    let cache_marks: Vec<FileCacheState> = if show_cached {
        let cache_dir = cache::hf_cache_dir()?;
        let repo_dir = hf_fetch_model::cache_layout::repo_dir(&cache_dir, repo_id);
        let revision_str = revision.unwrap_or("main");
        let commit_hash = cache::read_ref(&repo_dir, revision_str);
        let snapshot_dir =
            commit_hash.map(|h| hf_fetch_model::cache_layout::snapshot_dir(&repo_dir, &h));

        filtered
            .iter()
            .map(|f| {
                let local_path = snapshot_dir
                    .as_ref()
                    // BORROW: explicit .as_str() instead of Deref coercion
                    .map(|dir| dir.join(f.filename.as_str()));
                match local_path {
                    Some(ref path) if path.exists() => {
                        let local_size = std::fs::metadata(path).map_or(0, |m| m.len());
                        let expected = f.size.unwrap_or(0);
                        if expected > 0 && local_size < expected {
                            FileCacheState::Partial
                        } else {
                            FileCacheState::Complete
                        }
                    }
                    _ => FileCacheState::Missing,
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    // JSON output mode: serialize the filtered list (and cache state, when
    // requested) instead of printing the human table.
    if json {
        return print_list_files_json(repo_id, &filtered, &cache_marks, show_cached);
    }

    // Compute file-name column width from the actual data.
    let fw = filtered
        .iter()
        .map(|f| f.filename.len())
        .max()
        .unwrap_or(4)
        .max(4); // BORROW: "File".len()

    // Print table header.
    if no_checksum {
        if show_cached {
            println!("  {:<fw$} {:>10}  Cached", "File", "Size");
            println!("  {:<fw$} {:>10}  {:-<6}", "", "", "");
        } else {
            println!("  {:<fw$} {:>10}", "File", "Size");
            println!("  {:<fw$} {:>10}", "", "");
        }
    } else if show_cached {
        println!("  {:<fw$} {:>10}  {:<12}  Cached", "File", "Size", "SHA256");
        println!("  {:<fw$} {:>10}  {:<12}  {:-<6}", "", "", "", "");
    } else {
        println!("  {:<fw$} {:>10}  {:<12}", "File", "Size", "SHA256");
        println!("  {:<fw$} {:>10}  {:<12}", "", "", "");
    }

    // Print each file row.
    let mut total_bytes: u64 = 0;
    let mut cached_count: usize = 0;
    let mut any_no_sha = false;

    for (i, f) in filtered.iter().enumerate() {
        let size = f.size.unwrap_or(0);
        total_bytes = total_bytes.saturating_add(size);

        let size_str = format_size(size);
        let sha_str = if no_checksum {
            String::new()
        } else if let Some(hash) = f.sha256.as_deref().and_then(|s| s.get(..12)) {
            hash.to_owned() // BORROW: &str → String for column display
        } else {
            any_no_sha = true;
            "\u{2014}".to_owned()
        };

        if show_cached {
            let state = cache_marks
                .get(i)
                .copied()
                .unwrap_or(FileCacheState::Missing);
            if matches!(state, FileCacheState::Complete) {
                cached_count += 1;
            }
            let mark = state.glyph();
            if no_checksum {
                println!("  {:<fw$} {:>10}  {mark}", f.filename, size_str);
            } else {
                println!(
                    "  {:<fw$} {:>10}  {:<12}  {mark}",
                    f.filename, size_str, sha_str
                );
            }
        } else if no_checksum {
            println!("  {:<fw$} {:>10}", f.filename, size_str);
        } else {
            println!("  {:<fw$} {:>10}  {sha_str}", f.filename, size_str);
        }
    }

    // Summary line.
    let count = filtered.len();
    let file_label = pluralize(count, "file", "files");
    let row_width = fw + 2 + 10 + 2 + 12;
    println!("  {:\u{2500}<row_width$}", "");
    if show_cached {
        println!(
            "  {count} {file_label}, {} total ({cached_count} cached)",
            format_size(total_bytes)
        );
    } else {
        println!("  {count} {file_label}, {} total", format_size(total_bytes));
    }
    if any_no_sha && !no_checksum {
        println!("  \u{2014} = not an LFS file (no SHA256 tracked by the Hub)");
    }

    Ok(())
}

/// Serializable per-file entry for `list-files --json`.
#[derive(serde::Serialize)]
struct ListFileEntry {
    /// Repo-relative filename.
    filename: String,
    /// File size in bytes (`0` when the Hub reports no size).
    size: u64,
    /// Full SHA256 hex digest, or `null` for non-LFS files. Always emitted,
    /// independent of `--no-checksum`.
    sha256: Option<String>,
    /// Cache state (`complete` / `partial` / `missing`); present only with
    /// `--show-cached`.
    #[serde(skip_serializing_if = "Option::is_none")]
    cached: Option<String>,
}

/// Serializable result for `list-files --json`.
#[derive(serde::Serialize)]
struct ListFilesResult {
    /// Repository identifier.
    repo_id: String,
    /// Per-file entries (after `--filter` / `--exclude` / `--preset`).
    files: Vec<ListFileEntry>,
    /// Total bytes across all listed files (equals the sum of `files[].size`).
    total_bytes: u64,
    /// Number of files listed.
    file_count: usize,
    /// Number of fully-cached files; present only with `--show-cached`.
    #[serde(skip_serializing_if = "Option::is_none")]
    cached_count: Option<usize>,
}

/// Prints the `list-files` result as JSON.
fn print_list_files_json(
    repo_id: &str,
    files: &[repo::RepoFile],
    cache_marks: &[FileCacheState],
    show_cached: bool,
) -> Result<(), FetchError> {
    let mut entries: Vec<ListFileEntry> = Vec::with_capacity(files.len());
    let mut total_bytes: u64 = 0;
    let mut cached_count: usize = 0;

    // EXPLICIT: accumulates total_bytes / cached_count alongside entry
    // construction; an iterator chain would hide the running totals.
    for (i, f) in files.iter().enumerate() {
        let size = f.size.unwrap_or(0);
        total_bytes = total_bytes.saturating_add(size);

        let cached = if show_cached {
            let state = cache_marks
                .get(i)
                .copied()
                .unwrap_or(FileCacheState::Missing);
            if matches!(state, FileCacheState::Complete) {
                cached_count += 1;
            }
            // BORROW: explicit .to_owned() for &'static str → owned String
            Some(state.word().to_owned())
        } else {
            None
        };

        entries.push(ListFileEntry {
            // BORROW: explicit .clone() for owned String field
            filename: f.filename.clone(),
            size,
            // BORROW: explicit .clone() for Option<String> field
            sha256: f.sha256.clone(),
            cached,
        });
    }

    let result = ListFilesResult {
        // BORROW: explicit .to_owned() for &str → owned String
        repo_id: repo_id.to_owned(),
        file_count: entries.len(),
        files: entries,
        total_bytes,
        cached_count: if show_cached {
            Some(cached_count)
        } else {
            None
        },
    };

    emit_json(&result)
}

/// Parses a size string with binary suffix into a byte count.
///
/// Accepted forms (case-insensitive):
/// - Plain integer: `"1024"` → 1024 bytes.
/// - Integer with suffix: `"5GiB"`, `"500MiB"`, `"100KiB"`, `"2tib"`.
/// - Decimal with suffix: `"1.5GiB"`, `"0.5MiB"`.
///
/// Suffixes recognized: `B`, `KiB`, `MiB`, `GiB`, `TiB`. Decimal-prefixed
/// suffixes (`KB`, `MB`, `GB`, `TB`) are rejected with an error pointing at
/// the binary spelling — the rest of `hf-fm` formats sizes in binary units
/// and silent reinterpretation would mislead users. Used by the
/// `cache gc --max-size` clap value parser.
///
/// # Errors
///
/// Returns a clap-compatible error string when:
/// - the input is empty or has no digits,
/// - the numeric portion fails to parse as a non-negative finite number,
/// - the suffix is unrecognized or a decimal alias (`KB`/`MB`/`GB`/`TB`),
/// - the resulting byte count overflows [`u64`].
fn parse_size_arg(s: &str) -> Result<u64, String> {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * 1024 * 1024;
    const TIB: u64 = 1024 * GIB;

    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Err("empty size value".to_owned());
    }

    // Split numeric prefix from optional suffix at the first non-digit, non-dot char.
    let split_at = trimmed
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(trimmed.len());
    let (number_part, suffix_part) = trimmed.split_at(split_at);

    if number_part.is_empty() {
        return Err(format!("missing number in size value {s:?}"));
    }

    let suffix = suffix_part.trim();
    // BORROW: explicit .as_str() on the lowercased temporary String for match scrutinee.
    let unit: u64 = match suffix.to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "kib" => KIB,
        "mib" => MIB,
        "gib" => GIB,
        "tib" => TIB,
        "kb" | "mb" | "gb" | "tb" => {
            return Err(format!(
                "decimal size unit {suffix:?} not supported (use binary units: KiB, MiB, GiB, TiB)"
            ));
        }
        _ => {
            return Err(format!(
                "unrecognized size unit {suffix:?} (expected B, KiB, MiB, GiB, TiB)"
            ));
        }
    };

    // Integer fast path — avoids any float casts.
    if !number_part.contains('.') {
        let n: u64 = number_part
            .parse()
            .map_err(|e| format!("invalid number in size value {s:?}: {e}"))?;
        return n
            .checked_mul(unit)
            .ok_or_else(|| format!("size value {s:?} overflows u64"));
    }

    // Fractional path: parse as f64, multiply, narrow to u64.
    let n: f64 = number_part
        .parse()
        .map_err(|e| format!("invalid number in size value {s:?}: {e}"))?;
    if !n.is_finite() || n < 0.0 {
        return Err(format!("invalid number in size value {s:?}"));
    }
    // CAST: u64 → f64. The `KIB`..`TIB` constants are powers of 2 ≤ 2^40,
    // exactly representable in an f64 mantissa (53 bits).
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let unit_f = unit as f64;
    let bytes_f = n * unit_f;
    // CAST: u64 → f64 for the upper bound. `u64::MAX` rounds up to ≈1.8e19;
    // the comparison is conservative — values up to that f64 boundary pass.
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let max_f = u64::MAX as f64;
    if !bytes_f.is_finite() || bytes_f < 0.0 || bytes_f > max_f {
        return Err(format!("size value {s:?} overflows u64"));
    }
    // CAST: f64 → u64. Range checked above; truncation toward zero is the
    // intended rounding mode for fractional bytes (`"0.5KiB"` → 512).
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::as_conversions
    )]
    let bytes = bytes_f as u64;
    Ok(bytes)
}

// `format_size` lives in `src/format.rs` and is imported at the top of this
// file via `use format::format_size;`. Extraction (v0.10.1) was prompted by
// `src/gpu_check.rs` needing the same formatter for the `--check-gpu` verdict
// block; further binary-internal display helpers belong in that module.

/// Formats a [`SystemTime`] as a human-readable relative age string.
///
/// Buckets: `"< 1 hour"`, `"N hours ago"`, `"N days ago"`,
/// `"N months ago"`, `"N years ago"`.
///
/// [`SystemTime`]: std::time::SystemTime
fn format_age(time: std::time::SystemTime) -> String {
    const HOUR: u64 = 3600;
    const DAY: u64 = 86_400;
    const MONTH: u64 = 30 * DAY;
    const YEAR: u64 = 365 * DAY;

    let Ok(elapsed) = time.elapsed() else {
        return "\u{2014}".to_owned(); // clock skew or future timestamp
    };
    let secs = elapsed.as_secs();

    if secs < HOUR {
        "< 1 hour".to_owned()
    } else if secs < DAY {
        let hours = secs / HOUR;
        if hours == 1 {
            "1 hour ago".to_owned()
        } else {
            format!("{hours} hours ago")
        }
    } else if secs < MONTH {
        let days = secs / DAY;
        if days == 1 {
            "1 day ago".to_owned()
        } else {
            format!("{days} days ago")
        }
    } else if secs < YEAR {
        let months = secs / MONTH;
        if months == 1 {
            "1 month ago".to_owned()
        } else {
            format!("{months} months ago")
        }
    } else {
        let years = secs / YEAR;
        if years == 1 {
            "1 year ago".to_owned()
        } else {
            format!("{years} years ago")
        }
    }
}

/// Resolves the flat-copy target directory from an optional `--output-dir`.
///
/// Falls back to the current working directory when no explicit directory is given.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the current directory cannot be determined.
fn resolve_flat_target(output_dir: Option<&Path>) -> Result<PathBuf, FetchError> {
    match output_dir {
        // BORROW: explicit .to_path_buf() for &Path → owned PathBuf
        Some(dir) => Ok(dir.to_path_buf()),
        None => std::env::current_dir().map_err(|e| FetchError::Io {
            path: PathBuf::from("."),
            source: e,
        }),
    }
}

/// Copies downloaded files to a flat directory layout.
///
/// Each file is copied from the HF cache to `{target_dir}/{basename}`.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if directory creation or file copy fails.
fn flatten_files(
    file_map: &HashMap<String, PathBuf>,
    target_dir: &Path,
) -> Result<Vec<PathBuf>, FetchError> {
    // BORROW: explicit .to_path_buf() for &Path → owned PathBuf
    std::fs::create_dir_all(target_dir).map_err(|e| FetchError::Io {
        path: target_dir.to_path_buf(),
        source: e,
    })?;

    let mut flat_paths = Vec::with_capacity(file_map.len());
    for (filename, cache_path) in file_map {
        // BORROW: explicit .as_str() instead of Deref coercion
        let basename = Path::new(filename)
            .file_name()
            .unwrap_or(std::ffi::OsStr::new(filename.as_str()));
        let flat_path = target_dir.join(basename);
        // BORROW: explicit .clone() for owned PathBuf
        std::fs::copy(cache_path, &flat_path).map_err(|e| FetchError::Io {
            path: flat_path.clone(),
            source: e,
        })?;
        flat_paths.push(flat_path);
    }
    Ok(flat_paths)
}

/// Copies a single downloaded file to a flat directory layout.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if directory creation or file copy fails.
fn flatten_single_file(cache_path: &Path, target_dir: &Path) -> Result<PathBuf, FetchError> {
    // BORROW: explicit .to_path_buf() for &Path → owned PathBuf
    std::fs::create_dir_all(target_dir).map_err(|e| FetchError::Io {
        path: target_dir.to_path_buf(),
        source: e,
    })?;

    let basename = cache_path
        .file_name()
        .unwrap_or(std::ffi::OsStr::new("file"));
    let flat_path = target_dir.join(basename);
    // BORROW: explicit .clone() for owned PathBuf
    std::fs::copy(cache_path, &flat_path).map_err(|e| FetchError::Io {
        path: flat_path.clone(),
        source: e,
    })?;
    Ok(flat_path)
}

/// Warns when `--filter` globs are redundant with the active `--preset`.
fn warn_redundant_filters(preset: &Preset, filters: &[String]) {
    let (preset_globs, preset_name): (&[&str], &str) = match preset {
        Preset::Safetensors => (&["*.safetensors", "*.json", "*.txt"], "safetensors"),
        Preset::Gguf => (&["*.gguf", "*.json", "*.txt"], "gguf"),
        Preset::Npz => {
            let globs: &[&str] = &["*.npz", "*.npy", "config.yaml", "*.json", "*.txt"];
            (globs, "npz")
        }
        Preset::Pth => {
            let globs: &[&str] = &["pytorch_model*.bin", "*.json", "*.txt"];
            (globs, "pth")
        }
        Preset::ConfigOnly => (&["*.json", "*.txt", "*.md"], "config-only"),
    };
    for filter in filters {
        // BORROW: explicit .as_str() instead of Deref coercion
        if preset_globs.contains(&filter.as_str()) {
            eprintln!("warning: --filter \"{filter}\" is redundant with --preset {preset_name}");
        }
    }
}

/// Recursively sums the sizes of all files under `dir`.
///
/// Returns `0` if the directory cannot be read.
fn walk_dir_size(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut total: u64 = 0;
    for entry in entries.flatten() {
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if meta.is_dir() {
            total = total.saturating_add(walk_dir_size(&entry.path()));
        } else {
            total = total.saturating_add(meta.len());
        }
    }
    total
}

/// Prints a download summary line showing total size, elapsed time, and throughput.
fn print_download_summary(path: &Path, elapsed: Duration) {
    let total_bytes = if path.is_dir() {
        walk_dir_size(path)
    } else {
        std::fs::metadata(path).map_or(0, |m| m.len())
    };
    let elapsed_secs = elapsed.as_secs_f64();
    if total_bytes > 0 && elapsed_secs > 0.0 {
        // CAST: u64 → f64, precision loss acceptable; display-only throughput
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let throughput = total_bytes as f64 / elapsed_secs / (1024.0 * 1024.0);
        println!(
            "  {} in {:.1}s ({:.1} MiB/s)",
            format_size(total_bytes),
            elapsed_secs,
            throughput
        );
    }
}

/// Formats a download count with thousand separators (e.g., `1,234,567`).
fn format_downloads(n: u64) -> String {
    // BORROW: explicit .to_string() for u64 → String
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            result.push(',');
        }
        result.push(ch);
    }
    result
}

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;

    // ---------- is_auth_status_error / enrich_gated_content_error ----------

    #[test]
    fn auth_status_error_detects_401_and_403_http_errors() {
        let forbidden = FetchError::Http(
            "Range request for model-00002-of-00004.safetensors returned status 403 Forbidden"
                .to_owned(),
        );
        let unauthorized = FetchError::Http(
            "shard index request for org/model returned status 401 Unauthorized".to_owned(),
        );
        assert!(is_auth_status_error(&forbidden));
        assert!(is_auth_status_error(&unauthorized));
    }

    // ---------- matches_filter (case-insensitive substring) ----------

    #[test]
    fn matches_filter_is_case_insensitive() {
        // Exact-case substring still matches — no regression vs the old
        // case-sensitive `contains`.
        assert!(matches_filter(
            "model.layers.0.mlp.down_proj.weight",
            "layers.0"
        ));
        // Mixed-/upper-case pattern now matches a lower-case name: the v0.10.6
        // alignment with the case-insensitive `inspect --pick` contract.
        assert!(matches_filter(
            "model.layers.0.mlp.down_proj.weight",
            "Layers.0"
        ));
        assert!(matches_filter(
            "model.layers.0.mlp.down_proj.weight",
            "LAYERS"
        ));
        // Upper-case name, lower-case pattern (GGUF `blk.` style).
        assert!(matches_filter("BLK.0.ATTN_Q.weight", "blk.0"));
        // Non-substring still fails.
        assert!(!matches_filter("model.embed_tokens.weight", "layers.0"));
        // Empty pattern is a substring of every name.
        assert!(matches_filter("anything", ""));
    }

    #[test]
    fn auth_status_error_ignores_other_errors() {
        let not_found = FetchError::Http(
            "Range request for x.safetensors returned status 404 Not Found".to_owned(),
        );
        let io_like = FetchError::InvalidArgument("403 in a filename is not a status".to_owned());
        assert!(!is_auth_status_error(&not_found));
        assert!(!is_auth_status_error(&io_like));
    }

    #[test]
    fn enrich_gated_content_error_passes_non_auth_errors_through() {
        // Non-401/403 errors must come back untouched (and without any
        // network probe — the early return precedes the metadata fetch).
        let original =
            FetchError::Http("Range request for x returned status 404 Not Found".to_owned());
        let enriched = enrich_gated_content_error(original, "org/model", None);
        assert!(
            matches!(&enriched, FetchError::Http(msg) if msg.contains("404")),
            "expected the original Http error to pass through, got: {enriched}"
        );
    }

    // ---------- narrow_pick_candidates / parse_pick_input ----------

    fn sample_listing() -> Vec<(String, u64)> {
        vec![
            ("model-00001-of-00002.safetensors".to_owned(), 100),
            ("model-00002-of-00002.safetensors".to_owned(), 200),
            ("params.npz".to_owned(), 50),
            (
                "transformer/demonCORESFWNSFW_fluxV13.safetensors".to_owned(),
                300,
            ),
        ]
    }

    #[test]
    fn narrow_pick_candidates_no_needle_keeps_all() {
        let entries = sample_listing();
        assert_eq!(narrow_pick_candidates(&entries, None).len(), entries.len());
    }

    #[test]
    fn narrow_pick_candidates_substring_is_case_insensitive() {
        let entries = sample_listing();
        let hits = narrow_pick_candidates(&entries, Some("demoncore"));
        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits[0].0,
            "transformer/demonCORESFWNSFW_fluxV13.safetensors"
        );
    }

    #[test]
    fn narrow_pick_candidates_matches_path_prefix_too() {
        // The substring contract covers the full repo-relative name,
        // directories included.
        let entries = sample_listing();
        let hits = narrow_pick_candidates(&entries, Some("TRANSFORMER/"));
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn narrow_pick_candidates_multiple_matches_preserve_order() {
        let entries = sample_listing();
        let hits = narrow_pick_candidates(&entries, Some("model-0000"));
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, "model-00001-of-00002.safetensors");
        assert_eq!(hits[1].0, "model-00002-of-00002.safetensors");
    }

    #[test]
    fn narrow_pick_candidates_no_match_is_empty() {
        let entries = sample_listing();
        assert!(narrow_pick_candidates(&entries, Some("gguf")).is_empty());
    }

    #[test]
    fn parse_pick_input_accepts_in_range_with_whitespace() {
        assert_eq!(parse_pick_input("2", 3), Some(2));
        assert_eq!(parse_pick_input("  3 \n", 3), Some(3));
        assert_eq!(parse_pick_input("1", 1), Some(1));
    }

    #[test]
    fn parse_pick_input_rejects_zero_out_of_range_and_garbage() {
        assert_eq!(parse_pick_input("0", 3), None);
        assert_eq!(parse_pick_input("4", 3), None);
        assert_eq!(parse_pick_input("foo", 3), None);
        assert_eq!(parse_pick_input("-1", 3), None);
        assert_eq!(parse_pick_input("2.5", 3), None);
        assert_eq!(parse_pick_input("", 3), None);
    }

    // ---------- multi_file_source_label ----------

    #[test]
    fn multi_file_source_label_empty_is_unknown() {
        assert_eq!(multi_file_source_label(&[]), "unknown");
    }

    #[test]
    fn multi_file_source_label_all_cached() {
        use inspect::InspectSource::Cached;
        assert_eq!(multi_file_source_label(&[Cached, Cached]), "cached");
    }

    #[test]
    fn multi_file_source_label_all_remote() {
        use inspect::InspectSource::Remote;
        assert_eq!(multi_file_source_label(&[Remote, Remote, Remote]), "remote");
    }

    #[test]
    fn multi_file_source_label_mixed_reports_counts() {
        use inspect::InspectSource::{Cached, Remote};
        assert_eq!(
            multi_file_source_label(&[Cached, Remote, Remote]),
            "mixed (1 cached, 2 remote)"
        );
    }

    #[test]
    fn multi_file_source_label_single_cached() {
        use inspect::InspectSource::Cached;
        assert_eq!(multi_file_source_label(&[Cached]), "cached");
    }

    // ---------- parse_size_arg ----------

    #[test]
    fn parse_size_arg_plain_integer() {
        assert_eq!(parse_size_arg("1024").unwrap(), 1024);
        assert_eq!(parse_size_arg("0").unwrap(), 0);
    }

    #[test]
    fn parse_size_arg_bytes_suffix() {
        assert_eq!(parse_size_arg("512B").unwrap(), 512);
        assert_eq!(parse_size_arg("512b").unwrap(), 512);
    }

    #[test]
    fn parse_size_arg_binary_suffixes() {
        assert_eq!(parse_size_arg("1KiB").unwrap(), 1024);
        assert_eq!(parse_size_arg("1MiB").unwrap(), 1024 * 1024);
        assert_eq!(parse_size_arg("1GiB").unwrap(), 1024 * 1024 * 1024);
        assert_eq!(parse_size_arg("1TiB").unwrap(), 1024_u64.pow(4));
    }

    #[test]
    fn parse_size_arg_case_insensitive() {
        assert_eq!(parse_size_arg("5gib").unwrap(), 5 * 1024 * 1024 * 1024);
        assert_eq!(parse_size_arg("5GIB").unwrap(), 5 * 1024 * 1024 * 1024);
        assert_eq!(parse_size_arg("5GiB").unwrap(), 5 * 1024 * 1024 * 1024);
    }

    #[test]
    fn parse_size_arg_whitespace_tolerant() {
        assert_eq!(parse_size_arg("  5GiB  ").unwrap(), 5 * 1024 * 1024 * 1024);
        assert_eq!(parse_size_arg("5  GiB").unwrap(), 5 * 1024 * 1024 * 1024);
    }

    #[test]
    fn parse_size_arg_fractional() {
        assert_eq!(
            parse_size_arg("1.5GiB").unwrap(),
            1024 * 1024 * 1024 * 3 / 2
        );
        assert_eq!(parse_size_arg("0.5KiB").unwrap(), 512);
        assert_eq!(parse_size_arg(".5MiB").unwrap(), 512 * 1024);
    }

    #[test]
    fn parse_size_arg_rejects_decimal_suffix() {
        for input in ["5KB", "5MB", "5GB", "5TB", "5gb"] {
            let err = parse_size_arg(input).unwrap_err();
            assert!(
                err.contains("decimal size unit") && err.contains("binary units"),
                "input {input:?} should be rejected as decimal, got: {err}"
            );
        }
    }

    #[test]
    fn parse_size_arg_rejects_unknown_suffix() {
        let err = parse_size_arg("5xyz").unwrap_err();
        assert!(err.contains("unrecognized size unit"), "got: {err}");
    }

    #[test]
    fn parse_size_arg_rejects_empty() {
        assert!(parse_size_arg("").unwrap_err().contains("empty"));
        assert!(parse_size_arg("   ").unwrap_err().contains("empty"));
    }

    #[test]
    fn parse_size_arg_rejects_missing_digits() {
        assert!(parse_size_arg("GiB")
            .unwrap_err()
            .contains("missing number"));
        assert!(parse_size_arg("-5GiB")
            .unwrap_err()
            .contains("missing number"));
    }

    #[test]
    fn parse_size_arg_rejects_malformed_number() {
        assert!(parse_size_arg("1.2.3GiB").is_err());
    }

    #[test]
    fn parse_size_arg_overflow() {
        // Integer overflow path: 2^54 KiB > u64::MAX.
        let huge = format!("{}KiB", u64::MAX);
        assert!(parse_size_arg(huge.as_str())
            .unwrap_err()
            .contains("overflow"));
    }

    // ---------- gc fixtures ----------

    use std::time::{Duration, SystemTime};

    fn make_summary(
        repo_id: &str,
        size: u64,
        mtime: Option<SystemTime>,
        has_partial: bool,
    ) -> cache::CachedModelSummary {
        cache::CachedModelSummary {
            repo_id: repo_id.to_owned(),
            file_count: 1,
            total_size: size,
            has_partial,
            last_modified: mtime,
        }
    }

    /// Returns a fixed reference point so tests aren't sensitive to wall clock.
    fn fixed_now() -> SystemTime {
        // SystemTime::UNIX_EPOCH + 1_700_000_000 secs ≈ Nov 2023.
        SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000)
    }

    // ---------- select_age_evictions ----------

    #[test]
    fn select_age_evictions_skips_none_mtime() {
        let summaries = [make_summary("a/b", 100, None, false)];
        let set = select_age_evictions(&summaries, 86_400, fixed_now());
        assert!(set.is_empty(), "None mtime must not be age-evicted");
    }

    #[test]
    fn select_age_evictions_skips_future_mtime() {
        let now = fixed_now();
        let future = now + Duration::from_secs(86_400);
        let summaries = [make_summary("a/b", 100, Some(future), false)];
        let set = select_age_evictions(&summaries, 0, now);
        assert!(
            set.is_empty(),
            "future-dated mtime should be treated as age 0, not selected"
        );
    }

    #[test]
    fn select_age_evictions_includes_older_excludes_recent() {
        let now = fixed_now();
        let old = now - Duration::from_secs(86_400 * 31);
        let recent = now - Duration::from_secs(86_400 * 5);
        let summaries = [
            make_summary("old/repo", 100, Some(old), false),
            make_summary("recent/repo", 100, Some(recent), false),
        ];
        let set = select_age_evictions(&summaries, 86_400 * 30, now);
        assert!(set.contains("old/repo"));
        assert!(!set.contains("recent/repo"));
    }

    #[test]
    fn select_age_evictions_zero_threshold_evicts_everything_with_known_mtime() {
        let now = fixed_now();
        let summaries = [
            make_summary("a/b", 100, Some(now - Duration::from_secs(1)), false),
            make_summary("c/d", 100, None, false),
        ];
        let set = select_age_evictions(&summaries, 0, now);
        assert!(set.contains("a/b"));
        assert!(
            !set.contains("c/d"),
            "None mtime still skipped at threshold 0"
        );
    }

    // ---------- compute_gc_plan ----------

    fn empty_criteria() -> GcCriteria {
        GcCriteria {
            older_than_secs: None,
            max_size: None,
            except: HashSet::new(),
        }
    }

    #[test]
    fn compute_gc_plan_age_only() {
        let now = fixed_now();
        let summaries = [
            make_summary(
                "old/a",
                1_000,
                Some(now - Duration::from_secs(86_400 * 60)),
                false,
            ),
            make_summary(
                "new/b",
                2_000,
                Some(now - Duration::from_secs(86_400 * 5)),
                false,
            ),
        ];
        let mut crit = empty_criteria();
        crit.older_than_secs = Some(86_400 * 30);
        let plan = compute_gc_plan(&summaries, &crit, now, false);
        assert_eq!(plan.evict.len(), 1);
        assert_eq!(plan.evict[0].repo_id, "old/a");
        assert_eq!(plan.size_before, 3_000);
        assert_eq!(plan.size_after, 2_000);
        assert!(!plan.budget_shortfall);
    }

    #[test]
    fn compute_gc_plan_size_only_oldest_first() {
        let now = fixed_now();
        let summaries = [
            make_summary(
                "newest/c",
                500,
                Some(now - Duration::from_secs(86_400)),
                false,
            ),
            make_summary(
                "oldest/a",
                400,
                Some(now - Duration::from_secs(86_400 * 30)),
                false,
            ),
            make_summary(
                "middle/b",
                300,
                Some(now - Duration::from_secs(86_400 * 10)),
                false,
            ),
        ];
        let mut crit = empty_criteria();
        crit.max_size = Some(700); // total 1200, must drop ≥ 500.
        let plan = compute_gc_plan(&summaries, &crit, now, false);
        // Oldest evicted first, then middle (300+400=700 freed; 1200-700=500 ≤ 700).
        assert_eq!(plan.evict.len(), 2);
        assert_eq!(plan.evict[0].repo_id, "oldest/a");
        assert_eq!(plan.evict[1].repo_id, "middle/b");
        assert_eq!(plan.size_after, 500);
        assert!(!plan.budget_shortfall);
    }

    #[test]
    fn compute_gc_plan_combined_age_first_then_budget() {
        let now = fixed_now();
        let summaries = [
            make_summary(
                "ancient/a",
                100,
                Some(now - Duration::from_secs(86_400 * 90)),
                false,
            ),
            make_summary(
                "old/b",
                100,
                Some(now - Duration::from_secs(86_400 * 60)),
                false,
            ),
            make_summary(
                "midage/c",
                400,
                Some(now - Duration::from_secs(86_400 * 20)),
                false,
            ),
            make_summary(
                "fresh/d",
                400,
                Some(now - Duration::from_secs(86_400)),
                false,
            ),
        ];
        let mut crit = empty_criteria();
        crit.older_than_secs = Some(86_400 * 30); // catches ancient + old.
        crit.max_size = Some(500); // after age: total 1000-200=800; need 800-500=300 more.
        let plan = compute_gc_plan(&summaries, &crit, now, false);
        // Age step removes ancient + old (200 freed). Budget step adds midage (400)
        // → cache_after = 1000-200-400 = 400 ≤ 500.
        let evicted_ids: Vec<&str> = plan.evict.iter().map(|e| e.repo_id.as_str()).collect();
        assert_eq!(evicted_ids, vec!["ancient/a", "old/b", "midage/c"]);
        assert_eq!(plan.size_after, 400);
        assert!(!plan.budget_shortfall);
    }

    #[test]
    fn compute_gc_plan_except_protects_repo() {
        let now = fixed_now();
        let summaries = [
            make_summary(
                "old/a",
                500,
                Some(now - Duration::from_secs(86_400 * 60)),
                false,
            ),
            make_summary(
                "old/b",
                500,
                Some(now - Duration::from_secs(86_400 * 60)),
                false,
            ),
        ];
        let mut crit = empty_criteria();
        crit.older_than_secs = Some(86_400 * 30);
        crit.except = HashSet::from(["old/a".to_owned()]);
        let plan = compute_gc_plan(&summaries, &crit, now, false);
        assert_eq!(plan.evict.len(), 1);
        assert_eq!(plan.evict[0].repo_id, "old/b");
        assert_eq!(plan.protected.len(), 1);
        assert_eq!(plan.protected[0].repo_id, "old/a");
    }

    #[test]
    fn compute_gc_plan_except_causes_budget_shortfall() {
        let now = fixed_now();
        let summaries = [
            make_summary(
                "huge/a",
                1_000,
                Some(now - Duration::from_secs(86_400)),
                false,
            ),
            make_summary(
                "tiny/b",
                10,
                Some(now - Duration::from_secs(86_400 * 60)),
                false,
            ),
        ];
        let mut crit = empty_criteria();
        crit.max_size = Some(500); // unreachable while huge/a is protected.
        crit.except = HashSet::from(["huge/a".to_owned()]);
        let plan = compute_gc_plan(&summaries, &crit, now, false);
        // tiny/b gets evicted; huge/a stays → cache = 1000 > 500.
        assert_eq!(plan.evict.len(), 1);
        assert!(plan.budget_shortfall);
    }

    #[test]
    fn compute_gc_plan_skips_fresh_partial() {
        let now = fixed_now();
        let summaries = [make_summary(
            "active/repo",
            1_000,
            Some(now - Duration::from_secs(60 * 30)), // 30 min ago
            true,
        )];
        let mut crit = empty_criteria();
        crit.max_size = Some(0);
        let plan = compute_gc_plan(&summaries, &crit, now, false);
        assert!(plan.evict.is_empty());
        assert_eq!(plan.skipped_partials.len(), 1);
        assert!(plan.budget_shortfall);
    }

    #[test]
    fn compute_gc_plan_evicts_stale_partial() {
        let now = fixed_now();
        let summaries = [make_summary(
            "stale/repo",
            1_000,
            Some(now - Duration::from_secs(86_400 * 60)),
            true, // partial, but 60 days old → stale.
        )];
        let mut crit = empty_criteria();
        crit.older_than_secs = Some(86_400 * 30);
        let plan = compute_gc_plan(&summaries, &crit, now, false);
        assert_eq!(plan.evict.len(), 1);
        assert!(plan.skipped_partials.is_empty());
    }

    #[test]
    fn compute_gc_plan_deterministic_order_on_tied_mtime() {
        let now = fixed_now();
        let mtime = Some(now - Duration::from_secs(86_400 * 60));
        // Insertion order is reversed alphabetically; expect output sorted ascending.
        let summaries = [
            make_summary("zzz/last", 100, mtime, false),
            make_summary("aaa/first", 100, mtime, false),
            make_summary("mmm/middle", 100, mtime, false),
        ];
        let mut crit = empty_criteria();
        crit.older_than_secs = Some(86_400 * 30);
        let plan = compute_gc_plan(&summaries, &crit, now, false);
        let order: Vec<&str> = plan.evict.iter().map(|e| e.repo_id.as_str()).collect();
        assert_eq!(order, vec!["aaa/first", "mmm/middle", "zzz/last"]);
    }

    #[test]
    fn compute_gc_plan_lists_kept_when_flag_set() {
        let now = fixed_now();
        let summaries = [
            make_summary(
                "old/a",
                100,
                Some(now - Duration::from_secs(86_400 * 60)),
                false,
            ),
            make_summary("new/b", 100, Some(now - Duration::from_secs(86_400)), false),
        ];
        let mut crit = empty_criteria();
        crit.older_than_secs = Some(86_400 * 30);
        let plan = compute_gc_plan(&summaries, &crit, now, true);
        assert_eq!(plan.kept.len(), 1);
        assert_eq!(plan.kept[0].repo_id, "new/b");
    }

    #[test]
    fn compute_gc_plan_omits_kept_by_default() {
        let now = fixed_now();
        let summaries = [
            make_summary(
                "old/a",
                100,
                Some(now - Duration::from_secs(86_400 * 60)),
                false,
            ),
            make_summary("new/b", 100, Some(now - Duration::from_secs(86_400)), false),
        ];
        let mut crit = empty_criteria();
        crit.older_than_secs = Some(86_400 * 30);
        let plan = compute_gc_plan(&summaries, &crit, now, false);
        assert!(
            plan.kept.is_empty(),
            "kept list should be empty when list_kept=false"
        );
    }

    #[test]
    fn compute_gc_plan_zero_byte_repo_handled() {
        let now = fixed_now();
        let summaries = [
            make_summary(
                "empty/a",
                0,
                Some(now - Duration::from_secs(86_400 * 60)),
                false,
            ),
            make_summary(
                "real/b",
                100,
                Some(now - Duration::from_secs(86_400 * 60)),
                false,
            ),
        ];
        let mut crit = empty_criteria();
        crit.older_than_secs = Some(86_400 * 30);
        let plan = compute_gc_plan(&summaries, &crit, now, false);
        assert_eq!(plan.evict.len(), 2);
        assert_eq!(plan.size_before, 100);
        assert_eq!(plan.size_after, 0);
    }

    #[test]
    fn compute_gc_plan_unknown_mtime_sorted_first_for_budget() {
        let now = fixed_now();
        let summaries = [
            make_summary(
                "known/a",
                400,
                Some(now - Duration::from_secs(86_400 * 30)),
                false,
            ),
            make_summary("unknown/b", 400, None, false),
        ];
        let mut crit = empty_criteria();
        crit.max_size = Some(500); // total 800, drop ≥ 300.
        let plan = compute_gc_plan(&summaries, &crit, now, false);
        // unknown/b evicted first (None < Some).
        assert_eq!(plan.evict.len(), 1);
        assert_eq!(plan.evict[0].repo_id, "unknown/b");
    }

    // ---------- diff --dtypes ----------

    fn make_tensor_info(
        name: &str,
        dtype: &str,
        shape: Vec<usize>,
        byte_len: u64,
    ) -> inspect::TensorInfo {
        inspect::TensorInfo {
            name: name.to_owned(),
            dtype: dtype.to_owned(),
            shape,
            data_offsets: (0, byte_len),
        }
    }

    #[test]
    fn compute_dtype_groups_generic_works_with_hashmap_values() {
        let mut map: HashMap<String, inspect::TensorInfo> = HashMap::new();
        map.insert(
            "a".to_owned(),
            make_tensor_info("a", "BF16", vec![100, 100], 20_000),
        );
        map.insert(
            "b".to_owned(),
            make_tensor_info("b", "BF16", vec![50, 50], 5_000),
        );
        map.insert(
            "c".to_owned(),
            make_tensor_info("c", "F32", vec![10, 10], 400),
        );

        let vec_view: Vec<inspect::TensorInfo> = map.values().cloned().collect();
        let from_slice = compute_dtype_groups(&vec_view);
        let from_iter = compute_dtype_groups(map.values());

        assert_eq!(from_slice.len(), from_iter.len());
        for (dt, cnt, params, bytes) in &from_slice {
            let row = from_iter
                .iter()
                .find(|r| r.0 == *dt)
                .expect("dtype present in iter-based aggregation");
            assert_eq!(row.1, *cnt);
            assert_eq!(row.2, *params);
            assert_eq!(row.3, *bytes);
        }
    }

    #[test]
    fn aggregate_diff_dtypes_scaled_sibling() {
        // A and B share the same dtype mix; A is larger.
        let mut a: HashMap<String, inspect::TensorInfo> = HashMap::new();
        a.insert(
            "w1".to_owned(),
            make_tensor_info("w1", "BF16", vec![1000, 1000], 2_000_000),
        );
        a.insert(
            "w2".to_owned(),
            make_tensor_info("w2", "BF16", vec![1000, 1000], 2_000_000),
        );
        a.insert(
            "g".to_owned(),
            make_tensor_info("g", "F32", vec![1000], 4_000),
        );

        let mut b: HashMap<String, inspect::TensorInfo> = HashMap::new();
        b.insert(
            "w1".to_owned(),
            make_tensor_info("w1", "BF16", vec![500, 500], 500_000),
        );
        b.insert(
            "g".to_owned(),
            make_tensor_info("g", "F32", vec![500], 2_000),
        );

        let (rows_a, rows_b) = aggregate_diff_dtypes(&a, &b, None);
        let dtypes_a: BTreeSet<&str> = rows_a.iter().map(|r| r.dtype.as_str()).collect();
        let dtypes_b: BTreeSet<&str> = rows_b.iter().map(|r| r.dtype.as_str()).collect();
        assert_eq!(dtypes_a, dtypes_b);
        assert!(dtypes_a.contains("BF16"));
        assert!(dtypes_a.contains("F32"));
        let a_bf16 = rows_a
            .iter()
            .find(|r| r.dtype == "BF16")
            .expect("BF16 present in A");
        let b_bf16 = rows_b
            .iter()
            .find(|r| r.dtype == "BF16")
            .expect("BF16 present in B");
        assert!(a_bf16.bytes > b_bf16.bytes);
        assert!(a_bf16.tensors > b_bf16.tensors);
    }

    #[test]
    fn aggregate_diff_dtypes_architectural_variant() {
        // A has an F8_E4M3 tensor that B doesn't — the architectural-variant signature.
        let mut a: HashMap<String, inspect::TensorInfo> = HashMap::new();
        a.insert(
            "w".to_owned(),
            make_tensor_info("w", "BF16", vec![1000, 1000], 2_000_000),
        );
        a.insert(
            "expert".to_owned(),
            make_tensor_info("expert", "F8_E4M3", vec![100, 100], 10_000),
        );

        let mut b: HashMap<String, inspect::TensorInfo> = HashMap::new();
        b.insert(
            "w".to_owned(),
            make_tensor_info("w", "BF16", vec![1000, 1000], 2_000_000),
        );

        let (rows_a, rows_b) = aggregate_diff_dtypes(&a, &b, None);
        let dtypes_a: BTreeSet<&str> = rows_a.iter().map(|r| r.dtype.as_str()).collect();
        let dtypes_b: BTreeSet<&str> = rows_b.iter().map(|r| r.dtype.as_str()).collect();
        assert!(dtypes_a.contains("F8_E4M3"));
        assert!(!dtypes_b.contains("F8_E4M3"));
        // Both sides share BF16.
        assert!(dtypes_a.contains("BF16"));
        assert!(dtypes_b.contains("BF16"));
    }

    #[test]
    fn aggregate_diff_dtypes_with_filter() {
        let mut a: HashMap<String, inspect::TensorInfo> = HashMap::new();
        a.insert(
            "expert.w".to_owned(),
            make_tensor_info("expert.w", "BF16", vec![100], 200),
        );
        a.insert(
            "attn.q".to_owned(),
            make_tensor_info("attn.q", "BF16", vec![1000], 2_000),
        );

        let mut b: HashMap<String, inspect::TensorInfo> = HashMap::new();
        b.insert(
            "expert.w".to_owned(),
            make_tensor_info("expert.w", "BF16", vec![50], 100),
        );
        b.insert(
            "attn.q".to_owned(),
            make_tensor_info("attn.q", "BF16", vec![500], 1_000),
        );

        let (rows_a, rows_b) = aggregate_diff_dtypes(&a, &b, Some("expert"));
        // Filter retains only "expert.w" on both sides.
        assert_eq!(rows_a.len(), 1);
        assert_eq!(rows_b.len(), 1);
        assert_eq!(rows_a[0].dtype, "BF16");
        assert_eq!(rows_a[0].tensors, 1);
        assert_eq!(rows_a[0].bytes, 200);
        assert_eq!(rows_b[0].tensors, 1);
        assert_eq!(rows_b[0].bytes, 100);
    }

    #[test]
    fn diff_tensor_side_serializes_byte_count() {
        // The enrichment that powers the jq recipe planned for commit 3 of v0.10.2.
        let side = DiffTensorSide {
            dtype: "BF16".to_owned(),
            shape: vec![10, 10],
            byte_count: 200,
        };
        let json_str = serde_json::to_string(&side).expect("DiffTensorSide serializes cleanly");
        assert!(
            json_str.contains("\"byte_count\":200"),
            "expected byte_count in JSON, got: {json_str}"
        );
        assert!(json_str.contains("\"dtype\":\"BF16\""));
        assert!(json_str.contains("\"shape\":[10,10]"));
    }

    #[test]
    fn diff_dtypes_canonical_case_fits_design_width() {
        // Canonical scaled-sibling case for the candle #3530 use case:
        // three dtypes (BF16, U8, F32), GiB-range sizes. If a future column-format
        // change blows past the 75-char design budget on this shape, fail here so
        // the regression is caught before users see it.
        let rows_a = vec![
            DiffDtypeGroup {
                dtype: "BF16".to_owned(),
                tensors: 630,
                params: 3_610_000_000,
                bytes: 7_213_120_000,
            },
            DiffDtypeGroup {
                dtype: "U8".to_owned(),
                tensors: 192,
                params: 20_300_000_000,
                bytes: 20_303_437_824,
            },
            DiffDtypeGroup {
                dtype: "F32".to_owned(),
                tensors: 50,
                params: 70_000_000,
                bytes: 322_122_547,
            },
        ];
        let rows_b = vec![
            DiffDtypeGroup {
                dtype: "BF16".to_owned(),
                tensors: 126,
                params: 720_000_000,
                bytes: 1_438_986_240,
            },
            DiffDtypeGroup {
                dtype: "U8".to_owned(),
                tensors: 192,
                params: 16_000_000_000,
                bytes: 15_351_808_000,
            },
            DiffDtypeGroup {
                dtype: "F32".to_owned(),
                tensors: 40,
                params: 40_000_000,
                bytes: 268_435_456,
            },
        ];
        let w = diff_dtypes_column_widths(&rows_a, &rows_b);
        let total = w.total_width();
        assert!(
            total <= DIFF_DTYPES_DESIGN_WIDTH,
            "canonical histogram width = {total} chars, design target {DIFF_DTYPES_DESIGN_WIDTH}",
        );
    }

    // ---------- looks_like_default_template ----------

    #[test]
    fn template_detector_fires_on_default_hf_card() {
        // Verbatim opening of the HuggingFace default model card.
        let body = "\
# Model Card for Model ID

<!-- Provide a quick summary of what the model is/does. -->

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->
";
        assert!(looks_like_default_template(body));
    }

    #[test]
    fn template_detector_clears_a_real_readme() {
        // A content-rich README excerpt (no template markers, no HTML comments).
        let body = "\
# Llama 3.2 1B Instruct

A 1B-parameter instruction-tuned Llama 3.2 model trained on a mixture of
public and proprietary data. Optimized for low-latency on-device inference.

## Usage

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(\"meta-llama/Llama-3.2-1B\")
```

## License

This model is released under the Llama 3.2 community license.
";
        assert!(!looks_like_default_template(body));
    }

    #[test]
    fn template_detector_is_robust_to_blank_input() {
        assert!(!looks_like_default_template(""));
    }

    #[test]
    fn template_detector_fires_on_high_comment_density() {
        // 5 HTML comments out of 12 lines = 41% — above the 30% threshold.
        let body = "\
# Random Custom Title

Some intro text here.
<!-- comment 1 -->
<!-- comment 2 -->
<!-- comment 3 -->
<!-- comment 4 -->
<!-- comment 5 -->
More content.
More content.
More content.
More content.
";
        assert!(looks_like_default_template(body));
    }

    // ---------- format_header_line ----------

    #[test]
    fn format_header_line_safetensors_with_total() {
        let line = format_header_line(64_768, Some(1_073_741_824), false);
        assert!(line.starts_with("Header:"), "got: {line}");
        assert!(
            line.contains("(JSON)"),
            "safetensors line should keep `(JSON)` suffix, got: {line}"
        );
        assert!(
            line.contains("total"),
            "safetensors line with file_size should include `total`, got: {line}"
        );
    }

    #[test]
    fn format_header_line_safetensors_without_total() {
        let line = format_header_line(64_768, None, false);
        assert!(line.starts_with("Header:"), "got: {line}");
        assert!(line.contains("(JSON)"), "got: {line}");
        assert!(
            !line.contains("total"),
            "no file_size → no `total` clause, got: {line}"
        );
    }

    #[test]
    fn format_header_line_gguf_with_total() {
        let line = format_header_line(0, Some(3_705_032_704), true);
        assert!(
            line.starts_with("Size:"),
            "GGUF line should use `Size:` label, got: {line}"
        );
        assert!(
            !line.contains("(JSON)"),
            "GGUF line must not have safetensors-flavoured `(JSON)`, got: {line}"
        );
        assert!(
            !line.contains(" 0 B "),
            "GGUF line must not show the meaningless `0 B` prefix, got: {line}"
        );
    }

    #[test]
    fn format_header_line_gguf_without_total() {
        let line = format_header_line(0, None, true);
        assert_eq!(line, "Size:     (size unknown)");
    }

    // ---------- format_metadata_lines ----------

    #[test]
    fn format_metadata_lines_empty_returns_empty_vec() {
        let meta: HashMap<String, String> = HashMap::new();
        let lines = format_metadata_lines(&meta);
        assert!(
            lines.is_empty(),
            "empty metadata → no lines, got: {lines:?}"
        );
    }

    #[test]
    fn format_metadata_lines_inline_under_threshold() {
        let mut meta: HashMap<String, String> = HashMap::new();
        meta.insert("quant_method".to_owned(), "gptq".to_owned());
        meta.insert("bits".to_owned(), "4".to_owned());
        meta.insert("group_size".to_owned(), "128".to_owned());
        let lines = format_metadata_lines(&meta);
        assert_eq!(
            lines.len(),
            1,
            "≤ 6 keys → single inline line, got: {lines:?}"
        );
        let line = &lines[0];
        assert!(
            line.starts_with("Metadata: "),
            "inline form starts with `Metadata: `, got: {line}"
        );
        // All three k=v pairs must appear; ordering is alphabetical-sorted so deterministic.
        assert!(line.contains("bits=4"), "got: {line}");
        assert!(line.contains("group_size=128"), "got: {line}");
        assert!(line.contains("quant_method=gptq"), "got: {line}");
        // Sort check: `bits` < `group_size` < `quant_method` alphabetically.
        let bits_pos = line.find("bits=").expect("bits=");
        let group_pos = line.find("group_size=").expect("group_size=");
        let quant_pos = line.find("quant_method=").expect("quant_method=");
        assert!(bits_pos < group_pos, "expected alphabetical order");
        assert!(group_pos < quant_pos, "expected alphabetical order");
    }

    #[test]
    fn format_metadata_lines_tabular_over_threshold() {
        // 7 keys → above the threshold; switches to tabular block.
        let mut meta: HashMap<String, String> = HashMap::new();
        meta.insert("general.architecture".to_owned(), "llama".to_owned());
        meta.insert("general.name".to_owned(), "Mistral-7B".to_owned());
        meta.insert("general.quantization".to_owned(), "Q4_K_M".to_owned());
        meta.insert("llama.context_length".to_owned(), "32768".to_owned());
        meta.insert("llama.head_count".to_owned(), "32".to_owned());
        meta.insert("tokenizer.ggml.bos_token_id".to_owned(), "1".to_owned());
        meta.insert("gguf.version".to_owned(), "3".to_owned());

        let lines = format_metadata_lines(&meta);
        assert!(lines.len() > 1, "> 6 keys → tabular block, got: {lines:?}");
        assert_eq!(
            lines[0], "Metadata:",
            "first line is the `Metadata:` header"
        );
        // Sort clusters by prefix automatically: `general.*` then `gguf.*` then `llama.*` then `tokenizer.*`.
        let body = lines[1..].join("\n");
        let arch_pos = body
            .find("general.architecture=")
            .expect("general.architecture");
        let name_pos = body.find("general.name=").expect("general.name");
        let gguf_pos = body.find("gguf.version=").expect("gguf.version");
        let llama_pos = body
            .find("llama.context_length=")
            .expect("llama.context_length");
        let tok_pos = body
            .find("tokenizer.ggml.bos_token_id=")
            .expect("tokenizer.*");
        assert!(arch_pos < name_pos, "alphabetical within `general.*`");
        assert!(name_pos < gguf_pos, "`general.*` precedes `gguf.*`");
        assert!(gguf_pos < llama_pos, "`gguf.*` precedes `llama.*`");
        assert!(llama_pos < tok_pos, "`llama.*` precedes `tokenizer.*`");
        // Every key line has the two-space indent.
        for line in &lines[1..] {
            assert!(
                line.starts_with("  "),
                "tabular value line must be indented, got: {line:?}"
            );
        }
    }

    #[test]
    fn format_metadata_lines_multiline_value_renders_as_block() {
        // ≥ 7 keys so we're in tabular mode; one value contains newlines.
        let chat_template = "{%- for message in messages %}\n{{- '<|im_start|>' + message['role'] + '\\n' }}\n{%- endfor -%}";
        let mut meta: HashMap<String, String> = HashMap::new();
        meta.insert("a".to_owned(), "1".to_owned());
        meta.insert("b".to_owned(), "2".to_owned());
        meta.insert("c".to_owned(), "3".to_owned());
        meta.insert("d".to_owned(), "4".to_owned());
        meta.insert("e".to_owned(), "5".to_owned());
        meta.insert("f".to_owned(), "6".to_owned());
        meta.insert("z_chat_template".to_owned(), chat_template.to_owned());

        let lines = format_metadata_lines(&meta);
        // The multi-line key should appear as `  z_chat_template=` (no value on the same line)
        // followed by indented continuation lines.
        let key_line_idx = lines
            .iter()
            .position(|l| l == "  z_chat_template=")
            .expect("multi-line key gets its own header line");
        let next = lines
            .get(key_line_idx + 1)
            .expect("at least one continuation line");
        assert!(
            next.starts_with("    "),
            "continuation lines have 4-space indent, got: {next:?}"
        );
        assert!(
            next.contains("for message"),
            "first continuation line carries the value, got: {next:?}"
        );
    }

    #[test]
    fn format_metadata_lines_sort_is_deterministic() {
        let mut meta: HashMap<String, String> = HashMap::new();
        for i in 0..10 {
            meta.insert(format!("key_{i:02}"), format!("value_{i}"));
        }
        let first = format_metadata_lines(&meta);
        let second = format_metadata_lines(&meta);
        assert_eq!(first, second, "sort must be deterministic across runs");
    }

    // ---------- format_quant_lines ----------

    #[test]
    fn format_quant_lines_returns_empty_for_none() {
        let lines = format_quant_lines(None);
        assert!(lines.is_empty(), "None input → no lines, got: {lines:?}");
    }

    #[test]
    fn format_quant_lines_returns_two_lines_for_some() {
        let q = inspect::QuantInfo {
            scheme: "Bnb4".to_owned(),
            stored_bytes: 4_400_000_000,      // ~4.10 GiB
            dequantized_bytes: 8_810_000_000, // ~8.21 GiB
        };
        let lines = format_quant_lines(Some(&q));
        assert_eq!(
            lines.len(),
            2,
            "Some input → exactly two lines, got: {lines:?}"
        );
        assert!(
            lines[0].starts_with("Format:"),
            "first line is the Format: header, got: {}",
            lines[0]
        );
        assert!(
            lines[0].contains("Bnb4"),
            "Format line carries the scheme string, got: {}",
            lines[0]
        );
        assert!(
            lines[1].starts_with("Size:"),
            "second line is the Size: header, got: {}",
            lines[1]
        );
    }

    #[test]
    fn format_quant_lines_size_uses_stored_arrow_dequantised() {
        let q = inspect::QuantInfo {
            scheme: "FineGrainedFp8".to_owned(),
            stored_bytes: 4_400_000_000,
            dequantized_bytes: 8_810_000_000,
        };
        let lines = format_quant_lines(Some(&q));
        let size_line = &lines[1];
        assert!(
            size_line.contains(" stored -> "),
            "Size line must use the `stored -> ` arrow separator, got: {size_line}"
        );
        assert!(
            size_line.ends_with("(BF16)"),
            "Size line must annotate the dequantised side with `(BF16)`, got: {size_line}"
        );
    }
}
