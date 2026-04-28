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
    compile_glob_patterns, file_matches, has_glob_chars, FetchConfig, FetchConfigBuilder,
    FetchError, Filter,
};

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
    ListFamilies,
    /// Discover new model families from the `HuggingFace` Hub.
    Discover {
        /// Maximum number of models to scan.
        #[arg(long, default_value = "500")]
        limit: usize,
    },
    /// Search the `HuggingFace` Hub for models matching a query.
    ///
    /// Supports comma-separated multi-term filtering (e.g., `"mistral,3B,12"`).
    /// Slashes in queries are treated as spaces for broader matching.
    #[command(after_help = "See also: hf-fm list-families, hf-fm discover")]
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
        /// Show only tensors whose name contains this substring.
        #[arg(long)]
        filter: Option<String>,
        /// Show only the summary line (counts per category).
        #[arg(long)]
        summary: bool,
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
    },
    /// Inspect `.safetensors` file headers (tensor names, shapes, dtypes).
    ///
    /// Reads tensor metadata without downloading full weight data.
    /// Checks the local cache first; falls back to HTTP Range requests.
    #[command(after_help = "Examples:\n  \
        hf-fm inspect <repo>                             # inspect every .safetensors in the repo\n  \
        hf-fm inspect <repo> --list                      # list safetensors files (no headers read)\n  \
        hf-fm inspect <repo> 3                           # inspect file #3 from --list\n  \
        hf-fm inspect <repo> model.safetensors --tree    # hierarchical view of one file\n\n\
        Indices returned by --list are stable as long as the repo has not\n\
        changed remotely between invocations. Pass --revision <sha> on both\n\
        --list and the follow-up run to lock the view end-to-end.")]
    Inspect {
        /// The repository identifier (e.g., `"google/gemma-2-2b-it"`).
        repo_id: String,
        /// Specific `.safetensors` file, numeric index from `--list`, or omit for all.
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
        /// List `.safetensors` files in the repo (filename + size) and exit.
        ///
        /// Prints a numbered table; the `#` column can be used as the `filename`
        /// argument on a follow-up run (e.g. `hf-fm inspect <repo> 3`). Indices
        /// are alphabetical, so shard ordering is natural. No headers are read.
        #[arg(long, conflicts_with_all = ["filename", "no_metadata", "json", "filter", "dtypes", "limit", "tree"])]
        list: bool,
        /// Suppress the `Metadata:` line in human-readable output.
        #[arg(long)]
        no_metadata: bool,
        /// Output the full header as JSON instead of a human-readable table.
        #[arg(long)]
        json: bool,
        /// Show only tensors whose name contains this substring.
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
    /// Resolves the `main` ref only; repos downloaded at a non-default revision
    /// are not yet supported (planned for a future `--revision` flag).
    Path {
        /// Repository identifier or numeric index from `du` output.
        repo_id: String,
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
            Commands::ListFamilies
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
        Some(Commands::ListFamilies) => run_list_families(),
        Some(Commands::Discover { limit }) => run_discover(limit),
        // BORROW: explicit .as_str()/.as_deref() for owned → borrowed conversions
        Some(Commands::Search {
            query,
            limit,
            exact,
            library,
            pipeline,
            tag,
        }) => run_search(
            query.as_str(),
            limit,
            exact,
            library.as_deref(),
            pipeline.as_deref(),
            tag.as_deref(),
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
            flat,
        }),
        Some(Commands::Status {
            repo_id: Some(repo_id),
            revision,
            token,
            // BORROW: explicit .as_str()/.as_deref() for owned → borrowed conversions
        }) => run_status(repo_id.as_str(), revision.as_deref(), token.as_deref()),
        Some(Commands::Status { repo_id: None, .. }) => run_status_all(),
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
            json,
        ),
        // BORROW: explicit .as_str() for String → &str conversion
        Some(Commands::Du {
            repo_id: Some(repo_id),
            age: _,
            tree: _, // EXPLICIT: clap conflicts_with rejects --tree alongside repo_id
        }) => {
            // BORROW: explicit .as_str() instead of Deref coercion
            let resolved = resolve_du_arg(repo_id.as_str())?;
            run_du_repo(resolved.as_str())
        }
        Some(Commands::Du {
            repo_id: None,
            age,
            tree: true,
        }) => run_du_tree(age),
        Some(Commands::Du {
            repo_id: None,
            age,
            tree: false,
        }) => run_du(age),
        // BORROW: explicit .as_str()/.as_deref() for owned → borrowed conversions
        Some(Commands::Inspect {
            repo_id,
            filename,
            revision,
            token,
            cached,
            list,
            no_metadata,
            json,
            filter,
            dtypes,
            limit,
            tree,
        }) => run_inspect(
            repo_id.as_str(),
            filename.as_deref(),
            revision.as_deref(),
            token.as_deref(),
            cached,
            list,
            no_metadata,
            json,
            filter.as_deref(),
            dtypes,
            limit,
            tree,
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
        }) => run_list_files(
            repo_id.as_str(),
            revision.as_deref(),
            token.as_deref(),
            &filter,
            &exclude,
            preset.as_ref(),
            no_checksum,
            show_cached,
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
            // BORROW: explicit .as_str() for String → &str conversion
            CacheCommands::Path { repo_id } => {
                let resolved = resolve_du_arg(repo_id.as_str())?;
                // BORROW: explicit .as_str() instead of Deref coercion
                run_cache_path(resolved.as_str())
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
        let outcome = rt.block_on(hf_fetch_model::download_files_with_config(repo_id, &config))?;
        let elapsed = start.elapsed();

        if let Some(ref p) = indicatif {
            p.finish();
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
        let outcome = rt.block_on(hf_fetch_model::download_with_config(repo_id, &config))?;
        let elapsed = start.elapsed();

        // Finalize progress bar before printing to avoid interleaved output.
        if let Some(ref p) = indicatif {
            p.finish();
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
        flat,
    } = params;
    if !repo_id.contains('/') {
        return Err(FetchError::InvalidArgument(format!(
            "invalid REPO_ID \"{repo_id}\": expected \"org/model\" format (e.g., \"mntss/clt-gemma-2-2b-426k\")"
        )));
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

fn run_list_families() -> Result<(), FetchError> {
    let cache_dir = cache::hf_cache_dir()?;
    let families = cache::list_cached_families()?;

    println!("Cache: {}", cache_dir.display());
    println!();

    if families.is_empty() {
        println!("No model families found in local cache.");
        return Ok(());
    }

    let fw = families
        .keys()
        .map(String::len)
        .max()
        .unwrap_or(6)
        .max(6) // BORROW: "Family".len()
        + 2;
    let mw = families
        .values()
        .flat_map(|repos| repos.iter().map(String::len))
        .max()
        .unwrap_or(6)
        .max(6); // BORROW: "Models".len()
    println!("{:<fw$}Models", "Family");
    println!("{:-<fw$}{:-<mw$}", "", "");
    for (model_type, repos) in &families {
        for (i, repo) in repos.iter().enumerate() {
            if i == 0 {
                println!("{model_type:<fw$}{repo}");
            } else {
                println!("{:<fw$}{repo}", "");
            }
        }
    }

    Ok(())
}

fn run_discover(limit: usize) -> Result<(), FetchError> {
    let families = cache::list_cached_families()?;
    let local_types: HashSet<String> = families.into_keys().collect();

    let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
        path: PathBuf::from("<runtime>"),
        source: e,
    })?;

    let discovered = rt.block_on(discover::discover_new_families(&local_types, limit))?;

    if discovered.is_empty() {
        println!("No new model families found.");
        return Ok(());
    }

    println!("New families not in local cache (top models by downloads):\n");
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

fn run_search(
    query: &str,
    limit: usize,
    exact: bool,
    library: Option<&str>,
    pipeline: Option<&str>,
    tag: Option<&str>,
) -> Result<(), FetchError> {
    let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
        path: PathBuf::from("<runtime>"),
        source: e,
    })?;

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

    if exact {
        // Exact match: compare against the original query (not normalized)
        let exact_match = filtered
            .iter()
            .find(|r| r.model_id.eq_ignore_ascii_case(query));

        if let Some(matched) = exact_match {
            println!("Exact match:\n");
            print_search_result(matched, matched.model_id.len());

            // Fetch and display model card metadata
            match rt.block_on(discover::fetch_model_card(
                matched.model_id.as_str(), // BORROW: explicit .as_str()
            )) {
                Ok(card) => print_model_card(&card),
                Err(e) => eprintln!("\n  (could not fetch model card: {e})"),
            }
        } else {
            println!("No exact match for \"{query}\".");
            if !filtered.is_empty() {
                println!("\nDid you mean:\n");
                let nw = filtered.iter().map(|r| r.model_id.len()).max().unwrap_or(0);
                for result in &filtered {
                    print_search_result(result, nw);
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
                print_search_result(result, nw);
            }
        }
    }

    Ok(())
}

fn print_search_result(result: &discover::SearchResult, name_width: usize) {
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
    println!(
        "  hf-fm {:<nw$} ({} {downloads_label}){suffix}",
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

fn run_status_all() -> Result<(), FetchError> {
    let cache_dir = cache::hf_cache_dir()?;
    let summaries = cache::cache_summary()?;

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
fn run_du(age: bool) -> Result<(), FetchError> {
    let cache_dir = cache::hf_cache_dir()?;
    println!("Cache: {}\n", cache_dir.display());

    let mut summaries = cache::cache_summary()?;

    if summaries.is_empty() {
        println!("No models found in local cache.");
        return Ok(());
    }

    summaries.sort_by_key(|s| std::cmp::Reverse(s.total_size));

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
fn run_du_repo(repo_id: &str) -> Result<(), FetchError> {
    let cache_dir = cache::hf_cache_dir()?;
    println!("Cache: {}\n", cache_dir.display());

    let files = cache::cache_repo_usage(repo_id)?;

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

    // Check if this repo has partial downloads and hint the user
    // (targeted scan, not full cache).
    if cache::repo_has_partial(repo_id)? {
        println!("\n  \u{25cf} partial downloads — run `hf-fm status {repo_id}` for details");
    }

    Ok(())
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
fn run_du_tree(age: bool) -> Result<(), FetchError> {
    let cache_dir = cache::hf_cache_dir()?;
    println!("Cache: {}\n", cache_dir.display());

    let repos = build_cache_tree()?;

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
/// Resolves the `main` ref to a commit hash and constructs the snapshot
/// path. Output is a bare path with no decoration, intended for shell
/// substitution: `cd $(hf-fm cache path org/model)`.
fn run_cache_path(repo_id: &str) -> Result<(), FetchError> {
    let cache_dir = cache::hf_cache_dir()?;
    let repo_dir = hf_fetch_model::cache_layout::repo_dir(&cache_dir, repo_id);

    if !repo_dir.exists() {
        return Err(FetchError::InvalidArgument(format!(
            "{repo_id} is not cached"
        )));
    }

    let commit_hash = cache::read_ref(&repo_dir, "main").ok_or_else(|| {
        FetchError::InvalidArgument(format!("{repo_id} is cached but has no ref for \"main\""))
    })?;

    // BORROW: explicit .as_str() instead of Deref coercion
    let snapshot_dir = hf_fetch_model::cache_layout::snapshot_dir(&repo_dir, commit_hash.as_str());

    if !snapshot_dir.exists() {
        return Err(FetchError::InvalidArgument(format!(
            "snapshot directory for {repo_id} does not exist"
        )));
    }

    // Print bare path (no labels) for shell substitution.
    println!("{}", snapshot_dir.display());
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
    json: bool,
) -> Result<(), FetchError> {
    let tensors_a = collect_repo_tensors(repo_a, revision_a, token, cached)?;
    let tensors_b = collect_repo_tensors(repo_b, revision_b, token, cached)?;

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
        all_names.retain(|name| name.contains(pattern));
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
        );
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

        println!();

        // Print only-in-A.
        if !only_a.is_empty() {
            let label = if only_a.len() == 1 {
                "tensor"
            } else {
                "tensors"
            };
            println!("  Only in A ({} {label}):", only_a.len());
            for name in &only_a {
                if let Some(t) = tensors_a.get(*name) {
                    let shape_str = format!("{:?}", t.shape);
                    println!("    {name:<nw$} {:<8} {shape_str}", t.dtype);
                }
            }
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
            for name in &only_b {
                if let Some(t) = tensors_b.get(*name) {
                    let shape_str = format!("{:?}", t.shape);
                    println!("    {name:<nw$} {:<8} {shape_str}", t.dtype);
                }
            }
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
            for name in &differ {
                if let Some((a, b)) = tensors_a.get(*name).zip(tensors_b.get(*name)) {
                    let shape_a = format!("{:?}", a.shape);
                    let shape_b = format!("{:?}", b.shape);
                    println!("    {name}");
                    println!("      A: {:<8} {shape_a}", a.dtype);
                    println!("      B: {:<8} {shape_b}", b.dtype);
                }
            }
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

/// One side of a diff entry (dtype + shape).
#[derive(serde::Serialize)]
struct DiffTensorSide {
    /// Element dtype string.
    dtype: String,
    /// Tensor shape.
    shape: Vec<usize>,
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
            }),
            b: b.map(|t| DiffTensorSide {
                // BORROW: explicit .clone() for owned String
                dtype: t.dtype.clone(),
                shape: t.shape.clone(),
            }),
        }
    };

    let result = DiffResult {
        repo_a: repo_a.to_owned(),
        repo_b: repo_b.to_owned(),
        only_a: only_a
            .iter()
            .map(|n| make_entry(n, tensors_a.get(*n), None))
            .collect(),
        only_b: only_b
            .iter()
            .map(|n| make_entry(n, None, tensors_b.get(*n)))
            .collect(),
        differ: differ
            .iter()
            .map(|n| make_entry(n, tensors_a.get(*n), tensors_b.get(*n)))
            .collect(),
        matching_count: matching.len(),
        filter: filter.map(str::to_owned),
    };

    let output = serde_json::to_string_pretty(&result)
        .map_err(|e| FetchError::Http(format!("failed to serialize JSON: {e}")))?;
    println!("{output}");
    Ok(())
}

/// Returns whether `filename` ends in `.safetensors` (case-insensitive).
fn has_safetensors_extension(filename: &str) -> bool {
    Path::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
}

/// Inspects `.safetensors` file headers for tensor metadata.
#[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments)]
fn run_inspect(
    repo_id: &str,
    filename: Option<&str>,
    revision: Option<&str>,
    token: Option<&str>,
    cached: bool,
    list: bool,
    no_metadata: bool,
    json: bool,
    filter: Option<&str>,
    dtypes: bool,
    limit: Option<usize>,
    tree: bool,
) -> Result<(), FetchError> {
    if list {
        return run_inspect_list(repo_id, revision, token, cached);
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
            )
        }
        None => run_inspect_repo(repo_id, revision, token, cached, json, filter),
    }
}

/// Resolves an inspect filename argument to a concrete filename.
///
/// If `arg` parses as a positive `usize`, treats it as a 1-based index into
/// the repository's alphabetically-sorted list of `.safetensors` files and
/// returns the corresponding filename. Otherwise returns `arg` unchanged.
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

    let (entries, commit_sha) = gather_safetensors_listing(repo_id, revision, token, cached)?;

    if entries.is_empty() {
        return Err(FetchError::InvalidArgument(format!(
            "index {n} cannot be resolved: no .safetensors files in repository {repo_id} \
             (run `hf-fm inspect {repo_id} --list` to confirm)"
        )));
    }

    if n == 0 || n > entries.len() {
        return Err(FetchError::InvalidArgument(format!(
            "index {n} is out of range (repository has {count} .safetensors files — \
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

/// Fetches the `(filename, size_bytes)` list of safetensors files for a repo,
/// from either the local cache or the `HuggingFace` API, sorted alphabetically.
///
/// Also returns the commit SHA of the resolved revision when available.
fn gather_safetensors_listing(
    repo_id: &str,
    revision: Option<&str>,
    token: Option<&str>,
    cached: bool,
) -> Result<inspect::SafetensorsListing, FetchError> {
    if cached {
        return inspect::list_cached_safetensors(repo_id, revision);
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
        .filter(|f| f.filename.ends_with(".safetensors"))
        .map(|f| (f.filename, f.size.unwrap_or(0)))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    Ok((entries, commit_sha))
}

/// Prints the numbered list of `.safetensors` files in `repo_id`.
///
/// Used for discovery: tells the user what filenames / indices they can pass
/// to a follow-up `hf-fm inspect <repo> <n>` run. Does not read file headers.
fn run_inspect_list(
    repo_id: &str,
    revision: Option<&str>,
    token: Option<&str>,
    cached: bool,
) -> Result<(), FetchError> {
    let (entries, commit_sha) = gather_safetensors_listing(repo_id, revision, token, cached)?;

    println!("Repo: {repo_id}");
    let rev_label = revision.unwrap_or("main");
    match &commit_sha {
        Some(sha) => println!("Rev:  {sha} ({rev_label})"),
        None => println!("Rev:  (unknown) ({rev_label})"),
    }
    println!();

    if entries.is_empty() {
        println!("No .safetensors files in this repository.");
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
) -> Result<(), FetchError> {
    let forest = collapse_ranges(build_tree(tensors));
    let output = TreeJsonOutput {
        repo_id,
        filename,
        total_tensors: total_tensor_count,
        total_params,
        tree: tree_to_json(&forest),
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
#[derive(serde::Serialize)]
struct InspectJsonOutput<'a> {
    #[serde(flatten)]
    header: &'a inspect::SafetensorsHeaderInfo,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncated: Option<TruncationInfo>,
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
) -> Result<(), FetchError> {
    if !has_safetensors_extension(filename) {
        // BORROW: owned Strings for error variant fields
        let extension = Path::new(filename)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("unknown")
            .to_owned();
        return Err(FetchError::UnsupportedInspectFormat {
            filename: filename.to_owned(),
            extension,
        });
    }

    let (mut info, source) = if cached {
        let info = inspect::inspect_safetensors_cached(repo_id, filename, revision)?;
        (info, inspect::InspectSource::Cached)
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
        rt.block_on(inspect::inspect_safetensors(
            repo_id,
            filename,
            token.as_deref(),
            revision,
        ))?
    };

    // Apply tensor name filter.
    let total_tensor_count = info.tensors.len();
    let total_params = info.total_params();
    if let Some(pattern) = filter {
        // BORROW: explicit .as_str() instead of Deref coercion
        info.tensors.retain(|t| t.name.as_str().contains(pattern));
    }

    // Apply limit after filter. Track matched counts to report truncation.
    let matched_count = info.tensors.len();
    let matched_params = info.total_params();
    let truncated_by_limit = limit.is_some_and(|n| matched_count > n);
    if let Some(n) = limit {
        info.tensors.truncate(n);
    }

    // `--tree --json`: hierarchical tree as JSON (distinct schema from plain --json).
    if tree && json {
        return print_tree_json(
            repo_id,
            filename,
            &info.tensors,
            total_tensor_count,
            total_params,
        );
    }

    // `--dtypes --json`: compact dtype breakdown as JSON (distinct schema from plain --json).
    if dtypes && json {
        return print_dtype_summary_json(&info.tensors, total_tensor_count, total_params);
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

    let header_display = format_size(info.header_size);
    if let Some(fs) = info.file_size {
        println!(
            "  Header:   {header_display} (JSON), {} total",
            format_size(fs)
        );
    } else {
        println!("  Header:   {header_display} (JSON)");
    }

    if !no_metadata {
        if let Some(ref meta) = info.metadata {
            let entries: Vec<String> = meta.iter().map(|(k, v)| format!("{k}={v}")).collect();
            // BORROW: explicit .join() on slice
            println!("  Metadata: {}", entries.join(", "));
        }
    }

    // Hierarchical tree mode.
    if tree {
        print_tree_summary(&info.tensors, filter, total_tensor_count, total_params);
        return Ok(());
    }

    // Per-dtype summary mode.
    if dtypes {
        print_dtype_summary(&info.tensors, filter, total_tensor_count, total_params);
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
            // Three-number format: shown/matched/total.
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

    Ok(())
}

/// Inspects all `.safetensors` files in a repository (summary or per-file).
#[allow(clippy::too_many_arguments)]
fn run_inspect_repo(
    repo_id: &str,
    revision: Option<&str>,
    token: Option<&str>,
    cached: bool,
    json: bool,
    filter: Option<&str>,
) -> Result<(), FetchError> {
    if cached {
        // Cache-only: try shard index first, then walk snapshot.
        if let Some(index) = inspect::fetch_shard_index_cached(repo_id, revision)? {
            print_shard_index_summary(repo_id, &index, filter);
            print_adapter_config_if_present(repo_id, revision, None, true, json);
            return Ok(());
        }

        let results = inspect::inspect_repo_safetensors_cached(repo_id, revision)?;
        if results.is_empty() {
            println!("No cached .safetensors files found for {repo_id}.");
            println!("Hint: use `hf-fm list-files {repo_id}` to see available file types");
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

    if json {
        let mapped: Vec<(String, inspect::SafetensorsHeaderInfo)> = results
            .into_iter()
            .map(|(name, info, _source)| (name, info))
            .collect();
        print_multi_file_json(&mapped, filter)?;
        print_adapter_config_if_present(repo_id, revision, token.as_deref(), false, true);
        return Ok(());
    }

    let mapped: Vec<(String, inspect::SafetensorsHeaderInfo)> = results
        .into_iter()
        .map(|(name, info, _source)| (name, info))
        .collect();
    print_multi_file_summary(repo_id, "mixed", &mapped, filter);
    print_adapter_config_if_present(repo_id, revision, token.as_deref(), false, false);
    Ok(())
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
}

/// Groups tensors by dtype and returns rows sorted by tensor count descending.
///
/// Each row is `(dtype, count, params, bytes)`.
fn compute_dtype_groups(tensors: &[inspect::TensorInfo]) -> Vec<(&str, usize, u64, u64)> {
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

/// Prints shard index summary (tensor counts per shard).
fn print_shard_index_summary(repo_id: &str, index: &inspect::ShardedIndex, filter: Option<&str>) {
    println!("  Repo:   {repo_id}");
    println!("  Source: shard index (model.safetensors.index.json)");
    println!();

    // Count tensors per shard, optionally filtering by tensor name.
    let total_tensors = index.weight_map.len();
    let mut by_shard: HashMap<String, usize> = HashMap::new();
    let mut filtered_total: usize = 0;
    for (tensor_name, shard_name) in &index.weight_map {
        if let Some(pattern) = filter {
            if !tensor_name.contains(pattern) {
                continue;
            }
        }
        // BORROW: explicit .clone() for owned String key
        *by_shard.entry(shard_name.clone()).or_default() += 1;
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
        let count = by_shard.get(shard).copied().unwrap_or(0);
        if filter.is_some() && count == 0 {
            continue;
        }
        println!("  {shard:<fw$} {count:>8}");
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
    } else {
        println!("  {displayed_shards} {shard_label}, {filtered_total} {tensor_label}");
    }
    println!("  Hint: use `hf-fm inspect {repo_id} <filename>` for per-tensor detail");
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
                    .filter(|t| t.name.as_str().contains(pattern)) // BORROW: explicit .as_str()
                    .cloned()
                    .collect();
                if matching.is_empty() {
                    return None;
                }
                Some((
                    name.clone(), // BORROW: explicit .clone() for owned String
                    inspect::SafetensorsHeaderInfo {
                        tensors: matching,
                        metadata: info.metadata.clone(),
                        header_size: info.header_size,
                        file_size: info.file_size,
                    },
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

/// Prints multi-file inspection results as a human-readable summary.
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

        let (tensor_count, params) = if let Some(pattern) = filter {
            let matching: Vec<&inspect::TensorInfo> = info
                .tensors
                .iter()
                // BORROW: explicit .as_str() instead of Deref coercion
                .filter(|t| t.name.as_str().contains(pattern))
                .collect();
            let p: u64 = matching.iter().map(|t| t.num_elements()).sum();
            (matching.len(), p)
        } else {
            (info.tensors.len(), info.total_params())
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
}

fn run_status(
    repo_id: &str,
    revision: Option<&str>,
    token: Option<&str>,
) -> Result<(), FetchError> {
    // BORROW: explicit String::from (equivalent to .to_owned()) for Option<&str> → Option<String>
    let token = token
        .map(String::from)
        .or_else(|| std::env::var("HF_TOKEN").ok());

    let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
        path: PathBuf::from("<runtime>"),
        source: e,
    })?;

    // BORROW: explicit .as_deref() for Option<String> → Option<&str>
    let status = rt.block_on(cache::repo_status(repo_id, token.as_deref(), revision))?;

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
    println!();
    println!("{complete}/{total} complete, {partial} partial, {missing} missing");

    Ok(())
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
    let cache_marks: Vec<String> = if show_cached {
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
                            "partial".to_owned()
                        } else {
                            "\u{2713}".to_owned()
                        }
                    }
                    _ => "\u{2717}".to_owned(),
                }
            })
            .collect()
    } else {
        Vec::new()
    };

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
            let mark = cache_marks.get(i).map_or("\u{2717}", String::as_str);
            if mark == "\u{2713}" {
                cached_count += 1;
            }
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
    let row_width = fw + 2 + 10 + 2 + 12;
    println!("  {:\u{2500}<row_width$}", "");
    if show_cached {
        println!(
            "  {count} files, {} total ({cached_count} cached)",
            format_size(total_bytes)
        );
    } else {
        println!("  {count} files, {} total", format_size(total_bytes));
    }
    if any_no_sha && !no_checksum {
        println!("  \u{2014} = not an LFS file (no SHA256 tracked by the Hub)");
    }

    Ok(())
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

/// Formats a byte size with human-readable suffixes (B, KiB, MiB, GiB).
fn format_size(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * 1024 * 1024;
    const TIB: u64 = 1024 * GIB;

    // Use TiB for values >= 1000 GiB, GiB for >= 1000 MiB.
    if bytes >= 1000 * GIB {
        // CAST: u64 → f64, precision loss acceptable; value is a display-only size scalar
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let val = bytes as f64 / TIB as f64;
        format!("{val:.2} TiB")
    } else if bytes >= 1000 * MIB {
        // CAST: u64 → f64, precision loss acceptable; value is a display-only size scalar
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let val = bytes as f64 / GIB as f64;
        format!("{val:.2} GiB")
    } else if bytes >= MIB {
        // CAST: u64 → f64, precision loss acceptable; value is a display-only size scalar
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let val = bytes as f64 / MIB as f64;
        format!("{val:.2} MiB")
    } else if bytes >= KIB {
        // CAST: u64 → f64, precision loss acceptable; value is a display-only size scalar
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let val = bytes as f64 / KIB as f64;
        format!("{val:.1} KiB")
    } else {
        format!("{bytes} B")
    }
}

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
}
