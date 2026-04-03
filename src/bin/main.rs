// SPDX-License-Identifier: MIT OR Apache-2.0

//! CLI binary for hf-fetch-model.
//!
//! Installed as both `hf-fetch-model` and `hf-fm`.

use std::collections::{HashMap, HashSet};
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand, ValueEnum};
use tracing_subscriber::EnvFilter;

use hf_fetch_model::cache;
use hf_fetch_model::discover;
use hf_fetch_model::inspect;
use hf_fetch_model::progress::IndicatifProgress;
use hf_fetch_model::repo;
use hf_fetch_model::{
    compile_glob_patterns, file_matches, has_glob_chars, FetchConfig, FetchError, Filter,
};

/// Downloads all files from a `HuggingFace` model repository.
///
/// Use `--preset safetensors` to download only safetensors weights,
/// config, and tokenizer files.
#[derive(Parser)]
#[command(
    name = "hf-fetch-model",
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
    Search {
        /// Search query (e.g., `"RWKV-7"`, `"llama 3"`, `"mistral,3B,12"`).
        query: String,
        /// Maximum number of results.
        #[arg(long, default_value = "20")]
        limit: usize,
        /// Return only the exact model ID match; show model card metadata if found.
        #[arg(long)]
        exact: bool,
        /// Filter by library framework (e.g., `"transformers"`, `"peft"`, `"vllm"`).
        #[arg(long)]
        library: Option<String>,
        /// Filter by pipeline task (e.g., `"text-generation"`, `"text-classification"`).
        #[arg(long)]
        pipeline: Option<String>,
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
        /// The repository identifier (omit to show all cached repos).
        repo_id: Option<String>,
    },
    /// Inspect `.safetensors` file headers (tensor names, shapes, dtypes).
    ///
    /// Reads tensor metadata without downloading full weight data.
    /// Checks the local cache first; falls back to HTTP Range requests.
    Inspect {
        /// The repository identifier (e.g., `"google/gemma-2-2b-it"`).
        repo_id: String,
        /// Specific `.safetensors` file to inspect (omit for all).
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
        /// Suppress the `Metadata:` line in human-readable output.
        #[arg(long)]
        no_metadata: bool,
        /// Output the full header as JSON instead of a human-readable table.
        #[arg(long)]
        json: bool,
        /// Show only tensors whose name contains this substring.
        #[arg(long)]
        filter: Option<String>,
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
        /// Filter preset (`safetensors`, `gguf`, `pth`, `config-only`).
        #[arg(long, value_enum)]
        preset: Option<Preset>,
        /// Suppress the SHA256 column.
        #[arg(long)]
        no_checksum: bool,
        /// Show cache status for each file (complete, partial, or missing).
        #[arg(long)]
        show_cached: bool,
    },
}

// EXHAUSTIVE: internal CLI dispatch enum; crate owns all variants
#[derive(Clone, ValueEnum)]
enum Preset {
    Safetensors,
    Gguf,
    Pth,
    ConfigOnly,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

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
            | Commands::ListFiles { .. },
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
            eprintln!("error: {} file(s) failed to download:", failures.len());
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
        }) => run_search(
            query.as_str(),
            limit,
            exact,
            library.as_deref(),
            pipeline.as_deref(),
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
            flat,
        }) => run_download_file(DownloadFileParams {
            repo_id: repo_id.as_str(),
            filename: filename.as_str(),
            revision: revision.as_deref(),
            token: token.as_deref(),
            output_dir,
            chunk_threshold_mib,
            connections_per_file,
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
        }) => run_du_repo(repo_id.as_str()),
        Some(Commands::Du { repo_id: None }) => run_du(),
        // BORROW: explicit .as_str()/.as_deref() for owned → borrowed conversions
        Some(Commands::Inspect {
            repo_id,
            filename,
            revision,
            token,
            cached,
            no_metadata,
            json,
            filter,
        }) => run_inspect(
            repo_id.as_str(),
            filename.as_deref(),
            revision.as_deref(),
            token.as_deref(),
            cached,
            no_metadata,
            json,
            filter.as_deref(),
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
            "{} file(s) copied to {}:",
            flat_paths.len(),
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
    println!("  {:<48} {:>10}  Status", "File", "Size");
    println!(
        "  {:\u{2500}<48} {:\u{2500}<10}  {:\u{2500}<12}",
        "", "", ""
    );
    for fp in &plan.files {
        let status = if fp.cached {
            "cached \u{2713}"
        } else {
            "to download"
        };
        println!(
            "  {:<48} {:>10}  {status}",
            fp.filename,
            format_size(fp.size)
        );
    }

    // Summary.
    println!("{:\u{2500}<74}", "  ");
    let cached_count = plan.files.len() - plan.files_to_download();
    let to_dl = plan.files_to_download();
    println!(
        "  Total: {} ({} files, {} cached, {} to download)",
        format_size(plan.total_bytes),
        plan.files.len(),
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
            "{} file(s) copied to {}:",
            flat_paths.len(),
            target_dir.display()
        );
        for p in &flat_paths {
            println!("  {}", p.display());
        }
    } else {
        println!("{} file(s) matched pattern \"{pattern}\":", file_map.len());
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
    let families = cache::list_cached_families()?;

    if families.is_empty() {
        println!("No model families found in local cache.");
        return Ok(());
    }

    println!("{:<16}Models", "Family");
    println!("{:-<16}{:-<64}", "", "");
    for (model_type, repos) in &families {
        let repos_str = repos.join(", ");
        println!("{model_type:<16}{repos_str}");
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
    println!("{:<16}Top Model", "Family");
    println!("{:-<16}{:-<64}", "", "");
    for family in &discovered {
        println!("{:<16}{}", family.model_type, family.top_model);
    }

    Ok(())
}

fn run_search(
    query: &str,
    limit: usize,
    exact: bool,
    library: Option<&str>,
    pipeline: Option<&str>,
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
    let has_client_filter = filter_terms.len() > 1 || library.is_some() || pipeline.is_some();
    let api_limit = if has_client_filter {
        limit.saturating_mul(5)
    } else {
        limit
    };

    let results = rt.block_on(discover::search_models(
        api_query, api_limit, library, pipeline,
    ))?;

    // Client-side filtering: only applied when there are multiple comma-separated
    // terms. Single-term queries trust the API results as-is.
    // Model IDs are normalized the same way as the query (slash → space) so that
    // mixed queries like "mistralai/3B,12" match "mistralai/Ministral-3-3B...".
    let has_multi_term = filter_terms.len() > 1;
    let filtered: Vec<&discover::SearchResult> = results
        .iter()
        .filter(|r| {
            if !has_multi_term {
                return true;
            }
            let id_normalized = r.model_id.replace('/', " ").to_lowercase();
            filter_terms
                .iter()
                .all(|term| id_normalized.contains(term.as_str())) // BORROW: explicit .as_str()
        })
        .take(limit)
        .collect();

    if exact {
        // Exact match: compare against the original query (not normalized)
        let exact_match = filtered
            .iter()
            .find(|r| r.model_id.eq_ignore_ascii_case(query));

        if let Some(matched) = exact_match {
            println!("Exact match:\n");
            print_search_result(matched);

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
                for result in &filtered {
                    print_search_result(result);
                }
            }
        }
    } else {
        // Normal search display
        if filtered.is_empty() {
            println!("No models found matching \"{query}\".");
        } else {
            println!("Models matching \"{query}\" (by downloads):\n");
            for result in &filtered {
                print_search_result(result);
            }
        }
    }

    Ok(())
}

fn print_search_result(result: &discover::SearchResult) {
    let suffix = match (&result.library_name, &result.pipeline_tag) {
        (Some(lib), Some(pipe)) => format!("  [{lib}, {pipe}]"),
        (Some(lib), None) => format!("  [{lib}]"),
        (None, Some(pipe)) => format!("  [{pipe}]"),
        (None, None) => String::new(),
    };
    println!(
        "  hf-fm {:<48} ({} downloads){suffix}",
        result.model_id,
        format_downloads(result.downloads)
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
    println!(
        "  {:<48} {:>5}  {:>10}  Status",
        "Repository", "Files", "Size"
    );
    println!("  {:-<48} {:-<5}  {:-<10}  {:-<8}", "", "", "", "");

    for s in &summaries {
        let status_label = if s.has_partial { "PARTIAL" } else { "ok" };
        println!(
            "  {:<48} {:>5}  {:>10}  {}",
            s.repo_id,
            s.file_count,
            format_size(s.total_size),
            status_label
        );
    }

    println!("\n{} model(s) cached", summaries.len());

    Ok(())
}

/// Shows disk usage summary for all cached repos, sorted by size descending.
fn run_du() -> Result<(), FetchError> {
    let mut summaries = cache::cache_summary()?;

    if summaries.is_empty() {
        println!("No models found in local cache.");
        return Ok(());
    }

    summaries.sort_by(|a, b| b.total_size.cmp(&a.total_size));

    let mut total_size: u64 = 0;
    let mut total_files: usize = 0;

    for s in &summaries {
        total_size = total_size.saturating_add(s.total_size);
        total_files = total_files.saturating_add(s.file_count);

        let partial_marker = if s.has_partial { "  PARTIAL" } else { "" };
        println!(
            "  {:>10}  {:<48} ({} files){}",
            format_size(s.total_size),
            s.repo_id,
            s.file_count,
            partial_marker,
        );
    }

    println!("  {}", "\u{2500}".repeat(50),);
    println!(
        "  {:>10}  total ({} repos, {} files)",
        format_size(total_size),
        summaries.len(),
        total_files,
    );

    Ok(())
}

/// Shows per-file disk usage for a specific cached repo, sorted by size descending.
fn run_du_repo(repo_id: &str) -> Result<(), FetchError> {
    let files = cache::cache_repo_usage(repo_id)?;

    if files.is_empty() {
        println!("No cached files found for {repo_id}.");
        return Ok(());
    }

    let mut total_size: u64 = 0;

    for f in &files {
        total_size = total_size.saturating_add(f.size);
        println!("  {:>10}  {}", format_size(f.size), f.filename);
    }

    println!("  {}", "\u{2500}".repeat(50),);
    println!(
        "  {:>10}  total ({} files)",
        format_size(total_size),
        files.len(),
    );

    Ok(())
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
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
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

    // Collect all tensor names from both repos.
    let mut all_names: Vec<&str> = tensors_a
        .keys()
        .chain(tensors_b.keys())
        // BORROW: explicit .as_str() instead of Deref coercion
        .map(String::as_str)
        .collect::<HashSet<&str>>()
        .into_iter()
        .collect();
    all_names.sort_unstable();

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
                    println!("    {name:<50} {:<8} {shape_str}", t.dtype);
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
                    println!("    {name:<50} {:<8} {shape_str}", t.dtype);
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
        "  A: {} tensors | B: {} tensors | only-A: {} | only-B: {} | differ: {} | match: {}",
        total_a,
        total_b,
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

/// Inspects `.safetensors` file headers for tensor metadata.
#[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments)]
fn run_inspect(
    repo_id: &str,
    filename: Option<&str>,
    revision: Option<&str>,
    token: Option<&str>,
    cached: bool,
    no_metadata: bool,
    json: bool,
    filter: Option<&str>,
) -> Result<(), FetchError> {
    match filename {
        Some(f) => run_inspect_single(
            repo_id,
            f,
            revision,
            token,
            cached,
            no_metadata,
            json,
            filter,
        ),
        None => run_inspect_repo(repo_id, revision, token, cached, json, filter),
    }
}

/// Inspects a single `.safetensors` file and prints the result.
#[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments)]
fn run_inspect_single(
    repo_id: &str,
    filename: &str,
    revision: Option<&str>,
    token: Option<&str>,
    cached: bool,
    no_metadata: bool,
    json: bool,
    filter: Option<&str>,
) -> Result<(), FetchError> {
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

    if json {
        let output = serde_json::to_string_pretty(&info)
            .map_err(|e| FetchError::Http(format!("failed to serialize JSON: {e}")))?;
        println!("{output}");
        return Ok(());
    }

    // Human-readable output.
    let source_label = match source {
        inspect::InspectSource::Cached => "cached",
        inspect::InspectSource::Remote => "remote (2 HTTP requests)",
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

    println!();
    println!(
        "  {:<50} {:<8} {:<16} {:>10} {:>10}",
        "Tensor", "Dtype", "Shape", "Size", "Params"
    );

    for t in &info.tensors {
        let shape_str = format!("{:?}", t.shape);
        let size_str = format_size(t.byte_len());
        let params_str = inspect::format_params(t.num_elements());
        println!(
            "  {:<50} {:<8} {:<16} {:>10} {:>10}",
            t.name, t.dtype, shape_str, size_str, params_str
        );
    }

    println!("  {}", "\u{2500}".repeat(96));
    let filtered_count = info.tensors.len();
    let filtered_params = info.total_params();
    let tensor_label = if filtered_count == 1 {
        "tensor"
    } else {
        "tensors"
    };

    if filter.is_some() {
        println!(
            "  {filtered_count}/{total_tensor_count} {tensor_label}, {}/{} params (filter: {:?})",
            inspect::format_params(filtered_params),
            inspect::format_params(total_params),
            filter.unwrap_or_default(),
        );
    } else {
        println!(
            "  {filtered_count} {tensor_label}, {} params",
            inspect::format_params(filtered_params)
        );
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

    println!("  {:<48} {:>8}", "File", "Tensors");

    for shard in &index.shards {
        let count = by_shard.get(shard).copied().unwrap_or(0);
        if filter.is_some() && count == 0 {
            continue;
        }
        println!("  {shard:<48} {count:>8}");
    }

    println!("  {}", "\u{2500}".repeat(58));

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
        println!("  {displayed_shards} {shard_label}, {filtered_total} {tensor_label}",);
    }
    println!("  Hint: use `hf-fm inspect {repo_id} <filename>` for per-tensor detail");
}

/// Prints multi-file inspection results as JSON, optionally filtering tensors.
fn print_multi_file_json(
    results: &[(String, inspect::SafetensorsHeaderInfo)],
    filter: Option<&str>,
) -> Result<(), FetchError> {
    if let Some(pattern) = filter {
        // Filter tensors in each file before serializing.
        let filtered: Vec<(String, inspect::SafetensorsHeaderInfo)> = results
            .iter()
            .map(|(name, info)| {
                let mut filtered_info = info.clone();
                // BORROW: explicit .as_str() instead of Deref coercion
                filtered_info
                    .tensors
                    .retain(|t| t.name.as_str().contains(pattern));
                // BORROW: explicit .clone() for owned String
                (name.clone(), filtered_info)
            })
            .filter(|(_, info)| !info.tensors.is_empty())
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
    println!("  {:<48} {:>8} {:>12}", "File", "Tensors", "Params");

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
            "  {name:<48} {tensor_count:>8} {:>12}",
            inspect::format_params(params)
        );
    }

    println!("  {}", "\u{2500}".repeat(70));
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
    for (filename, file_status) in &status.files {
        match file_status {
            cache::FileStatus::Complete { local_size } => {
                println!(
                    "  {:<48} {:>10}  complete",
                    filename,
                    format_size(*local_size)
                );
            }
            cache::FileStatus::Partial {
                local_size,
                expected_size,
            } => {
                println!(
                    "  {:<48} {:>10} / {:<10}  PARTIAL",
                    filename,
                    format_size(*local_size),
                    format_size(*expected_size)
                );
            }
            cache::FileStatus::Missing { expected_size } => {
                if *expected_size > 0 {
                    println!(
                        "  {:<48} {:>10}  MISSING",
                        filename,
                        format_size(*expected_size)
                    );
                } else {
                    println!("  {filename:<48}          —  MISSING");
                }
            }
            // EXPLICIT: future FileStatus variants display as UNKNOWN
            _ => {
                println!("  {filename:<48}                UNKNOWN");
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
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
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
    let files = rt.block_on(repo::list_repo_files_with_metadata(
        repo_id,
        resolved_token.as_deref(),
        revision,
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
        let repo_folder = format!("models--{}", repo_id.replace('/', "--"));
        let repo_dir = cache_dir.join(&repo_folder);
        let revision_str = revision.unwrap_or("main");
        let commit_hash = cache::read_ref(&repo_dir, revision_str);
        let snapshot_dir = commit_hash.map(|h| repo_dir.join("snapshots").join(h));

        filtered
            .iter()
            .map(|f| {
                let local_path = snapshot_dir
                    .as_ref()
                    // BORROW: explicit .as_str() instead of Deref coercion
                    .map(|dir| dir.join(f.filename.as_str()));
                match local_path {
                    Some(ref path) if path.exists() => {
                        let local_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
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

    // Print table header.
    if no_checksum {
        if show_cached {
            println!("  {:<48} {:>10}  Cached", "File", "Size");
            println!("  {:<48} {:>10}  {:-<6}", "", "", "");
        } else {
            println!("  {:<48} {:>10}", "File", "Size");
            println!("  {:<48} {:>10}", "", "");
        }
    } else if show_cached {
        println!("  {:<48} {:>10}  {:<12}  Cached", "File", "Size", "SHA256");
        println!("  {:<48} {:>10}  {:<12}  {:-<6}", "", "", "", "");
    } else {
        println!("  {:<48} {:>10}  {:<12}", "File", "Size", "SHA256");
        println!("  {:<48} {:>10}  {:<12}", "", "", "");
    }

    // Print each file row.
    let mut total_bytes: u64 = 0;
    let mut cached_count: usize = 0;

    for (i, f) in filtered.iter().enumerate() {
        let size = f.size.unwrap_or(0);
        total_bytes = total_bytes.saturating_add(size);

        let size_str = format_size(size);
        let sha_str = if no_checksum {
            String::new()
        } else {
            f.sha256
                .as_deref()
                .and_then(|s| s.get(..12))
                .unwrap_or("\u{2014}")
                .to_owned()
        };

        if show_cached {
            let mark = cache_marks.get(i).map_or("\u{2717}", String::as_str);
            if mark == "\u{2713}" {
                cached_count += 1;
            }
            if no_checksum {
                println!("  {:<48} {:>10}  {mark}", f.filename, size_str);
            } else {
                println!(
                    "  {:<48} {:>10}  {:<12}  {mark}",
                    f.filename, size_str, sha_str
                );
            }
        } else if no_checksum {
            println!("  {:<48} {:>10}", f.filename, size_str);
        } else {
            println!("  {:<48} {:>10}  {sha_str}", f.filename, size_str);
        }
    }

    // Summary line.
    let count = filtered.len();
    println!("  {:\u{2500}<72}", "");
    if show_cached {
        println!(
            "  {count} files, {} total ({cached_count} cached)",
            format_size(total_bytes)
        );
    } else {
        println!("  {count} files, {} total", format_size(total_bytes));
    }

    Ok(())
}

/// Formats a byte size with human-readable suffixes (B, KiB, MiB, GiB).
fn format_size(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * 1024 * 1024;

    if bytes >= GIB {
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
        std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
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
