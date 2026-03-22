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
use hf_fetch_model::progress::IndicatifProgress;
use hf_fetch_model::repo;
use hf_fetch_model::{compile_glob_patterns, file_matches, FetchConfig, FetchError, Filter};

/// Downloads all files from a `HuggingFace` model repository.
///
/// Use `--preset safetensors` to download only safetensors weights,
/// config, and tokenizer files.
#[derive(Parser)]
#[command(name = "hf-fetch-model", version, about)]
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
    },
    /// Download a single file from a `HuggingFace` repository.
    DownloadFile {
        /// Enable verbose output (download diagnostics).
        #[arg(short, long)]
        verbose: bool,

        /// The repository identifier (e.g., "mntss/clt-gemma-2-2b-426k").
        repo_id: String,

        /// Exact filename within the repository (e.g., `"W_dec_0.safetensors"`).
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
        /// Filter preset (`safetensors`, `gguf`, `config-only`).
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
            | Commands::Status { .. }
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
        // BORROW: explicit .as_str() for String → &str conversion
        Some(Commands::Search {
            query,
            limit,
            exact,
        }) => run_search(query.as_str(), limit, exact),
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
        }) => run_download_file(
            repo_id.as_str(),
            filename.as_str(),
            revision.as_deref(),
            token.as_deref(),
            output_dir,
            chunk_threshold_mib,
            connections_per_file,
        ),
        Some(Commands::Status {
            repo_id: Some(repo_id),
            revision,
            token,
            // BORROW: explicit .as_str()/.as_deref() for owned → borrowed conversions
        }) => run_status(repo_id.as_str(), revision.as_deref(), token.as_deref()),
        Some(Commands::Status { repo_id: None, .. }) => run_status_all(),
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

    // Build FetchConfig from CLI args.
    let mut builder = match args.preset {
        Some(Preset::Safetensors) => Filter::safetensors(),
        Some(Preset::Gguf) => Filter::gguf(),
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
    if let Some(dir) = args.output_dir {
        builder = builder.output_dir(dir);
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
    Ok(())
}

/// Displays a download plan without downloading anything.
fn run_dry_run(repo_id: &str, args: &DownloadArgs) -> Result<(), FetchError> {
    // Build FetchConfig from CLI args (same builder logic, minus on_progress).
    let mut builder = match args.preset {
        Some(Preset::Safetensors) => Filter::safetensors(),
        Some(Preset::Gguf) => Filter::gguf(),
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

fn run_download_file(
    repo_id: &str,
    filename: &str,
    revision: Option<&str>,
    token: Option<&str>,
    output_dir: Option<PathBuf>,
    chunk_threshold_mib: Option<u64>,
    connections_per_file: Option<usize>,
) -> Result<(), FetchError> {
    if !repo_id.contains('/') {
        return Err(FetchError::InvalidArgument(format!(
            "invalid REPO_ID \"{repo_id}\": expected \"org/model\" format (e.g., \"mntss/clt-gemma-2-2b-426k\")"
        )));
    }

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
    if let Some(dir) = output_dir {
        builder = builder.output_dir(dir);
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

    if outcome.is_cached() {
        println!("Cached at: {}", outcome.inner().display());
    } else {
        println!("Downloaded to: {}", outcome.inner().display());
        print_download_summary(outcome.inner(), elapsed);
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

fn run_search(query: &str, limit: usize, exact: bool) -> Result<(), FetchError> {
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
    let api_limit = if filter_terms.len() > 1 {
        limit.saturating_mul(5)
    } else {
        limit
    };

    let results = rt.block_on(discover::search_models(api_query, api_limit))?;

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
    println!(
        "  hf-fm {:<48} ({} downloads)",
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

/// Warns when `--filter` globs are redundant with the active `--preset`.
fn warn_redundant_filters(preset: &Preset, filters: &[String]) {
    let (preset_globs, preset_name): (&[&str], &str) = match preset {
        Preset::Safetensors => (&["*.safetensors", "*.json", "*.txt"], "safetensors"),
        Preset::Gguf => (&["*.gguf", "*.json", "*.txt"], "gguf"),
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
