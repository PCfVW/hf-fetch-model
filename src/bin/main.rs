// SPDX-License-Identifier: MIT OR Apache-2.0

//! CLI binary for hf-fetch-model.
//!
//! Installed as both `hf-fetch-model` and `hf-fm`.

use std::collections::HashSet;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use clap::{Args, Parser, Subcommand, ValueEnum};
use tracing_subscriber::EnvFilter;

use hf_fetch_model::cache;
use hf_fetch_model::discover;
use hf_fetch_model::progress::IndicatifProgress;
use hf_fetch_model::{FetchConfig, FetchError, Filter};

/// Fast `HuggingFace` model downloads.
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

    /// Number of concurrent file downloads.
    #[arg(long, default_value = "4")]
    concurrency: usize,

    /// Minimum file size (MiB) for parallel chunked download.
    #[arg(long, default_value = "100")]
    chunk_threshold_mib: u64,

    /// Number of parallel HTTP connections per large file.
    #[arg(long, default_value = "8")]
    connections_per_file: usize,
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

        /// Minimum file size (MiB) for parallel chunked download.
        #[arg(long, default_value = "100")]
        chunk_threshold_mib: u64,

        /// Number of parallel HTTP connections per large file.
        #[arg(long, default_value = "8")]
        connections_per_file: usize,
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
            | Commands::Status { .. },
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
        None => run_download(cli.download),
    }
}

fn run_download(args: DownloadArgs) -> Result<(), FetchError> {
    let repo_id = args.repo_id.ok_or_else(|| {
        FetchError::InvalidArgument(
            "REPO_ID is required for download. Usage: hf-fm <REPO_ID>".to_owned(),
        )
    })?;

    if !repo_id.contains('/') {
        return Err(FetchError::InvalidArgument(format!(
            "invalid REPO_ID \"{repo_id}\": expected \"org/model\" format (e.g., \"EleutherAI/pythia-1.4b\")"
        )));
    }

    // Build FetchConfig from CLI args.
    let mut builder = match args.preset {
        Some(Preset::Safetensors) => Filter::safetensors(),
        Some(Preset::Gguf) => Filter::gguf(),
        Some(Preset::ConfigOnly) => Filter::config_only(),
        None => FetchConfig::builder(),
    };

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
    builder = builder.concurrency(args.concurrency);
    builder = builder.chunk_threshold(args.chunk_threshold_mib.saturating_mul(1024 * 1024));
    builder = builder.connections_per_file(args.connections_per_file);
    if let Some(dir) = args.output_dir {
        builder = builder.output_dir(dir);
    }

    // Set up indicatif progress bars.
    // Shared via Arc so we can call finish() before printing the result.
    let progress = Arc::new(IndicatifProgress::new());
    let progress_handle = Arc::clone(&progress);
    builder = builder.on_progress(move |e| progress_handle.handle(e));

    let config = builder.build()?;

    // Run the download using a new Tokio runtime.
    let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
        path: PathBuf::from("<runtime>"),
        source: e,
    })?;

    let outcome = rt.block_on(hf_fetch_model::download_with_config(repo_id, &config))?;

    // Finalize progress bar before printing to avoid interleaved output.
    progress.finish();

    if outcome.is_cached() {
        println!("Cached at: {}", outcome.inner().display());
    } else {
        println!("Downloaded to: {}", outcome.inner().display());
    }
    Ok(())
}

fn run_download_file(
    repo_id: &str,
    filename: &str,
    revision: Option<&str>,
    token: Option<&str>,
    output_dir: Option<PathBuf>,
    chunk_threshold_mib: u64,
    connections_per_file: usize,
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
    builder = builder.chunk_threshold(chunk_threshold_mib.saturating_mul(1024 * 1024));
    builder = builder.connections_per_file(connections_per_file);
    if let Some(dir) = output_dir {
        builder = builder.output_dir(dir);
    }

    // Set up indicatif progress bars.
    let progress = Arc::new(IndicatifProgress::new());
    let progress_handle = Arc::clone(&progress);
    builder = builder.on_progress(move |e| progress_handle.handle(e));

    let config = builder.build()?;

    // Run the download using a new Tokio runtime.
    let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
        path: PathBuf::from("<runtime>"),
        source: e,
    })?;

    // BORROW: explicit .to_owned() for &str → owned String
    let outcome = rt.block_on(hf_fetch_model::download_file(
        repo_id.to_owned(),
        filename,
        &config,
    ))?;

    // Finalize progress bar before printing to avoid interleaved output.
    progress.finish();

    if outcome.is_cached() {
        println!("Cached at: {}", outcome.inner().display());
    } else {
        println!("Downloaded to: {}", outcome.inner().display());
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

/// Formats a download count with K/M/B suffixes for readability.
fn format_downloads(n: u64) -> String {
    if n >= 1_000_000_000 {
        // CAST: u64 → f64, precision loss acceptable; value is a display-only download count
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let val = n as f64 / 1_000_000_000.0;
        format!("{val:.1}B")
    } else if n >= 1_000_000 {
        // CAST: u64 → f64, precision loss acceptable; value is a display-only download count
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let val = n as f64 / 1_000_000.0;
        format!("{val:.1}M")
    } else if n >= 1_000 {
        // CAST: u64 → f64, precision loss acceptable; value is a display-only download count
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let val = n as f64 / 1_000.0;
        format!("{val:.1}K")
    } else {
        n.to_string()
    }
}
