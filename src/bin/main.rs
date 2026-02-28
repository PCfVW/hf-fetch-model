// SPDX-License-Identifier: MIT OR Apache-2.0

//! CLI binary for hf-fetch-model.
//!
//! Installed as both `hf-fetch-model` and `hf-fm`.

use std::collections::HashSet;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand, ValueEnum};

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
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
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
    if let Some(dir) = args.output_dir {
        builder = builder.output_dir(dir);
    }

    // Set up indicatif progress bars.
    let progress = IndicatifProgress::new();
    builder = builder.on_progress(move |e| progress.handle(e));

    let config = builder.build()?;

    // Run the download using a new Tokio runtime.
    let rt = tokio::runtime::Runtime::new().map_err(|e| FetchError::Io {
        path: PathBuf::from("<runtime>"),
        source: e,
    })?;

    let path = rt.block_on(hf_fetch_model::download_with_config(repo_id, &config))?;

    println!("Downloaded to: {}", path.display());
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
