// SPDX-License-Identifier: MIT OR Apache-2.0

//! CLI integration tests for `hf-fetch-model`.
//!
//! These tests exercise the binary output directly using `std::process::Command`.
//! They require the `cli` feature to compile (the binary needs `clap`).
//! Network tests use `julien-c/dummy-unknown`, a tiny public `HuggingFace` Hub repo.
//! Run with: `cargo test --all-features`

#![cfg(feature = "cli")]
#![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]

use std::process::Command;

/// Builds a `Command` targeting the `hf-fetch-model` binary.
fn hf_fm() -> Command {
    Command::new(env!("CARGO_BIN_EXE_hf-fetch-model"))
}

/// Runs a command and returns `(stdout, stderr, success)`.
fn run(cmd: &mut Command) -> (String, String, bool) {
    let output = cmd
        .output()
        .expect("failed to execute hf-fetch-model binary");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    (stdout, stderr, output.status.success())
}

// -----------------------------------------------------------------------
// Help text (offline — no network needed)
// -----------------------------------------------------------------------

#[test]
fn help_shows_download_description() {
    let (stdout, stderr, success) = run(hf_fm().arg("--help"));
    assert!(success, "--help failed: {stderr}");
    assert!(
        stdout.contains("Downloads all files"),
        "help should describe default download behavior, got:\n{stdout}"
    );
}

#[test]
fn help_shows_list_files_subcommand() {
    let (stdout, stderr, success) = run(hf_fm().arg("--help"));
    assert!(success, "--help failed: {stderr}");
    assert!(
        stdout.contains("list-files"),
        "help should list the list-files subcommand, got:\n{stdout}"
    );
}

#[test]
fn help_shows_version_number() {
    let (stdout, stderr, success) = run(hf_fm().arg("--help"));
    assert!(success, "--help failed: {stderr}");
    let version = env!("CARGO_PKG_VERSION");
    assert!(
        stdout.contains(version),
        "help should contain version {version}, got:\n{stdout}"
    );
}

#[test]
fn help_shows_pth_preset() {
    let (stdout, stderr, success) = run(hf_fm().arg("--help"));
    assert!(success, "--help failed: {stderr}");
    assert!(
        stdout.contains("pth"),
        "help should mention pth preset, got:\n{stdout}"
    );
}

#[test]
fn help_shows_dry_run_flag() {
    let (stdout, stderr, success) = run(hf_fm().arg("--help"));
    assert!(success, "--help failed: {stderr}");
    assert!(
        stdout.contains("--dry-run"),
        "help should list the --dry-run flag, got:\n{stdout}"
    );
}

#[test]
fn help_shows_flat_flag() {
    let (stdout, stderr, success) = run(hf_fm().arg("--help"));
    assert!(success, "--help failed: {stderr}");
    assert!(
        stdout.contains("--flat"),
        "help should show --flat flag, got:\n{stdout}"
    );
}

#[test]
fn download_file_help_shows_flat_flag() {
    let (stdout, stderr, success) = run(hf_fm().args(["download-file", "--help"]));
    assert!(success, "download-file --help failed: {stderr}");
    assert!(
        stdout.contains("--flat"),
        "download-file help should show --flat flag, got:\n{stdout}"
    );
}

#[test]
fn download_file_help_mentions_glob() {
    let (stdout, stderr, success) = run(hf_fm().args(["download-file", "--help"]));
    assert!(success, "download-file --help failed: {stderr}");
    assert!(
        stdout.contains("glob"),
        "download-file help should mention glob support, got:\n{stdout}"
    );
}

#[test]
fn help_shows_timeout_flags() {
    let (stdout, stderr, success) = run(hf_fm().arg("--help"));
    assert!(success, "--help failed: {stderr}");
    for flag in ["--timeout-per-file-secs", "--timeout-total-secs"] {
        assert!(
            stdout.contains(flag),
            "help should show {flag} flag, got:\n{stdout}"
        );
    }
}

#[test]
fn download_file_help_shows_timeout_flags() {
    let (stdout, stderr, success) = run(hf_fm().args(["download-file", "--help"]));
    assert!(success, "download-file --help failed: {stderr}");
    for flag in ["--timeout-per-file-secs", "--timeout-total-secs"] {
        assert!(
            stdout.contains(flag),
            "download-file help should show {flag} flag, got:\n{stdout}"
        );
    }
}

#[test]
fn timeout_flags_accept_numeric_values() {
    // Sanity check: clap should parse the values without erroring.
    // We pass --dry-run so no network call is attempted.
    let (_stdout, stderr, _success) = run(hf_fm().args([
        "--timeout-per-file-secs",
        "1800",
        "--timeout-total-secs",
        "3600",
        "--dry-run",
        "julien-c/dummy-unknown",
    ]));
    // We don't require success here — the dry-run still hits the API to list
    // files and may fail offline. We only require that clap accepted the
    // flags (no "error: invalid value" parse failure).
    assert!(
        !stderr.contains("error: invalid value"),
        "timeout flags should parse cleanly, got:\n{stderr}"
    );
    assert!(
        !stderr.contains("error: unexpected argument"),
        "timeout flags should be recognized, got:\n{stderr}"
    );
}

#[test]
fn list_files_help_shows_all_flags() {
    let (stdout, stderr, success) = run(hf_fm().args(["list-files", "--help"]));
    assert!(success, "list-files --help failed: {stderr}");
    for flag in ["--no-checksum", "--show-cached", "--filter", "--preset"] {
        assert!(
            stdout.contains(flag),
            "list-files help should contain {flag}, got:\n{stdout}"
        );
    }
    // --show-cached help should describe the three cache states.
    assert!(
        stdout.contains("partial"),
        "list-files help should describe partial state, got:\n{stdout}"
    );
}

// -----------------------------------------------------------------------
// Error handling (offline or fast-fail)
// -----------------------------------------------------------------------

#[test]
fn list_files_invalid_repo_format() {
    let (_stdout, stderr, success) = run(hf_fm().args(["list-files", "noSlash"]));
    assert!(!success, "list-files with invalid repo should fail");
    assert!(
        stderr.contains("org/model"),
        "error should mention expected format, got:\n{stderr}"
    );
}

#[test]
fn dry_run_invalid_repo_format() {
    let (_stdout, stderr, success) = run(hf_fm().args(["noSlash", "--dry-run"]));
    assert!(!success, "--dry-run with invalid repo should fail");
    assert!(
        stderr.contains("org/model"),
        "error should mention expected format, got:\n{stderr}"
    );
}

#[test]
fn list_files_nonexistent_repo() {
    let (_stdout, stderr, success) =
        run(hf_fm().args(["list-files", "fake/nonexistent-repo-12345"]));
    assert!(!success, "list-files with nonexistent repo should fail");
    // CI environments without HF_TOKEN may get 401 instead of 404.
    assert!(
        stderr.contains("not found") || stderr.contains("401") || stderr.contains("Unauthorized"),
        "error should indicate repo inaccessible, got:\n{stderr}"
    );
}

// -----------------------------------------------------------------------
// list-files (network — uses julien-c/dummy-unknown)
// -----------------------------------------------------------------------

#[test]
fn list_files_default_output() {
    let (stdout, stderr, success) = run(hf_fm().args(["list-files", "julien-c/dummy-unknown"]));
    assert!(success, "list-files failed: {stderr}");

    // Should contain known filenames.
    assert!(
        stdout.contains("config.json"),
        "output should contain config.json, got:\n{stdout}"
    );
    assert!(
        stdout.contains("pytorch_model.bin"),
        "output should contain pytorch_model.bin, got:\n{stdout}"
    );

    // Should contain summary line with file count and "total".
    assert!(
        stdout.contains("files") && stdout.contains("total"),
        "output should contain summary with file count and total, got:\n{stdout}"
    );

    // Should contain SHA256 header by default.
    assert!(
        stdout.contains("SHA256"),
        "default output should contain SHA256 header, got:\n{stdout}"
    );
}

#[test]
fn list_files_no_checksum_hides_sha256() {
    let (stdout, stderr, success) =
        run(hf_fm().args(["list-files", "julien-c/dummy-unknown", "--no-checksum"]));
    assert!(success, "list-files --no-checksum failed: {stderr}");
    assert!(
        !stdout.contains("SHA256"),
        "--no-checksum should hide SHA256 header, got:\n{stdout}"
    );
}

#[test]
fn list_files_show_cached_adds_column() {
    let (stdout, stderr, success) =
        run(hf_fm().args(["list-files", "julien-c/dummy-unknown", "--show-cached"]));
    assert!(success, "list-files --show-cached failed: {stderr}");
    assert!(
        stdout.contains("Cached"),
        "--show-cached should add Cached header, got:\n{stdout}"
    );
}

#[test]
fn list_files_show_cached_marks_complete_files() {
    // julien-c/dummy-unknown should be fully cached from other tests.
    // Complete files must show ✓, never "partial".
    let (stdout, stderr, success) =
        run(hf_fm().args(["list-files", "julien-c/dummy-unknown", "--show-cached"]));
    assert!(success, "list-files --show-cached failed: {stderr}");
    assert!(
        stdout.contains('\u{2713}'),
        "cached files should show \u{2713} mark, got:\n{stdout}"
    );
    assert!(
        !stdout.contains("partial"),
        "fully cached files should not show 'partial', got:\n{stdout}"
    );
    // Summary should report cached count.
    assert!(
        stdout.contains("cached"),
        "summary should mention cached count, got:\n{stdout}"
    );
}

#[test]
fn list_files_filter_limits_output() {
    let (stdout, stderr, success) =
        run(hf_fm().args(["list-files", "julien-c/dummy-unknown", "--filter", "*.json"]));
    assert!(success, "list-files --filter failed: {stderr}");
    assert!(
        stdout.contains("config.json"),
        "filtered output should contain config.json, got:\n{stdout}"
    );
    assert!(
        !stdout.contains("pytorch_model.bin"),
        "filtered output should NOT contain pytorch_model.bin, got:\n{stdout}"
    );
}

// -----------------------------------------------------------------------
// --dry-run (network — uses julien-c/dummy-unknown, likely cached)
// -----------------------------------------------------------------------

#[test]
fn dry_run_shows_repo_and_revision() {
    let (stdout, stderr, success) = run(hf_fm().args(["julien-c/dummy-unknown", "--dry-run"]));
    assert!(success, "--dry-run failed: {stderr}");
    assert!(
        stdout.contains("Repo:"),
        "dry-run should show Repo: header, got:\n{stdout}"
    );
    assert!(
        stdout.contains("Revision:"),
        "dry-run should show Revision: header, got:\n{stdout}"
    );
}

#[test]
fn dry_run_cached_repo_shows_zero_download() {
    // julien-c/dummy-unknown should already be cached from other tests.
    let (stdout, stderr, success) = run(hf_fm().args(["julien-c/dummy-unknown", "--dry-run"]));
    assert!(success, "--dry-run failed: {stderr}");
    assert!(
        stdout.contains("0 to download"),
        "cached repo should show 0 to download, got:\n{stdout}"
    );
    assert!(
        stdout.contains("Download: 0 B"),
        "cached repo should show Download: 0 B, got:\n{stdout}"
    );
}

#[test]
fn dry_run_no_astronomical_chunk_threshold() {
    // Regression test: chunk threshold must not display a galactic number.
    // Run against a repo with mixed file sizes to exercise the optimizer.
    let (stdout, stderr, success) = run(hf_fm().args([
        "mistralai/Ministral-3-3B-Instruct-2512",
        "--preset",
        "safetensors",
        "--dry-run",
    ]));
    assert!(success, "--dry-run failed: {stderr}");

    // Find the chunk threshold line and verify the number is sane.
    for line in stdout.lines() {
        if line.contains("chunk threshold") {
            // Either "disabled" or a number ≤ 10000 MiB (10 GiB).
            let is_disabled = line.contains("disabled");
            let has_sane_number = line
                .split_whitespace()
                .filter_map(|w| w.parse::<u64>().ok())
                .all(|n| n <= 10_000);
            assert!(
                is_disabled || has_sane_number,
                "chunk threshold should be sane or disabled, got:\n{line}"
            );
        }
    }
}

#[test]
fn dry_run_with_filter_shows_filter_info() {
    let (stdout, stderr, success) =
        run(hf_fm().args(["julien-c/dummy-unknown", "--dry-run", "--filter", "*.json"]));
    assert!(success, "--dry-run --filter failed: {stderr}");
    assert!(
        stdout.contains("Filter:"),
        "filtered dry-run should show Filter: line, got:\n{stdout}"
    );
    // Should only show JSON files.
    assert!(
        stdout.contains("config.json"),
        "filtered dry-run should contain config.json, got:\n{stdout}"
    );
    assert!(
        !stdout.contains("pytorch_model.bin"),
        "filtered dry-run should NOT contain pytorch_model.bin, got:\n{stdout}"
    );
}

// -----------------------------------------------------------------------
// du subcommand
// -----------------------------------------------------------------------

#[test]
fn cache_delete_help_shows_flags() {
    let (stdout, stderr, success) = run(hf_fm().args(["cache", "delete", "--help"]));
    assert!(success, "cache delete --help failed: {stderr}");
    assert!(
        stdout.contains("--yes"),
        "cache delete help should contain --yes, got:\n{stdout}"
    );
}

#[test]
fn cache_delete_nonexistent_repo() {
    let (_, stderr, success) = run(hf_fm().args([
        "cache",
        "delete",
        "nonexistent-org/nonexistent-model-xyz",
        "--yes",
    ]));
    assert!(!success, "cache delete of missing repo should fail");
    assert!(
        stderr.contains("not cached"),
        "cache delete should report not cached, got:\n{stderr}"
    );
}

#[test]
fn help_shows_cache_delete() {
    let (stdout, stderr, success) = run(hf_fm().args(["cache", "--help"]));
    assert!(success, "cache --help failed: {stderr}");
    assert!(
        stdout.contains("delete"),
        "cache help should mention delete subcommand, got:\n{stdout}"
    );
}

#[test]
fn help_shows_cache_subcommand() {
    let (stdout, _stderr, success) = run(hf_fm().arg("--help"));
    assert!(success, "help should succeed");
    assert!(
        stdout.contains("cache"),
        "help should mention cache subcommand, got:\n{stdout}"
    );
}

#[test]
fn cache_clean_partial_help_shows_flags() {
    let (stdout, stderr, success) = run(hf_fm().args(["cache", "clean-partial", "--help"]));
    assert!(success, "cache clean-partial --help failed: {stderr}");
    for flag in ["--yes", "--dry-run"] {
        assert!(
            stdout.contains(flag),
            "cache clean-partial help should contain {flag}, got:\n{stdout}"
        );
    }
}

#[test]
fn cache_clean_partial_dry_run_is_nondestructive() {
    // --dry-run, NOT --yes: this test runs against the developer's real
    // HuggingFace cache, and a --yes sweep silently destroys real resume
    // state for any download interrupted on this machine (it wiped two
    // deliberately-staged partials during the 2026-06-12 cache-tutorial
    // capture session). The dry run exercises the same scan and renders
    // either the no-partials notice or the removal preview.
    let (stdout, stderr, success) = run(hf_fm().args(["cache", "clean-partial", "--dry-run"]));
    assert!(success, "cache clean-partial should succeed: {stderr}");
    assert!(
        stdout.contains("No partial downloads found")
            || stdout.contains("No HuggingFace cache found")
            || stdout.contains("partial download"),
        "cache clean-partial --dry-run should report the cache's partial state, got:\n{stdout}"
    );
}

#[test]
fn cache_gc_help_shows_flags() {
    let (stdout, stderr, success) = run(hf_fm().args(["cache", "gc", "--help"]));
    assert!(success, "cache gc --help failed: {stderr}");
    for flag in [
        "--older-than",
        "--max-size",
        "--except",
        "--dry-run",
        "--yes",
        "--list-kept",
    ] {
        assert!(
            stdout.contains(flag),
            "cache gc help should contain {flag}, got:\n{stdout}"
        );
    }
}

#[test]
fn help_shows_cache_gc() {
    let (stdout, stderr, success) = run(hf_fm().args(["cache", "--help"]));
    assert!(success, "cache --help failed: {stderr}");
    assert!(
        stdout.contains("gc"),
        "cache help should mention gc subcommand, got:\n{stdout}"
    );
}

#[test]
fn cache_gc_requires_strategy() {
    let (_, stderr, success) = run(hf_fm().args(["cache", "gc"]));
    assert!(!success, "bare `cache gc` should be rejected by clap");
    // clap's ArgGroup error mentions both required-arg names.
    assert!(
        stderr.contains("--older-than") && stderr.contains("--max-size"),
        "cache gc with no flags should mention --older-than and --max-size, got:\n{stderr}"
    );
}

#[test]
fn cache_gc_rejects_decimal_size() {
    let (_, stderr, success) = run(hf_fm().args(["cache", "gc", "--max-size", "5GB", "--dry-run"]));
    assert!(!success, "--max-size 5GB should be rejected");
    assert!(
        stderr.contains("decimal size unit") && stderr.contains("binary units"),
        "cache gc should reject decimal units with helpful error, got:\n{stderr}"
    );
}

#[test]
fn cache_gc_dry_run_no_matches() {
    // 99999 days ≈ 273 years — older than any plausible cache entry.
    let (stdout, stderr, success) =
        run(hf_fm().args(["cache", "gc", "--older-than", "99999", "--dry-run"]));
    assert!(success, "cache gc --dry-run should succeed: {stderr}");
    assert!(
        stdout.contains("No repos matched eviction criteria")
            || stdout.contains("No models in cache")
            || stdout.contains("No HuggingFace cache found"),
        "cache gc --older-than 99999 should report no matches, got:\n{stdout}"
    );
}

#[test]
fn cache_verify_help_shows_flags() {
    let (stdout, stderr, success) = run(hf_fm().args(["cache", "verify", "--help"]));
    assert!(success, "cache verify --help failed: {stderr}");
    for flag in ["--revision", "--token"] {
        assert!(
            stdout.contains(flag),
            "cache verify help should contain {flag}, got:\n{stdout}"
        );
    }
}

#[test]
fn help_shows_cache_verify() {
    let (stdout, stderr, success) = run(hf_fm().args(["cache", "--help"]));
    assert!(success, "cache --help failed: {stderr}");
    assert!(
        stdout.contains("verify"),
        "cache help should mention verify subcommand, got:\n{stdout}"
    );
}

#[test]
fn cache_verify_nonexistent_repo() {
    let (_, stderr, success) =
        run(hf_fm().args(["cache", "verify", "nonexistent-org/nonexistent-model-xyz"]));
    assert!(!success, "cache verify of missing repo should fail");
    assert!(
        stderr.contains("not cached"),
        "cache verify should report not cached, got:\n{stderr}"
    );
}

#[test]
fn cache_verify_against_dummy_unknown() {
    // Pre-cache the test repo so verify has files to look at.
    let (_, _, dl_success) = run(hf_fm().args(["julien-c/dummy-unknown"]));
    assert!(dl_success, "download should succeed to populate cache");

    let (stdout, stderr, success) =
        run(hf_fm().args(["cache", "verify", "julien-c/dummy-unknown"]));
    assert!(success, "cache verify should succeed: {stderr}\n{stdout}");
    // The dummy repo's small JSON files have no LFS metadata, so every file
    // is reported as `no LFS hash`. Either marker satisfies the test.
    assert!(
        stdout.contains("SHA256 OK") || stdout.contains("no LFS hash"),
        "cache verify should show OK or no-LFS-hash markers, got:\n{stdout}"
    );
    // Footer must always be present.
    assert!(
        stdout.contains("SHA256 OK") && stdout.contains("skipped"),
        "cache verify footer should mention OK and skipped counts, got:\n{stdout}"
    );
}

#[test]
fn help_shows_du_subcommand() {
    let (stdout, _stderr, success) = run(hf_fm().arg("--help"));
    assert!(success, "help should succeed");
    assert!(
        stdout.contains("du"),
        "help should mention du subcommand, got:\n{stdout}"
    );
}

#[test]
fn du_summary_lists_cached_repos() {
    // Pre-cache the test repo (list-files triggers a cache entry via API, but
    // a download ensures files exist in snapshots). Use dry-run to avoid
    // large downloads — the cache_summary() function only reads snapshot dirs,
    // so the repo must have been downloaded at least once.
    // Rely on previous test runs having cached julien-c/dummy-unknown.
    let (stdout, stderr, success) = run(hf_fm().args(["du"]));
    // du should succeed even if no models are cached (prints "No models found").
    assert!(success, "du should succeed: {stderr}");
    // If models are cached, output contains numbered header and total.
    assert!(
        stdout.contains("total") || stdout.contains("No models found"),
        "du should show total or empty message, got:\n{stdout}"
    );
    // Numbered output: header row should contain column labels.
    if stdout.contains("total") {
        assert!(
            stdout.contains('#') && stdout.contains("SIZE") && stdout.contains("REPO"),
            "du should show numbered column headers, got:\n{stdout}"
        );
    }
}

#[test]
fn du_repo_shows_files() {
    // First ensure the test repo is cached by downloading it.
    let (_, _, dl_success) = run(hf_fm().args(["julien-c/dummy-unknown"]));
    assert!(dl_success, "download should succeed to populate cache");

    let (stdout, stderr, success) = run(hf_fm().args(["du", "julien-c/dummy-unknown"]));
    assert!(success, "du repo should succeed: {stderr}");
    assert!(
        stdout.contains("julien-c/dummy-unknown:"),
        "du repo should show repo name header, got:\n{stdout}"
    );
    assert!(
        stdout.contains('#') && stdout.contains("SIZE") && stdout.contains("FILE"),
        "du repo should show numbered column headers, got:\n{stdout}"
    );
    assert!(
        stdout.contains("config.json"),
        "du repo should list config.json, got:\n{stdout}"
    );
    assert!(
        stdout.contains("total"),
        "du repo should show total line, got:\n{stdout}"
    );
}

#[test]
fn du_numeric_index_drills_down() {
    // Ensure the test repo is cached.
    let (_, _, dl_success) = run(hf_fm().args(["julien-c/dummy-unknown"]));
    assert!(dl_success, "download should succeed to populate cache");

    // du 1 should drill into the first (largest) repo — same as du <repo_id>.
    let (stdout, stderr, success) = run(hf_fm().args(["du", "1"]));
    assert!(success, "du 1 should succeed: {stderr}");
    assert!(
        stdout.contains("total"),
        "du 1 should show per-file total, got:\n{stdout}"
    );
    assert!(
        stdout.contains("FILE"),
        "du 1 should show file column header, got:\n{stdout}"
    );
}

#[test]
fn du_invalid_index_fails() {
    let (_, stderr, success) = run(hf_fm().args(["du", "99999"]));
    assert!(!success, "du 99999 should fail");
    assert!(
        stderr.contains("out of range"),
        "du 99999 should report out of range, got:\n{stderr}"
    );
}

#[test]
fn du_nonexistent_repo_shows_empty() {
    let (stdout, stderr, success) =
        run(hf_fm().args(["du", "nonexistent-org/nonexistent-model-xyz"]));
    assert!(success, "du for missing repo should succeed: {stderr}");
    assert!(
        stdout.contains("No cached files found"),
        "du for missing repo should say no files found, got:\n{stdout}"
    );
}

#[test]
fn du_tree_succeeds() {
    // Ensure julien-c/dummy-unknown is cached so the tree has at least one branch.
    let (_, _, dl_success) = run(hf_fm().args(["julien-c/dummy-unknown"]));
    assert!(dl_success, "download for du-tree fixture should succeed");

    let (stdout, stderr, success) = run(hf_fm().args(["du", "--tree"]));
    assert!(success, "du --tree should succeed: {stderr}");
    assert!(
        stdout.starts_with("Cache: "),
        "du --tree should announce the cache path, got:\n{stdout}"
    );
    assert!(
        stdout.contains("\u{251c}\u{2500}\u{2500} ")
            || stdout.contains("\u{2514}\u{2500}\u{2500} "),
        "du --tree should render box-drawing connectors, got:\n{stdout}"
    );
    assert!(
        stdout.contains("total ("),
        "du --tree should print a totals line, got:\n{stdout}"
    );
}

#[test]
fn du_tree_with_age_succeeds() {
    let (_, _, dl_success) = run(hf_fm().args(["julien-c/dummy-unknown"]));
    assert!(dl_success, "download for du-tree fixture should succeed");

    let (stdout, stderr, success) = run(hf_fm().args(["du", "--tree", "--age"]));
    assert!(success, "du --tree --age should succeed: {stderr}");
    // `julien-c/dummy-unknown` was just downloaded, so an age string must
    // appear somewhere on the repo branch line.
    assert!(
        stdout.contains("hour") || stdout.contains("day") || stdout.contains("month"),
        "du --tree --age should include a relative age, got:\n{stdout}"
    );
}

#[test]
fn du_tree_conflicts_with_repo_arg() {
    let (_, stderr, success) = run(hf_fm().args(["du", "--tree", "julien-c/dummy-unknown"]));
    assert!(!success, "du --tree <REPO> should be rejected by clap");
    assert!(
        stderr.contains("cannot be used with") || stderr.contains("conflict"),
        "du --tree <REPO> should report a clap conflict, got:\n{stderr}"
    );
}

// -----------------------------------------------------------------------
// inspect subcommand (cache-only tests — no network)
// -----------------------------------------------------------------------

#[test]
fn help_shows_inspect_subcommand() {
    let (stdout, _stderr, success) = run(hf_fm().arg("--help"));
    assert!(success, "help should succeed");
    assert!(
        stdout.contains("inspect"),
        "help should mention inspect subcommand, got:\n{stdout}"
    );
}

/// Finds a cached repo with `.safetensors` files by scanning the HF cache.
///
/// Returns `(repo_id, safetensors_filename)` or `None` if no suitable repo
/// is cached.
fn find_cached_safetensors_repo() -> Option<(String, String)> {
    let cache_dir = dirs::home_dir()?.join(".cache/huggingface/hub");
    if !cache_dir.exists() {
        return None;
    }
    for entry in std::fs::read_dir(&cache_dir).ok()? {
        let entry = entry.ok()?;
        let dir_name = entry.file_name().to_string_lossy().to_string();
        let Some(repo_part) = dir_name.strip_prefix("models--") else {
            continue;
        };
        let repo_id = match repo_part.find("--") {
            Some(pos) => {
                let (org, name_with_sep) = repo_part.split_at(pos);
                let name = name_with_sep.get(2..).unwrap_or_default();
                format!("{org}/{name}")
            }
            None => continue,
        };
        // Walk snapshots to find a .safetensors file.
        let snapshots_dir = entry.path().join("snapshots");
        let Ok(snapshots) = std::fs::read_dir(&snapshots_dir) else {
            continue;
        };
        for snap in snapshots.flatten() {
            if !snap.path().is_dir() {
                continue;
            }
            let Ok(files) = std::fs::read_dir(snap.path()) else {
                continue;
            };
            for file in files.flatten() {
                let fname = file.file_name().to_string_lossy().to_string();
                if fname.ends_with(".safetensors") {
                    return Some((repo_id, fname));
                }
            }
        }
    }
    None
}

/// Returns `(repo_id, filename)` of the first cached `.gguf` file found
/// across all locally-cached repos, or `None` if the local cache has none.
///
/// Used by GGUF-flavoured `inspect --cached` integration tests; these tests
/// SKIP on CI / fresh machines where no `.gguf` has been downloaded. Mirrors
/// the shape of `find_cached_safetensors_repo` exactly so future GGUF tests
/// can drop in the same skip-pattern.
fn find_cached_gguf_repo() -> Option<(String, String)> {
    let cache_dir = dirs::home_dir()?.join(".cache/huggingface/hub");
    if !cache_dir.exists() {
        return None;
    }
    for entry in std::fs::read_dir(&cache_dir).ok()? {
        let entry = entry.ok()?;
        let dir_name = entry.file_name().to_string_lossy().to_string();
        let Some(repo_part) = dir_name.strip_prefix("models--") else {
            continue;
        };
        let repo_id = match repo_part.find("--") {
            Some(pos) => {
                let (org, name_with_sep) = repo_part.split_at(pos);
                let name = name_with_sep.get(2..).unwrap_or_default();
                format!("{org}/{name}")
            }
            None => continue,
        };
        let snapshots_dir = entry.path().join("snapshots");
        let Ok(snapshots) = std::fs::read_dir(&snapshots_dir) else {
            continue;
        };
        for snap in snapshots.flatten() {
            if !snap.path().is_dir() {
                continue;
            }
            let Ok(files) = std::fs::read_dir(snap.path()) else {
                continue;
            };
            for file in files.flatten() {
                let path = file.path();
                if path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
                {
                    let fname = file.file_name().to_string_lossy().to_string();
                    return Some((repo_id, fname));
                }
            }
        }
    }
    None
}

/// Returns `(repo_id, filename)` of the first cached `.pth` file found
/// across all locally-cached repos, or `None` if the local cache has none.
///
/// Used by PTH-flavoured `inspect --cached` integration tests; these tests
/// SKIP on CI / fresh machines where no `.pth` has been downloaded.
fn find_cached_pth_repo() -> Option<(String, String)> {
    let cache_dir = dirs::home_dir()?.join(".cache/huggingface/hub");
    if !cache_dir.exists() {
        return None;
    }
    for entry in std::fs::read_dir(&cache_dir).ok()? {
        let entry = entry.ok()?;
        let dir_name = entry.file_name().to_string_lossy().to_string();
        let Some(repo_part) = dir_name.strip_prefix("models--") else {
            continue;
        };
        let repo_id = match repo_part.find("--") {
            Some(pos) => {
                let (org, name_with_sep) = repo_part.split_at(pos);
                let name = name_with_sep.get(2..).unwrap_or_default();
                format!("{org}/{name}")
            }
            None => continue,
        };
        let snapshots_dir = entry.path().join("snapshots");
        let Ok(snapshots) = std::fs::read_dir(&snapshots_dir) else {
            continue;
        };
        for snap in snapshots.flatten() {
            if !snap.path().is_dir() {
                continue;
            }
            let Ok(files) = std::fs::read_dir(snap.path()) else {
                continue;
            };
            for file in files.flatten() {
                let path = file.path();
                if path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("pth"))
                {
                    let fname = file.file_name().to_string_lossy().to_string();
                    return Some((repo_id, fname));
                }
            }
        }
    }
    None
}

/// Returns `(repo_id, filename)` of the first cached `.npz` file found
/// across all locally-cached repos, or `None` if the local cache has none.
///
/// Used by NPZ-flavoured `inspect --cached` integration tests; these tests
/// SKIP on CI / fresh machines where no `.npz` has been downloaded. Mirrors
/// the shape of `find_cached_gguf_repo` exactly.
fn find_cached_npz_repo() -> Option<(String, String)> {
    let cache_dir = dirs::home_dir()?.join(".cache/huggingface/hub");
    if !cache_dir.exists() {
        return None;
    }
    for entry in std::fs::read_dir(&cache_dir).ok()? {
        let entry = entry.ok()?;
        let dir_name = entry.file_name().to_string_lossy().to_string();
        let Some(repo_part) = dir_name.strip_prefix("models--") else {
            continue;
        };
        let repo_id = match repo_part.find("--") {
            Some(pos) => {
                let (org, name_with_sep) = repo_part.split_at(pos);
                let name = name_with_sep.get(2..).unwrap_or_default();
                format!("{org}/{name}")
            }
            None => continue,
        };
        let snapshots_dir = entry.path().join("snapshots");
        let Ok(snapshots) = std::fs::read_dir(&snapshots_dir) else {
            continue;
        };
        for snap in snapshots.flatten() {
            if !snap.path().is_dir() {
                continue;
            }
            let Ok(files) = std::fs::read_dir(snap.path()) else {
                continue;
            };
            for file in files.flatten() {
                let path = file.path();
                if path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("npz"))
                {
                    let fname = file.file_name().to_string_lossy().to_string();
                    return Some((repo_id, fname));
                }
            }
        }
    }
    None
}

/// Finds all cached repos that have at least one `.safetensors` file.
fn find_all_cached_safetensors_repos() -> Vec<String> {
    let mut repos = Vec::new();
    let Some(cache_dir) = dirs::home_dir().map(|h| h.join(".cache/huggingface/hub")) else {
        return repos;
    };
    if !cache_dir.exists() {
        return repos;
    }
    let Ok(entries) = std::fs::read_dir(&cache_dir) else {
        return repos;
    };
    for entry in entries.flatten() {
        let dir_name = entry.file_name().to_string_lossy().to_string();
        let Some(repo_part) = dir_name.strip_prefix("models--") else {
            continue;
        };
        let repo_id = match repo_part.find("--") {
            Some(pos) => {
                let (org, name_with_sep) = repo_part.split_at(pos);
                let name = name_with_sep.get(2..).unwrap_or_default();
                format!("{org}/{name}")
            }
            None => continue,
        };
        let snapshots_dir = entry.path().join("snapshots");
        let Ok(snapshots) = std::fs::read_dir(&snapshots_dir) else {
            continue;
        };
        'snap: for snap in snapshots.flatten() {
            if !snap.path().is_dir() {
                continue;
            }
            let Ok(files) = std::fs::read_dir(snap.path()) else {
                continue;
            };
            for file in files.flatten() {
                let fname = file.file_name().to_string_lossy().to_string();
                if fname.ends_with(".safetensors") {
                    repos.push(repo_id.clone());
                    break 'snap;
                }
            }
        }
    }
    repos
}

/// Finds a cached .safetensors file that has `__metadata__` in its header.
///
/// Reads the raw header JSON and checks for the `__metadata__` key.
fn find_cached_safetensors_with_metadata() -> Option<(String, String)> {
    use std::io::Read;

    let cache_dir = dirs::home_dir()?.join(".cache/huggingface/hub");
    if !cache_dir.exists() {
        return None;
    }
    for entry in std::fs::read_dir(&cache_dir).ok()? {
        let entry = entry.ok()?;
        let dir_name = entry.file_name().to_string_lossy().to_string();
        let Some(repo_part) = dir_name.strip_prefix("models--") else {
            continue;
        };
        let repo_id = match repo_part.find("--") {
            Some(pos) => {
                let (org, name_with_sep) = repo_part.split_at(pos);
                let name = name_with_sep.get(2..).unwrap_or_default();
                format!("{org}/{name}")
            }
            None => continue,
        };
        let snapshots_dir = entry.path().join("snapshots");
        let Ok(snapshots) = std::fs::read_dir(&snapshots_dir) else {
            continue;
        };
        for snap in snapshots.flatten() {
            if !snap.path().is_dir() {
                continue;
            }
            let Ok(files) = std::fs::read_dir(snap.path()) else {
                continue;
            };
            for file in files.flatten() {
                let fname = file.file_name().to_string_lossy().to_string();
                if !fname.ends_with(".safetensors") {
                    continue;
                }
                // Read header and check for __metadata__.
                let Ok(mut f) = std::fs::File::open(file.path()) else {
                    continue;
                };
                let mut len_buf = [0u8; 8];
                if f.read_exact(&mut len_buf).is_err() {
                    continue;
                }
                let Ok(header_size) = usize::try_from(u64::from_le_bytes(len_buf)) else {
                    continue;
                };
                if header_size > 10_000_000 {
                    continue; // Skip unreasonably large headers
                }
                let mut json_buf = vec![0u8; header_size];
                if f.read_exact(&mut json_buf).is_err() {
                    continue;
                }
                if let Ok(text) = std::str::from_utf8(&json_buf) {
                    if text.contains("__metadata__") {
                        return Some((repo_id, fname));
                    }
                }
            }
        }
    }
    None
}

/// Finds a cached `.safetensors` file whose header contains BnB-quantized
/// tensor names — the positive signals anamnesis keys on for
/// `QuantScheme::Bnb4` / `BnbInt8` detection (`.weight.quant_map`,
/// `.weight.absmax`, `.SCB`).
///
/// Stronger than `find_cached_safetensors_with_metadata`: that helper only
/// requires the universal `__metadata__` block, which is present on
/// every transformers safetensors export regardless of quantization. The
/// `BnB` markers below only appear on actually-bitsandbytes-quantized files,
/// so a test built on this helper can ASSERT the Format/Size lines
/// instead of conditionally checking them — closing the regression gap
/// where the v0.10.3 Phase C feature shipped silently no-op for `BnB`
/// because hf-fm's `anamnesis` dep was missing `features = ["bnb"]`.
fn find_cached_bnb_safetensors_repo() -> Option<(String, String)> {
    use std::io::Read;

    let cache_dir = dirs::home_dir()?.join(".cache/huggingface/hub");
    if !cache_dir.exists() {
        return None;
    }
    for entry in std::fs::read_dir(&cache_dir).ok()? {
        let entry = entry.ok()?;
        let dir_name = entry.file_name().to_string_lossy().to_string();
        let Some(repo_part) = dir_name.strip_prefix("models--") else {
            continue;
        };
        let repo_id = match repo_part.find("--") {
            Some(pos) => {
                let (org, name_with_sep) = repo_part.split_at(pos);
                let name = name_with_sep.get(2..).unwrap_or_default();
                format!("{org}/{name}")
            }
            None => continue,
        };
        let snapshots_dir = entry.path().join("snapshots");
        let Ok(snapshots) = std::fs::read_dir(&snapshots_dir) else {
            continue;
        };
        for snap in snapshots.flatten() {
            if !snap.path().is_dir() {
                continue;
            }
            let Ok(files) = std::fs::read_dir(snap.path()) else {
                continue;
            };
            for file in files.flatten() {
                let fname = file.file_name().to_string_lossy().to_string();
                if !fname.ends_with(".safetensors") {
                    continue;
                }
                let Ok(mut f) = std::fs::File::open(file.path()) else {
                    continue;
                };
                let mut len_buf = [0u8; 8];
                if f.read_exact(&mut len_buf).is_err() {
                    continue;
                }
                let Ok(header_size) = usize::try_from(u64::from_le_bytes(len_buf)) else {
                    continue;
                };
                if header_size > 10_000_000 {
                    continue;
                }
                let mut json_buf = vec![0u8; header_size];
                if f.read_exact(&mut json_buf).is_err() {
                    continue;
                }
                if let Ok(text) = std::str::from_utf8(&json_buf) {
                    // Match against tensor-name markers anamnesis classifies
                    // as BnB (NF4/FP4: `.weight.quant_map`; INT8: `.SCB`).
                    // `.weight.absmax` would also work but `.weight.quant_map`
                    // is the definitive NF4/FP4 marker per
                    // anamnesis::parse::safetensors::detect_scheme.
                    if text.contains(".weight.quant_map") || text.contains(".SCB\"") {
                        return Some((repo_id, fname));
                    }
                }
            }
        }
    }
    None
}

/// Finds a cached repo that has a `model.safetensors.index.json` (sharded model).
fn find_cached_sharded_repo() -> Option<String> {
    let cache_dir = dirs::home_dir()?.join(".cache/huggingface/hub");
    if !cache_dir.exists() {
        return None;
    }
    for entry in std::fs::read_dir(&cache_dir).ok()? {
        let entry = entry.ok()?;
        let dir_name = entry.file_name().to_string_lossy().to_string();
        let Some(repo_part) = dir_name.strip_prefix("models--") else {
            continue;
        };
        let repo_id = match repo_part.find("--") {
            Some(pos) => {
                let (org, name_with_sep) = repo_part.split_at(pos);
                let name = name_with_sep.get(2..).unwrap_or_default();
                format!("{org}/{name}")
            }
            None => continue,
        };
        let snapshots_dir = entry.path().join("snapshots");
        let Ok(snapshots) = std::fs::read_dir(&snapshots_dir) else {
            continue;
        };
        for snap in snapshots.flatten() {
            if !snap.path().is_dir() {
                continue;
            }
            if snap.path().join("model.safetensors.index.json").exists() {
                return Some(repo_id);
            }
        }
    }
    None
}

#[test]
fn inspect_cached_single_file() {
    let Some((repo_id, filename)) = find_cached_safetensors_repo() else {
        eprintln!("SKIP: no cached safetensors repo found");
        return;
    };
    let (stdout, stderr, success) = run(hf_fm().args(["inspect", &repo_id, &filename, "--cached"]));
    assert!(
        success,
        "inspect --cached should succeed for {repo_id}/{filename}: {stderr}"
    );
    assert!(
        stdout.contains("Source:   cached"),
        "should report cached source, got:\n{stdout}"
    );
    assert!(
        stdout.contains("Tensor") && stdout.contains("Dtype") && stdout.contains("Shape"),
        "should show tensor table headers, got:\n{stdout}"
    );
    assert!(
        stdout.contains("tensors"),
        "should show tensor count summary, got:\n{stdout}"
    );
}

#[test]
fn inspect_cached_json_output() {
    let Some((repo_id, filename)) = find_cached_safetensors_repo() else {
        eprintln!("SKIP: no cached safetensors repo found");
        return;
    };
    let (stdout, stderr, success) =
        run(hf_fm().args(["inspect", &repo_id, &filename, "--cached", "--json"]));
    assert!(success, "inspect --cached --json should succeed: {stderr}");
    // Verify it's valid JSON with expected fields.
    assert!(
        stdout.contains("\"tensors\""),
        "JSON should contain tensors field, got:\n{stdout}"
    );
    assert!(
        stdout.contains("\"header_size\""),
        "JSON should contain header_size field, got:\n{stdout}"
    );
}

#[test]
fn inspect_check_gpu_context_renders_kv_line() {
    // The KV-cache line prints before the GPU probe, so this assertion is
    // GPU-agnostic: it holds whether the device is present, absent, or the
    // config is unavailable (the line then reads "unavailable"/"skipped").
    let Some((repo_id, filename)) = find_cached_safetensors_repo() else {
        eprintln!("SKIP: no cached safetensors repo found");
        return;
    };
    let (stdout, stderr, success) = run(hf_fm().args([
        "inspect",
        &repo_id,
        &filename,
        "--cached",
        "--check-gpu",
        "--context",
        "8192",
    ]));
    // `--check-gpu` is informational and never gates — always exit 0.
    assert!(
        success,
        "inspect --check-gpu --context should succeed: {stderr}"
    );
    assert!(
        stdout.contains("KV cache"),
        "should render a KV cache line, got:\n{stdout}"
    );
}

#[test]
fn inspect_context_requires_check_gpu() {
    // `--context` without `--check-gpu` is rejected at parse time (clap
    // `requires`), before any network/cache access — so a placeholder repo is
    // fine and no cache is needed.
    let (_stdout, stderr, success) = run(hf_fm().args([
        "inspect",
        "kvtest/placeholder",
        "--cached",
        "--context",
        "4096",
    ]));
    assert!(!success, "missing --check-gpu should fail at parse time");
    assert!(
        stderr.contains("--check-gpu"),
        "error should name the required --check-gpu flag, got:\n{stderr}"
    );
}

#[test]
fn inspect_check_gpu_context_json_has_kv_cache() {
    // With `--context`, the JSON verdict always carries a `kv_cache` object
    // (computed, skipped, or unavailable) — independent of GPU presence.
    let Some((repo_id, filename)) = find_cached_safetensors_repo() else {
        eprintln!("SKIP: no cached safetensors repo found");
        return;
    };
    let (stdout, stderr, success) = run(hf_fm().args([
        "inspect",
        &repo_id,
        &filename,
        "--cached",
        "--check-gpu",
        "--context",
        "4096",
        "--json",
    ]));
    assert!(
        success,
        "inspect --check-gpu --context --json should succeed: {stderr}"
    );
    assert!(
        stdout.contains("\"kv_cache\""),
        "JSON should contain a kv_cache object, got:\n{stdout}"
    );
}

#[test]
fn inspect_check_gpu_without_context_omits_kv() {
    // Regression: without `--context`, no KV/Total lines appear (GPU-agnostic —
    // both print before the GPU probe). Guards the byte-identical legacy path.
    let Some((repo_id, filename)) = find_cached_safetensors_repo() else {
        eprintln!("SKIP: no cached safetensors repo found");
        return;
    };
    let (stdout, stderr, success) =
        run(hf_fm().args(["inspect", &repo_id, &filename, "--cached", "--check-gpu"]));
    assert!(success, "inspect --check-gpu should succeed: {stderr}");
    // The `Total:` line and the `KV cache @ ctx=` line are unique to the
    // `--context` path. (The plain weights-only note legitimately mentions
    // "KV cache", so we discriminate on these markers, not the bare phrase.)
    assert!(
        !stdout.contains("Total:") && !stdout.contains("KV cache @"),
        "no-context output must omit the KV/Total lines, got:\n{stdout}"
    );
}

#[test]
fn inspect_cached_no_metadata() {
    let Some((repo_id, filename)) = find_cached_safetensors_repo() else {
        eprintln!("SKIP: no cached safetensors repo found");
        return;
    };
    let (stdout, stderr, success) =
        run(hf_fm().args(["inspect", &repo_id, &filename, "--cached", "--no-metadata"]));
    assert!(
        success,
        "inspect --cached --no-metadata should succeed: {stderr}"
    );
    assert!(
        !stdout.contains("Metadata:"),
        "--no-metadata should suppress Metadata line, got:\n{stdout}"
    );
}

#[test]
fn inspect_cached_repo_summary() {
    let Some((repo_id, _filename)) = find_cached_safetensors_repo() else {
        eprintln!("SKIP: no cached safetensors repo found");
        return;
    };
    let (stdout, stderr, success) = run(hf_fm().args(["inspect", &repo_id, "--cached"]));
    assert!(
        success,
        "inspect --cached repo summary should succeed: {stderr}"
    );
    // Should show either shard index or multi-file summary.
    assert!(
        stdout.contains("tensors") || stdout.contains("Tensors"),
        "should mention tensors in output, got:\n{stdout}"
    );
}

#[test]
fn inspect_cached_metadata_present() {
    // Find a cached safetensors file that has __metadata__ in its header.
    let Some((repo_id, filename)) = find_cached_safetensors_with_metadata() else {
        eprintln!("SKIP: no cached safetensors file with __metadata__ found");
        return;
    };
    let (stdout, stderr, success) = run(hf_fm().args(["inspect", &repo_id, &filename, "--cached"]));
    assert!(
        success,
        "inspect --cached should succeed for {repo_id}/{filename}: {stderr}"
    );
    assert!(
        stdout.contains("Metadata:"),
        "output should contain Metadata: line by default, got:\n{stdout}"
    );
    // Output should contain at least one `key=value` pair. v0.10.3's render
    // polish switched metadata blocks with >6 keys to a tabular form where
    // the `Metadata:` line is just the header and `key=value` pairs live on
    // subsequent indented lines — so we check across all lines, not just the
    // header line itself.
    let has_kv = stdout.lines().any(|l| l.contains('='));
    assert!(
        has_kv,
        "should contain at least one key=value pair, got:\n{stdout}"
    );
}

#[test]
fn inspect_cached_sharded_model() {
    let Some(repo_id) = find_cached_sharded_repo() else {
        eprintln!("SKIP: no cached sharded safetensors model found");
        return;
    };
    let (stdout, stderr, success) = run(hf_fm().args(["inspect", &repo_id, "--cached"]));
    assert!(
        success,
        "inspect --cached sharded model should succeed: {stderr}"
    );
    assert!(
        stdout.contains("shard index"),
        "sharded model should show shard index source, got:\n{stdout}"
    );
    assert!(
        stdout.contains("shards") || stdout.contains("shard,"),
        "should show shard count, got:\n{stdout}"
    );
    assert!(
        stdout.contains("Hint:"),
        "should show per-tensor detail hint, got:\n{stdout}"
    );
}

#[test]
fn inspect_cached_sharded_dtypes_aggregates() {
    let Some(repo_id) = find_cached_sharded_repo() else {
        eprintln!("SKIP: no cached sharded safetensors model found");
        return;
    };
    let (stdout, stderr, success) =
        run(hf_fm().args(["inspect", &repo_id, "--cached", "--dtypes"]));
    assert!(
        success,
        "inspect --cached --dtypes sharded model should succeed: {stderr}"
    );
    // Bypassing the shard-index fast path replaces "Source: shard index" with
    // the aggregator's "Source: aggregated across N shards".
    assert!(
        stdout.contains("aggregated across"),
        "sharded --dtypes should report aggregated source, got:\n{stdout}"
    );
    // The dtype histogram header is the marker that the renderer ran.
    assert!(
        stdout.contains("Dtype") && stdout.contains("Tensors") && stdout.contains("Params"),
        "sharded --dtypes should show histogram columns, got:\n{stdout}"
    );
    assert!(
        !stdout.contains("Hint: use `hf-fm inspect"),
        "aggregated --dtypes should NOT print the per-file hint, got:\n{stdout}"
    );
}

#[test]
fn inspect_cached_sharded_tree_aggregates() {
    let Some(repo_id) = find_cached_sharded_repo() else {
        eprintln!("SKIP: no cached sharded safetensors model found");
        return;
    };
    let (stdout, stderr, success) = run(hf_fm().args(["inspect", &repo_id, "--cached", "--tree"]));
    assert!(
        success,
        "inspect --cached --tree sharded model should succeed: {stderr}"
    );
    assert!(
        stdout.contains("aggregated across"),
        "sharded --tree should report aggregated source, got:\n{stdout}"
    );
    // Tree connectors are the unambiguous marker that the renderer ran.
    assert!(
        stdout.contains("├──") || stdout.contains("└──"),
        "sharded --tree should render box-drawing connectors, got:\n{stdout}"
    );
}

#[test]
fn inspect_cached_gguf_tree_renders() {
    // v0.10.3 Phase A commit 2: confirm `--tree` works for cached GGUF.
    // The tree pipeline (`build_tree` → `collapse_ranges`) is format-agnostic;
    // GGUF's `blk.<N>.<part>` naming should collapse the same way safetensors'
    // `model.layers.<N>.<part>` does. We do not assert on specific tensor names
    // because different cached GGUFs have different structures; the
    // box-drawing-and-footer check confirms the renderer ran end-to-end.
    let Some((repo_id, filename)) = find_cached_gguf_repo() else {
        eprintln!("SKIP: no cached .gguf file found");
        return;
    };
    let (stdout, stderr, success) =
        run(hf_fm().args(["inspect", &repo_id, &filename, "--cached", "--tree"]));
    assert!(
        success,
        "inspect --cached --tree on GGUF should succeed: {stderr}"
    );
    assert!(
        stdout.contains('\u{2500}') || stdout.contains('\u{2514}') || stdout.contains('\u{251c}'),
        "GGUF --tree output should contain box-drawing connectors, got:\n{stdout}"
    );
    assert!(
        stdout.contains("tensor"),
        "GGUF --tree should print a tensor-count footer, got:\n{stdout}"
    );
}

#[test]
fn inspect_cached_gguf_dtypes_renders() {
    // v0.10.3 Phase A commit 3: confirm `--dtypes` works for cached GGUF.
    // `compute_dtype_groups` already buckets by `t.dtype` string and sums
    // `t.byte_len()` directly (no `dtype_bytes()` lookup), so GGUF
    // quantization dtypes (`Q4_K_M`, `Q2_K`, `IQ4_NL`, `F32`, …) bucket
    // transparently with the byte counts anamnesis populates at parse time.
    let Some((repo_id, filename)) = find_cached_gguf_repo() else {
        eprintln!("SKIP: no cached .gguf file found");
        return;
    };
    let (stdout, stderr, success) =
        run(hf_fm().args(["inspect", &repo_id, &filename, "--cached", "--dtypes"]));
    assert!(
        success,
        "inspect --cached --dtypes on GGUF should succeed: {stderr}"
    );
    assert!(
        stdout.contains("Dtype") && stdout.contains("Tensors") && stdout.contains("Params"),
        "GGUF --dtypes should show histogram columns, got:\n{stdout}"
    );
    // Any cached GGUF will have at least one row matching one of the
    // common dtype prefixes — the OR chain stays robust across quantizations
    // (Q4_K_M, Q5_0, Q8_0, IQ4_NL, …) and the F32 / F16 passthrough types.
    let has_gguf_dtype = stdout.lines().any(|l| {
        l.contains("Q4_")
            || l.contains("Q5_")
            || l.contains("Q6_")
            || l.contains("Q8_")
            || l.contains("Q2_K")
            || l.contains("Q3_K")
            || l.contains("IQ")
            || l.contains("F32")
            || l.contains("F16")
    });
    assert!(
        has_gguf_dtype,
        "GGUF --dtypes should bucket GGUF dtype names, got:\n{stdout}"
    );
}

#[test]
fn inspect_cached_npz_renders() {
    // v0.10.3 Phase B commit 5: confirm `--cached` inspect works for `.npz`.
    // Delegates to `anamnesis::inspect_npz` which reads only the ZIP central
    // directory + per-entry NPY headers (no tensor data). hf-fm synthesises
    // cumulative `(start, end)` offsets from per-tensor `byte_len`.
    let Some((repo_id, filename)) = find_cached_npz_repo() else {
        eprintln!("SKIP: no cached .npz file found");
        return;
    };
    let (stdout, stderr, success) = run(hf_fm().args(["inspect", &repo_id, &filename, "--cached"]));
    assert!(success, "inspect --cached on NPZ should succeed: {stderr}");
    assert!(
        stdout.contains("Tensor") && stdout.contains("Dtype"),
        "NPZ inspect should print the standard tensor table, got:\n{stdout}"
    );
}

#[test]
fn inspect_cached_pth_renders() {
    // v0.10.3 Phase B commit 6: confirm `--cached` inspect works for `.pth`.
    // Delegates to `anamnesis::parse_pth` + `ParsedPth::tensor_info()` (new
    // in anamnesis 0.5.0 — metadata-only, no data materialisation). hf-fm
    // synthesises cumulative `(start, end)` offsets from per-tensor
    // `byte_len`, same pattern as NPZ.
    let Some((repo_id, filename)) = find_cached_pth_repo() else {
        eprintln!("SKIP: no cached .pth file found");
        return;
    };
    let (stdout, stderr, success) = run(hf_fm().args(["inspect", &repo_id, &filename, "--cached"]));
    assert!(success, "inspect --cached on PTH should succeed: {stderr}");
    assert!(
        stdout.contains("Tensor") && stdout.contains("Dtype"),
        "PTH inspect should print the standard tensor table, got:\n{stdout}"
    );
}

#[test]
fn inspect_cached_quantized_renders_format_line() {
    // v0.10.3 Phase C: confirm the `Format:` + `Size: <stored> stored -> <deq> (BF16)`
    // lines appear on quantized safetensors. `find_cached_safetensors_with_metadata`
    // locates a `__metadata__`-bearing file (typical for GPTQ/AWQ/BnB-quantized
    // models); any returned file may or may not be detected as quantized by
    // anamnesis (unquantized files with __metadata__ exist), so we only assert
    // that IF the Format: line appears, it's paired with a Size: line carrying
    // the `stored -> … (BF16)` shape.
    let Some((repo_id, filename)) = find_cached_safetensors_with_metadata() else {
        eprintln!("SKIP: no cached safetensors file with __metadata__ found");
        return;
    };
    let (stdout, stderr, success) = run(hf_fm().args(["inspect", &repo_id, &filename, "--cached"]));
    assert!(
        success,
        "inspect --cached on metadata-bearing safetensors should succeed: {stderr}"
    );
    if stdout
        .lines()
        .any(|l| l.trim_start().starts_with("Format:"))
    {
        // Quantization was detected — the paired Size: line MUST follow the
        // anamnesis-flavoured `stored -> ... (BF16)` shape.
        let has_size_line = stdout
            .lines()
            .any(|l| l.contains(" stored -> ") && l.contains("(BF16)"));
        assert!(
            has_size_line,
            "Format: line present but no `Size: <stored> stored -> <deq> (BF16)` companion, got:\n{stdout}"
        );
    } else {
        eprintln!(
            "SKIP-DETAIL: quantization not detected on {repo_id}/{filename}; \
             cannot exercise the Format/Size pair (file may be unquantized \
             despite carrying __metadata__)."
        );
    }
}

#[test]
fn inspect_cached_bnb_renders_format_and_size_lines() {
    // Regression guard for the v0.10.3 → unreleased fix: prior to enabling
    // the `bnb` feature on the `anamnesis` dep, this exact invocation
    // returned no Format:/Size: lines and `quant_info: None` in JSON for
    // every BnB-quantized cached file — silently. Unlike
    // `inspect_cached_quantized_renders_format_line` (which conditionally
    // checks the lines IF present), this test asserts they MUST be present
    // when the source file's header contains BnB tensor-name markers.
    let Some((repo_id, filename)) = find_cached_bnb_safetensors_repo() else {
        eprintln!("SKIP: no cached BnB-quantized safetensors file found");
        return;
    };
    let (stdout, stderr, success) =
        run(hf_fm().args(["inspect", &repo_id, &filename, "--cached", "--limit", "1"]));
    assert!(
        success,
        "inspect --cached on BnB safetensors should succeed: {stderr}"
    );
    let has_format_line = stdout
        .lines()
        .any(|l| l.trim_start().starts_with("Format:") && l.contains("BitsAndBytes"));
    assert!(
        has_format_line,
        "BnB-quantized {repo_id}/{filename} must render a `Format: BitsAndBytes …` line \
         (regression guard: missing `bnb` feature on the anamnesis dep silently \
         disables this), got:\n{stdout}"
    );
    let has_size_line = stdout
        .lines()
        .any(|l| l.contains(" stored -> ") && l.contains("(BF16)"));
    assert!(
        has_size_line,
        "BnB-quantized {repo_id}/{filename} must render a `Size: <stored> stored -> <deq> (BF16)` \
         line companion to Format:, got:\n{stdout}"
    );
}

#[test]
fn inspect_cached_bnb_json_carries_quant_info() {
    // JSON-shape sibling of the above. The `quant_info` field is gated
    // `#[serde(skip_serializing_if = "Option::is_none")]` so its absence
    // on a known-BnB file is the exact JSON-side signature of the
    // missing-feature-flag regression this guards against.
    let Some((repo_id, filename)) = find_cached_bnb_safetensors_repo() else {
        eprintln!("SKIP: no cached BnB-quantized safetensors file found");
        return;
    };
    let (stdout, stderr, success) = run(hf_fm().args([
        "inspect", &repo_id, &filename, "--cached", "--json", "--limit", "1",
    ]));
    assert!(
        success,
        "inspect --cached --json on BnB safetensors should succeed: {stderr}"
    );
    assert!(
        stdout.contains("\"quant_info\""),
        "BnB-quantized {repo_id}/{filename} JSON output must carry the `quant_info` \
         field (regression guard: silent `None` on BnB indicated missing `bnb` feature \
         on the anamnesis dep), got:\n{stdout}"
    );
}

#[test]
fn inspect_cached_sharded_dtypes_json_aggregates() {
    let Some(repo_id) = find_cached_sharded_repo() else {
        eprintln!("SKIP: no cached sharded safetensors model found");
        return;
    };
    let (stdout, stderr, success) =
        run(hf_fm().args(["inspect", &repo_id, "--cached", "--dtypes", "--json"]));
    assert!(
        success,
        "inspect --cached --dtypes --json sharded model should succeed: {stderr}"
    );
    assert!(
        stdout.contains("\"dtypes\"") && stdout.contains("\"total_tensors\""),
        "JSON should contain dtypes + total_tensors fields, got:\n{stdout}"
    );
}

#[test]
fn inspect_cached_sharded_limit_shows_shard_column() {
    let Some(repo_id) = find_cached_sharded_repo() else {
        eprintln!("SKIP: no cached sharded safetensors model found");
        return;
    };
    let (stdout, stderr, success) =
        run(hf_fm().args(["inspect", &repo_id, "--cached", "--limit", "3"]));
    assert!(
        success,
        "inspect --cached --limit sharded model should succeed: {stderr}"
    );
    assert!(
        stdout.contains("aggregated across"),
        "sharded --limit should report aggregated source, got:\n{stdout}"
    );
    // The Shard column is the marker that distinguishes the multi-shard
    // table from the single-file inspect table.
    assert!(
        stdout.contains("Shard"),
        "sharded --limit table should include a Shard column, got:\n{stdout}"
    );
    assert!(
        stdout.contains("limit: 3"),
        "footer should mention the active limit, got:\n{stdout}"
    );
}

#[test]
fn inspect_cached_filter() {
    let Some((repo_id, filename)) = find_cached_safetensors_repo() else {
        eprintln!("SKIP: no cached safetensors repo found");
        return;
    };
    // Run unfiltered to get total tensor count.
    let (stdout_all, _stderr, success) =
        run(hf_fm().args(["inspect", &repo_id, &filename, "--cached"]));
    assert!(success, "inspect --cached should succeed");
    // Extract total from summary line (e.g., "364 tensors").
    let has_tensors = stdout_all.contains("tensor");
    assert!(has_tensors, "unfiltered output should show tensors");

    // Run with --filter "embed" (most models have an embedding tensor).
    let (stdout, stderr, success) = run(hf_fm().args([
        "inspect", &repo_id, &filename, "--cached", "--filter", "embed",
    ]));
    assert!(
        success,
        "inspect --cached --filter should succeed: {stderr}"
    );
    // Every displayed tensor name should contain "embed".
    for line in stdout.lines() {
        // Skip header, separator, summary, and metadata lines.
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with("Repo:")
            || trimmed.starts_with("File:")
            || trimmed.starts_with("Source:")
            || trimmed.starts_with("Header:")
            || trimmed.starts_with("Metadata:")
            || trimmed.starts_with("Tensor")
            || trimmed.starts_with('\u{2500}')
            || trimmed.starts_with("Showing ")
            || trimmed.starts_with("Param counts:")
        {
            continue;
        }
        assert!(
            trimmed.contains("embed"),
            "filtered line should contain 'embed': {trimmed}"
        );
    }
    // Summary line should use the labelled "Showing X of Y tensors matching
    // filter ..." form introduced in v0.10.3.
    assert!(
        stdout.contains("Showing ") && stdout.contains("matching filter"),
        "filtered summary should show 'Showing X of Y tensors matching filter ...', got:\n{stdout}"
    );
}

// -----------------------------------------------------------------------
// diff subcommand (cache-only tests — no network)
// -----------------------------------------------------------------------

#[test]
fn help_shows_diff_subcommand() {
    let (stdout, _stderr, success) = run(hf_fm().arg("--help"));
    assert!(success, "help should succeed");
    assert!(
        stdout.contains("diff"),
        "help should mention diff subcommand, got:\n{stdout}"
    );
}

#[test]
fn diff_cached_identical_model() {
    let Some((repo_id, _filename)) = find_cached_safetensors_repo() else {
        eprintln!("SKIP: no cached safetensors repo found");
        return;
    };
    // Diff a model against itself — everything should match.
    let (stdout, stderr, success) = run(hf_fm().args(["diff", &repo_id, &repo_id, "--cached"]));
    assert!(success, "diff --cached self-diff should succeed: {stderr}");
    // Should show zero only-A, only-B, and differ.
    assert!(
        stdout.contains("only-A: 0"),
        "self-diff should have 0 only-A, got:\n{stdout}"
    );
    assert!(
        stdout.contains("only-B: 0"),
        "self-diff should have 0 only-B, got:\n{stdout}"
    );
    assert!(
        stdout.contains("differ: 0"),
        "self-diff should have 0 differ, got:\n{stdout}"
    );
    // Match count should be > 0.
    assert!(
        stdout.contains("Matching:") && !stdout.contains("Matching: 0"),
        "self-diff should have matching tensors, got:\n{stdout}"
    );
}

#[test]
fn diff_cached_different_models() {
    // Find two different cached repos with safetensors.
    let repos = find_all_cached_safetensors_repos();
    // INDEX: length checked before access
    let (Some(repo_a), Some(repo_b)) = (repos.first(), repos.get(1)) else {
        eprintln!("SKIP: need at least 2 cached safetensors repos for diff test");
        return;
    };

    let (stdout, stderr, success) =
        run(hf_fm().args(["diff", repo_a.as_str(), repo_b.as_str(), "--cached"]));
    assert!(
        success,
        "diff --cached different models should succeed: {stderr}"
    );
    // Should have the A: and B: labels.
    assert!(
        stdout.contains(&format!("A: {repo_a}")),
        "should show repo A label, got:\n{stdout}"
    );
    assert!(
        stdout.contains(&format!("B: {repo_b}")),
        "should show repo B label, got:\n{stdout}"
    );
    // Summary line should be present.
    assert!(
        stdout.contains("A:") && stdout.contains("tensors"),
        "should show summary line, got:\n{stdout}"
    );
}
