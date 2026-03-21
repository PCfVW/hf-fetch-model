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
fn help_shows_dry_run_flag() {
    let (stdout, stderr, success) = run(hf_fm().arg("--help"));
    assert!(success, "--help failed: {stderr}");
    assert!(
        stdout.contains("--dry-run"),
        "help should list the --dry-run flag, got:\n{stdout}"
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
