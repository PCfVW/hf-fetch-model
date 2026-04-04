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
    // Metadata line should contain key=value pairs.
    let meta_line = stdout.lines().find(|l| l.contains("Metadata:")).unwrap();
    assert!(
        meta_line.contains('='),
        "Metadata line should contain key=value pairs, got: {meta_line}"
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
            || trimmed.contains("tensor")
            || trimmed.contains("params")
        {
            continue;
        }
        assert!(
            trimmed.contains("embed"),
            "filtered line should contain 'embed': {trimmed}"
        );
    }
    // Summary should show filtered/total format.
    assert!(
        stdout.contains('/'),
        "filtered summary should show filtered/total format, got:\n{stdout}"
    );
    assert!(
        stdout.contains("filter:"),
        "filtered summary should mention filter, got:\n{stdout}"
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
