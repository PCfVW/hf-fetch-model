// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multi-connection HTTP Range-based parallel download for large files.
//!
//! When a file exceeds the configured `chunk_threshold`, this module splits
//! the download into `connections_per_file` parallel HTTP Range requests,
//! each downloading a byte range concurrently. Chunks are written to a
//! pre-allocated temporary file, then placed into the `hf-hub` cache layout
//! for compatibility.

use std::io::SeekFrom;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use reqwest::Client;
use tokio::io::{AsyncSeekExt, AsyncWriteExt};
use tokio::sync::Mutex as AsyncMutex;
use tokio::task::JoinSet;

use serde::Deserialize;

use crate::chunked_state::ChunkedState;
use crate::error::FetchError;
use crate::progress::{self, ProgressEvent};
use crate::retry::{self, RetryPolicy};

/// Per-chunk progress checkpoint cadence: each chunk persists its
/// `completed` byte count to the sidecar after this many new bytes
/// arrive. Smaller values give finer-grained resume but more I/O on the
/// sidecar; 16 MiB lands roughly one save every 1.6 s on a 10 MiB/s
/// connection per chunk.
const SIDECAR_CHECKPOINT_BYTES: u64 = 16 * 1024 * 1024;

// TRAIT_OBJECT: heterogeneous progress handlers from different callers
type ProgressCallback = Arc<dyn Fn(&ProgressEvent) + Send + Sync>;

/// Default `HuggingFace` API endpoint.
const HF_ENDPOINT: &str = "https://huggingface.co";

/// Maximum time to establish a TCP connection to the remote server.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Result of probing a URL for HTTP Range support and cache metadata.
pub(crate) struct RangeInfo {
    /// Total file size in bytes.
    pub content_length: u64,
    /// The commit SHA from `x-repo-commit` header.
    pub commit_hash: String,
    /// The etag used for the blob path in the `hf-hub` cache.
    pub etag: String,
    /// The CDN URL to use for Range requests (after redirect).
    pub cdn_url: String,
    /// When the CDN signed URL expires, parsed from `X-Amz-Expires`.
    ///
    /// `None` if the URL has no recognizable expiry parameter.
    pub cdn_expires_at: Option<Instant>,
}

/// Constructs the HF download URL for a model file.
#[must_use]
pub(crate) fn build_download_url(repo_id: &str, revision: &str, filename: &str) -> String {
    // BORROW: explicit .replace() for URL-encoding
    let url_revision = revision.replace('/', "%2F");
    format!("{HF_ENDPOINT}/{repo_id}/resolve/{url_revision}/{filename}")
}

/// Builds a `reqwest::Client` with no-redirect policy for probing.
///
/// The client enforces a 30-second TCP connect timeout ([`CONNECT_TIMEOUT`]).
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the client or auth header cannot be constructed.
pub(crate) fn build_no_redirect_client(token: Option<&str>) -> Result<Client, FetchError> {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::USER_AGENT,
        reqwest::header::HeaderValue::from_static("hf-fetch-model"),
    );
    if let Some(tok) = token {
        let auth_value = format!("Bearer {tok}");
        // BORROW: explicit .as_str() for String → &str conversion
        let header_val = reqwest::header::HeaderValue::from_str(auth_value.as_str())
            .map_err(|e| FetchError::Http(e.to_string()))?;
        headers.insert(reqwest::header::AUTHORIZATION, header_val);
    }
    Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(CONNECT_TIMEOUT)
        .default_headers(headers)
        .build()
        .map_err(|e| FetchError::Http(e.to_string()))
}

/// Builds a `reqwest::Client` with auth token, user-agent, and 30-second
/// TCP connect timeout.
///
/// Use this to create a shared client for [`crate::repo::list_repo_files_with_metadata`]
/// and other API calls that benefit from connection reuse.
///
/// # Errors
///
/// Returns [`FetchError::Http`](crate::FetchError::Http) if the client or auth header cannot be constructed.
pub fn build_client(token: Option<&str>) -> Result<Client, FetchError> {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::USER_AGENT,
        reqwest::header::HeaderValue::from_static("hf-fetch-model"),
    );
    if let Some(tok) = token {
        let auth_value = format!("Bearer {tok}");
        // BORROW: explicit .as_str() for String → &str conversion
        let header_val = reqwest::header::HeaderValue::from_str(auth_value.as_str())
            .map_err(|e| FetchError::Http(e.to_string()))?;
        headers.insert(reqwest::header::AUTHORIZATION, header_val);
    }
    Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .default_headers(headers)
        .build()
        .map_err(|e| FetchError::Http(e.to_string()))
}

/// Probes the HF download URL for Range support and extracts cache metadata.
///
/// Sends a `Range: bytes=0-0` request mirroring `hf-hub`'s `metadata()` method.
/// Extracts `x-repo-commit` (commit hash) and `x-linked-etag`/`etag` from the
/// HF API response, then follows the redirect to the CDN to get the file size
/// from `Content-Range`. Also parses `X-Amz-Expires` from the CDN signed URL
/// to populate [`RangeInfo::cdn_expires_at`].
///
/// Returns `None` if Range requests are not supported.
///
/// # Errors
///
/// Returns [`FetchError::Http`] on network or header parsing failures.
pub(crate) async fn probe_range_support(
    client: Client,
    url: String,
    token: Option<String>,
) -> Result<Option<RangeInfo>, FetchError> {
    // Build a no-redirect client with the same auth for the initial probe.
    let no_redirect_client = build_no_redirect_client(token.as_deref())?;

    // BORROW: explicit .as_str() for request URL
    let response = no_redirect_client
        .get(url.as_str())
        .header(reqwest::header::RANGE, "bytes=0-0")
        .send()
        .await
        .map_err(|e| FetchError::Http(e.to_string()))?;

    // If not a redirect or partial content, Range may not be supported via this pattern.
    // Try the regular client path which follows redirects.
    let (hf_headers, redirect_url) = if response.status().is_redirection() {
        let headers = response.headers().clone();
        let location = headers
            .get(reqwest::header::LOCATION)
            .and_then(|v| v.to_str().ok())
            // BORROW: explicit .to_owned() for owned String
            .map(str::to_owned);
        (headers, location)
    } else if response.status() == reqwest::StatusCode::PARTIAL_CONTENT {
        // No redirect — the server responded directly with 206.
        let headers = response.headers().clone();
        (headers, None)
    } else {
        // Server does not support Range requests for this file.
        return Ok(None);
    };

    // Extract commit_hash from x-repo-commit header.
    let commit_hash = hf_headers
        .get("x-repo-commit")
        .and_then(|v| v.to_str().ok())
        // BORROW: explicit .to_owned() for owned String
        .map(str::to_owned)
        .ok_or_else(|| FetchError::Http("missing x-repo-commit header".to_owned()))?;

    // Extract etag: prefer x-linked-etag, fall back to etag.
    let etag = hf_headers
        .get("x-linked-etag")
        .or_else(|| hf_headers.get(reqwest::header::ETAG))
        .and_then(|v| v.to_str().ok())
        // BORROW: explicit .to_owned() for owned String
        .map(str::to_owned)
        .ok_or_else(|| FetchError::Http("missing etag header".to_owned()))?;
    // Clean extra quotes (same as hf-hub does).
    let etag = etag.replace('"', "");

    // Follow redirect to CDN and get Content-Range for file size.
    let (cdn_url, content_length) = if let Some(ref loc) = redirect_url {
        // BORROW: explicit .as_str() for request URL
        let cdn_response = client
            .get(loc.as_str())
            .header(reqwest::header::RANGE, "bytes=0-0")
            .send()
            .await
            .map_err(|e| FetchError::Http(e.to_string()))?;

        let size = parse_content_length_from_range(&cdn_response)?;
        // BORROW: explicit .clone() for owned String
        (loc.clone(), size)
    } else {
        // No redirect — parse size from the direct response headers.
        // We need to re-request since we consumed the response.
        let direct_response = client
            .get(url.as_str())
            .header(reqwest::header::RANGE, "bytes=0-0")
            .send()
            .await
            .map_err(|e| FetchError::Http(e.to_string()))?;

        let size = parse_content_length_from_range(&direct_response)?;
        (url, size)
    };

    let cdn_expires_at = parse_cdn_expiry(&cdn_url);

    Ok(Some(RangeInfo {
        content_length,
        commit_hash,
        etag,
        cdn_url,
        cdn_expires_at,
    }))
}

/// Parses the total file size from a `Content-Range: bytes 0-0/{size}` header.
fn parse_content_length_from_range(response: &reqwest::Response) -> Result<u64, FetchError> {
    let content_range = response
        .headers()
        .get(reqwest::header::CONTENT_RANGE)
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| FetchError::Http("missing Content-Range header".to_owned()))?;

    // Format: "bytes 0-0/{total}"
    content_range
        .split('/')
        .next_back()
        .and_then(|s| s.parse::<u64>().ok())
        .ok_or_else(|| FetchError::Http(format!("invalid Content-Range header: {content_range}")))
}

/// Parses the expiry deadline from an AWS presigned URL's `X-Amz-Expires` parameter.
///
/// Returns the approximate expiry instant, assuming the URL was just issued by
/// the CDN. Returns `None` if the parameter is absent or unparseable — this
/// includes non-AWS CDNs (e.g., GCS uses `X-Goog-Expires`, Cloudflare uses
/// proprietary tokens) where the re-probe path is silently skipped.
fn parse_cdn_expiry(url: &str) -> Option<Instant> {
    let query = url.split('?').nth(1)?;
    let expires_str = query
        .split('&')
        .find_map(|param| param.strip_prefix("X-Amz-Expires="))?;
    let seconds: u64 = expires_str.parse().ok()?;
    Some(Instant::now() + Duration::from_secs(seconds))
}

/// Downloads a file using parallel Range requests and writes it to the `hf-hub` cache.
///
/// Pre-allocates a `.chunked.part` temp file protected by a [`TempFileGuard`].
/// On successful finalization the temp file is renamed to its blob path
/// (so `Drop` finds nothing left to clean up). On transient failures —
/// timeout-induced future drop, Ctrl-C, panic, retryable chunk error —
/// the partial bytes are preserved for a future resume; only confirmed
/// corruption (e.g. an etag mismatch detected on resume) wipes via
/// [`TempFileGuard::mark_corrupt`].
///
/// # Arguments
///
/// * `client` — HTTP client with auth headers.
/// * `range_info` — Probe result with CDN URL, size, commit hash, and etag.
/// * `cache_dir` — Root of the HF cache (e.g., `~/.cache/huggingface/hub/`).
/// * `repo_folder` — Repo folder name (e.g., `"models--google--gemma-2-2b"`).
/// * `revision` — Branch/tag name for the refs file (e.g., `"main"`).
/// * `filename` — Relative filename in the repo (e.g., `"model.safetensors"`).
/// * `connections` — Number of parallel connections.
/// * `retry_policy` — Retry policy for individual chunks.
/// * `on_progress` — Optional progress callback.
/// * `files_remaining` — Files remaining after this one (for progress events).
///
/// # Errors
///
/// Returns [`FetchError::ChunkedDownload`] if any chunk fails after retries.
/// Returns [`FetchError::Io`] on filesystem errors.
// EXPLICIT: orchestrates per-file path setup, resume detection,
// chunk-boundary computation, JoinSet spawn/collect, and finalize.
// Splitting fragments the download lifecycle. The indexing_slicing
// allow covers `resume_offsets[chunk_idx]` — chunk_idx is in
// [0, connections) by construction and resume_offsets always has
// `connections` entries (`prepare_or_resume_temp_file` enforces it
// via `is_compatible_with`'s chunks.len() check).
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::indexing_slicing
)]
pub(crate) async fn download_chunked(
    client: Client,
    range_info: RangeInfo,
    cache_dir: PathBuf,
    repo_folder: String,
    revision: String,
    filename: String,
    connections: usize,
    retry_policy: RetryPolicy,
    on_progress: Option<ProgressCallback>,
    files_remaining: usize,
) -> Result<PathBuf, FetchError> {
    let total_size = range_info.content_length;

    // Build cache paths following hf-hub layout.
    let repo_dir = cache_dir.join(repo_folder.as_str());
    let blob_path = crate::cache_layout::blob_path(&repo_dir, range_info.etag.as_str());
    let pointer_path = crate::cache_layout::pointer_path(
        &repo_dir,
        range_info.commit_hash.as_str(),
        filename.as_str(),
    );

    // If the pointer path already exists, the file is cached — skip download.
    if pointer_path.exists() {
        return Ok(pointer_path);
    }

    // Compute chunk boundaries first — we need them for both fresh-start
    // and resume paths so the sidecar can be primed with the correct
    // per-chunk byte ranges.
    // EXPLICIT: try_from for usize → u64 (infallible on 64-bit, safe fallback otherwise)
    let chunk_size = total_size / u64::try_from(connections).unwrap_or(1);
    let chunks_layout: Vec<(usize, u64, u64)> = (0..connections)
        .map(|i| {
            // EXPLICIT: try_from for usize → u64 (infallible on 64-bit, safe fallback otherwise)
            let idx = u64::try_from(i).unwrap_or(0);
            let start = idx * chunk_size;
            let end = if i == connections - 1 {
                total_size - 1
            } else {
                (idx + 1) * chunk_size - 1
            };
            (i, start, end)
        })
        .collect();

    // Prepare or resume the partial. On match the existing `.chunked.part`
    // is left untouched and we get back the per-chunk completion offsets;
    // on mismatch (stale etag, different connection count, etc.) the
    // partial and sidecar are removed and a fresh state is written.
    let temp_path = crate::cache_layout::temp_blob_path(&repo_dir, range_info.etag.as_str());
    let state_path = crate::cache_layout::temp_state_path(&repo_dir, range_info.etag.as_str());
    let resume_state = prepare_or_resume_temp_file(
        &blob_path,
        &pointer_path,
        &temp_path,
        &state_path,
        range_info.etag.as_str(),
        total_size,
        chunks_layout.as_slice(),
        connections,
    )
    .await?;
    let _temp_guard = TempFileGuard::new(temp_path.clone());

    // Pre-charge the global byte counter with bytes already on disk from
    // a prior session, so the progress callback shows correct totals
    // including resumed bytes.
    let already_done: u64 = resume_state.chunks.iter().map(|c| c.completed).sum();
    let bytes_downloaded = Arc::new(AtomicU64::new(already_done));

    // Per-chunk completion offsets (relative to chunk start), captured
    // before the state moves into the shared mutex.
    let resume_offsets: Vec<u64> = resume_state.chunks.iter().map(|c| c.completed).collect();

    // Shared sidecar state: each chunk task locks it briefly to update
    // its own `completed` field and snapshot for atomic save.
    let shared_state = Arc::new(AsyncMutex::new(resume_state));

    // Spawn chunk download tasks.
    let mut join_set = JoinSet::new();
    for (chunk_idx, start, end) in chunks_layout {
        let task_client = client.clone();
        // BORROW: explicit .clone() for owned String
        let task_url = range_info.cdn_url.clone();
        // BORROW: explicit .clone() for owned PathBuf
        let task_temp = temp_path.clone();
        let task_state_path = state_path.clone();
        let task_state = Arc::clone(&shared_state);
        let task_policy = retry_policy.clone();
        let task_bytes = Arc::clone(&bytes_downloaded);
        let task_progress = on_progress.clone();
        // BORROW: explicit .clone() for owned String
        let task_filename = filename.clone();
        // INDEX: chunk_idx is in [0, connections); resume_offsets has connections entries
        let task_initial_offset = resume_offsets[chunk_idx];

        join_set.spawn(async move {
            download_chunk(
                task_client,
                task_url,
                task_temp,
                start,
                end,
                chunk_idx,
                task_initial_offset,
                task_state,
                task_state_path,
                &task_policy,
                &task_bytes,
                task_progress.as_ref(),
                task_filename.as_str(),
                total_size,
                files_remaining,
            )
            .await
        });
    }

    // Collect results.
    let mut failures: Vec<String> = Vec::new();
    while let Some(join_result) = join_set.join_next().await {
        match join_result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => failures.push(e.to_string()),
            Err(e) => failures.push(format!("chunk task failed: {e}")),
        }
    }

    if !failures.is_empty() {
        // Chunk failures are treated as transient — partial bytes on disk
        // remain valid-but-incomplete and a future invocation can resume
        // from them. `temp_guard` drops in keep-on-drop mode, leaving the
        // `.chunked.part` file in place. Use `hf-fm cache clean-partial`
        // to remove it manually if the user has abandoned the download.
        return Err(FetchError::ChunkedDownload {
            // BORROW: explicit .clone() for owned String
            filename: filename.clone(),
            reason: failures.join("; "),
        });
    }

    finalize_chunked_download(
        &temp_path,
        &blob_path,
        &pointer_path,
        &repo_dir,
        revision.as_str(),
        range_info.commit_hash.as_str(),
    )
    .await?;

    // The chunked download succeeded — the sidecar's job is done. Best-
    // effort removal: a stale sidecar without its partial (the partial
    // has just been renamed away) is harmless to leave behind and would
    // be discarded on the next invocation by the temp-exists check in
    // `prepare_or_resume_temp_file`.
    let _ = ChunkedState::remove(&state_path).await;

    // `finalize_chunked_download` renamed the temp file to `blob_path`,
    // so when `_temp_guard` drops at the end of this scope its keep-on-drop
    // policy is moot — there is nothing left at `temp_path` to act on.
    Ok(pointer_path)
}

/// RAII guard for a `.chunked.part` temp file.
///
/// Default policy is **keep on drop** so that partial bytes survive transient
/// interruptions (timeout-induced future drop, Ctrl-C, panic) and remain
/// available for resume on the next invocation. Callers that detect genuine
/// corruption (e.g. an etag mismatch on resume) opt into wipe-on-drop by
/// calling [`mark_corrupt`]; in that case `Drop` removes the file.
///
/// Note: a successful finalize renames the temp file to its blob path before
/// the guard drops, so `Drop` finds nothing at `path` and the keep-default
/// is harmless. No `commit` method is needed — keep-by-default is the
/// happy path.
///
/// [`mark_corrupt`]: TempFileGuard::mark_corrupt
struct TempFileGuard {
    /// Absolute path to the `.chunked.part` temp file this guard owns.
    path: PathBuf,
    /// When `true`, `Drop` removes the file at `path`. Defaults to `false`
    /// (preserve partials on transient failures); set explicitly via
    /// [`mark_corrupt`](TempFileGuard::mark_corrupt).
    wipe_on_drop: bool,
}

impl TempFileGuard {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            wipe_on_drop: false,
        }
    }

    /// Marks the partial as corrupt — `Drop` will remove it.
    ///
    /// Call when the bytes already on disk are known to be unusable — etag
    /// mismatch on resume, total-size mismatch in the sidecar, or a
    /// finalization-time checksum failure. Transient interruptions
    /// (timeout, Ctrl-C, retryable I/O) must NOT call this — their bytes
    /// are valid-but-incomplete and a future invocation can resume from
    /// them.
    //
    // Currently exercised only from the unit tests in this module; the
    // resume path that detects corruption (Phase 3) will introduce
    // production callers. The `dead_code` allow falls away then.
    #[allow(dead_code)]
    fn mark_corrupt(&mut self) {
        self.wipe_on_drop = true;
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if self.wipe_on_drop {
            // Sync remove is safe here: runs on the aborting thread, single syscall.
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Prepares the temp file for a chunked download, resuming from an existing
/// `.chunked.part` + `.chunked.part.state` pair when their resume invariants
/// (schema version, etag, total size, connection count) match the current
/// download configuration; otherwise truncates and starts fresh.
///
/// Always returns a `ChunkedState` describing the per-chunk completion
/// offsets to use — zeroed for a fresh start, populated for resume.
///
/// On a mismatch path, both the partial and the sidecar are removed
/// best-effort (the partial bytes are useless against a new etag or total
/// size; keeping them would just waste disk and confuse the next run).
// EXPLICIT: each parameter is genuinely independent (paths × identity
// invariants × layout) and packing them into a struct would only push
// the count to a fresh per-call constructor. Single private callsite.
#[allow(clippy::too_many_arguments)]
async fn prepare_or_resume_temp_file(
    blob_path: &Path,
    pointer_path: &Path,
    temp_path: &Path,
    state_path: &Path,
    etag: &str,
    total_size: u64,
    chunks_layout: &[(usize, u64, u64)],
    connections: usize,
) -> Result<ChunkedState, FetchError> {
    if let Some(parent) = blob_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| FetchError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
    }
    if let Some(parent) = pointer_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| FetchError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
    }

    // Try to resume from an existing partial + sidecar pair.
    let existing_state = ChunkedState::load(state_path).await?;
    let temp_exists = tokio::fs::try_exists(temp_path)
        .await
        .map_err(|e| FetchError::Io {
            path: temp_path.to_path_buf(),
            source: e,
        })?;

    if let Some(state) = existing_state {
        if state.is_compatible_with(etag, total_size, connections) && temp_exists {
            // Resume: leave the existing partial untouched and reuse the
            // per-chunk offsets recorded in the sidecar.
            return Ok(state);
        }
    }

    // Either no usable sidecar, an incompatible one, or the partial is
    // missing — start fresh. Best-effort cleanup of stale state.
    let _ = tokio::fs::remove_file(temp_path).await;
    ChunkedState::remove(state_path).await?;

    let file = tokio::fs::File::create(temp_path)
        .await
        .map_err(|e| FetchError::Io {
            path: temp_path.to_path_buf(),
            source: e,
        })?;
    file.set_len(total_size).await.map_err(|e| FetchError::Io {
        path: temp_path.to_path_buf(),
        source: e,
    })?;
    drop(file);

    // BORROW: explicit .to_owned() for &str → owned String
    let fresh = ChunkedState::new_fresh(etag.to_owned(), total_size, connections, chunks_layout);
    fresh.save_atomic(state_path).await?;
    Ok(fresh)
}

/// Finalizes a chunked download: renames temp → blob, creates pointer symlink, writes refs.
async fn finalize_chunked_download(
    temp_path: &std::path::Path,
    blob_path: &std::path::Path,
    pointer_path: &std::path::Path,
    repo_dir: &std::path::Path,
    revision: &str,
    commit_hash: &str,
) -> Result<(), FetchError> {
    // Rename temp file to blob path.
    tokio::fs::rename(temp_path, blob_path)
        .await
        .map_err(|e| FetchError::Io {
            path: blob_path.to_path_buf(),
            source: e,
        })?;

    // Create pointer symlink (or copy on Windows).
    symlink_or_copy(blob_path, pointer_path).map_err(|e| FetchError::Io {
        path: pointer_path.to_path_buf(),
        source: e,
    })?;

    // Write refs file.
    let refs_dir = crate::cache_layout::refs_dir(repo_dir);
    tokio::fs::create_dir_all(&refs_dir)
        .await
        .map_err(|e| FetchError::Io {
            // BORROW: explicit .clone() for owned PathBuf
            path: refs_dir.clone(),
            source: e,
        })?;
    let ref_path = crate::cache_layout::ref_path(repo_dir, revision);
    tokio::fs::write(&ref_path, commit_hash.as_bytes())
        .await
        .map_err(|e| FetchError::Io {
            path: ref_path,
            source: e,
        })?;

    Ok(())
}

/// Downloads a single byte-range chunk, writing to the temp file at the correct offset.
// EXPLICIT: linear retry-loop body composing range-request, file-seek,
// stream-write, in-flight progress, and sidecar checkpointing. Splitting
// hides the chunk lifecycle. The indexing_slicing allow covers three
// `state.chunks[chunk_idx]` reads/writes — chunk_idx is in
// [0, connections), and the shared state's `chunks` vector always has
// exactly `connections` entries (priming and resume both guarantee it).
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::indexing_slicing
)]
async fn download_chunk(
    client: Client,
    url: String,
    temp_path: PathBuf,
    start: u64,
    end: u64,
    chunk_idx: usize,
    initial_offset: u64,
    shared_state: Arc<AsyncMutex<ChunkedState>>,
    state_path: PathBuf,
    retry_policy: &RetryPolicy,
    bytes_downloaded: &AtomicU64,
    on_progress: Option<&ProgressCallback>,
    filename: &str,
    total_size: u64,
    files_remaining: usize,
) -> Result<(), FetchError> {
    // BORROW: explicit .clone()/.to_owned() for owned values in retry closure
    let url_owned = url.clone();
    let temp_owned = temp_path.clone();
    let filename_owned = filename.to_owned();

    retry::retry_async(retry_policy, retry::is_retryable, || {
        let task_client = client.clone();
        // BORROW: explicit .clone() for owned String
        let task_url = url_owned.clone();
        // BORROW: explicit .clone() for owned PathBuf
        let task_temp = temp_owned.clone();
        let task_state_path = state_path.clone();
        let task_state = Arc::clone(&shared_state);
        let task_filename = filename_owned.clone();

        async move {
            // Re-read the current completion offset for this chunk on
            // every retry attempt: an earlier attempt may have made
            // progress that survived in the sidecar even if this attempt's
            // overall result is `Err`.
            let (resume_completed, already_done) = {
                let guard = task_state.lock().await;
                // INDEX: chunk_idx is in [0, connections); state.chunks has connections entries
                let progress = &guard.chunks[chunk_idx];
                (progress.completed, progress.is_complete())
            };
            if already_done {
                // Chunk already fully downloaded by a prior session.
                return Ok(());
            }
            let resume_byte = start.saturating_add(resume_completed.max(initial_offset));
            let effective_resume_completed = resume_byte.saturating_sub(start);

            let range_header = format!("bytes={resume_byte}-{end}");
            // BORROW: explicit .as_str() for request URL and header
            let response = task_client
                .get(task_url.as_str())
                .header(reqwest::header::RANGE, range_header.as_str())
                .send()
                .await
                .map_err(|e| FetchError::ChunkedDownload {
                    filename: task_filename.clone(),
                    reason: format!("chunk {chunk_idx} request failed: {e}"),
                })?;

            if !response.status().is_success() {
                return Err(FetchError::ChunkedDownload {
                    filename: task_filename.clone(),
                    reason: format!("chunk {chunk_idx} HTTP {}", response.status()),
                });
            }

            // Open file and seek to the resume byte (= start + completed).
            let mut file = tokio::fs::OpenOptions::new()
                .write(true)
                .open(&task_temp)
                .await
                .map_err(|e| FetchError::Io {
                    path: task_temp.clone(),
                    source: e,
                })?;
            file.seek(SeekFrom::Start(resume_byte))
                .await
                .map_err(|e| FetchError::Io {
                    path: task_temp.clone(),
                    source: e,
                })?;

            // Stream bytes and write to file. Track in-attempt completion
            // locally so we can checkpoint the sidecar without locking on
            // every batch.
            let mut stream = response.bytes_stream();
            let mut current_completed = effective_resume_completed;
            let mut last_checkpoint = current_completed;
            while let Some(chunk_result) = stream.next().await {
                let bytes = chunk_result.map_err(|e| FetchError::ChunkedDownload {
                    filename: task_filename.clone(),
                    reason: format!("chunk {chunk_idx} stream error: {e}"),
                })?;

                file.write_all(&bytes).await.map_err(|e| FetchError::Io {
                    path: task_temp.clone(),
                    source: e,
                })?;

                // EXPLICIT: try_from for usize → u64 (infallible on 64-bit, safe fallback otherwise)
                let added = u64::try_from(bytes.len()).unwrap_or(0);
                current_completed = current_completed.saturating_add(added);
                let current = bytes_downloaded.fetch_add(added, Ordering::Relaxed) + added;

                // Fire progress callback (throttled by caller if needed).
                if let Some(cb) = on_progress {
                    let event = progress::streaming_event(
                        task_filename.as_str(),
                        current,
                        total_size,
                        files_remaining,
                    );
                    cb(&event);
                }

                // Checkpoint the sidecar every SIDECAR_CHECKPOINT_BYTES of
                // chunk progress. We snapshot the state under the lock and
                // do the (slow) atomic save outside the lock so other
                // chunks aren't blocked on our I/O.
                if current_completed.saturating_sub(last_checkpoint) >= SIDECAR_CHECKPOINT_BYTES {
                    let snapshot = {
                        let mut guard = task_state.lock().await;
                        // INDEX: chunk_idx is in [0, connections); state.chunks has connections entries
                        guard.chunks[chunk_idx].completed = current_completed;
                        guard.clone()
                    };
                    // Best-effort: a sidecar save failure is logged via
                    // the FetchError display but does not abort the
                    // chunk download — the in-memory state is still
                    // authoritative for the rest of this run.
                    let _ = snapshot.save_atomic(task_state_path.as_path()).await;
                    last_checkpoint = current_completed;
                }
            }

            file.flush().await.map_err(|e| FetchError::Io {
                path: task_temp,
                source: e,
            })?;

            // Final checkpoint: persist the chunk's terminal completion
            // count so a successful chunk's progress is durable even if
            // a later chunk fails.
            let snapshot = {
                let mut guard = task_state.lock().await;
                // INDEX: chunk_idx is in [0, connections); state.chunks has connections entries
                guard.chunks[chunk_idx].completed = current_completed;
                guard.clone()
            };
            let _ = snapshot.save_atomic(task_state_path.as_path()).await;

            Ok(())
        }
    })
    .await
}

/// Minimal API response for resolving a revision to a commit SHA.
#[derive(Deserialize)]
struct ApiCommitInfo {
    sha: String,
}

/// Resolves the commit hash for a given revision, reading from the refs file
/// if available or fetching from the HF API otherwise.
///
/// When fetched from the API, the refs file is created for future use.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the API request fails.
/// Returns [`FetchError::Io`] if the refs file cannot be written.
async fn resolve_commit_hash(
    client: &Client,
    repo_id: &str,
    revision: &str,
    repo_dir: &Path,
) -> Result<String, FetchError> {
    let ref_path = crate::cache_layout::ref_path(repo_dir, revision);

    // Try reading from the refs file first (written by hf-hub for other files).
    if let Ok(hash) = tokio::fs::read_to_string(&ref_path).await {
        let trimmed = hash.trim().to_owned();
        if !trimmed.is_empty() {
            return Ok(trimmed);
        }
    }

    // Fetch from the HF API.
    let mut url = format!("{HF_ENDPOINT}/api/models/{repo_id}");
    if revision != "main" {
        url = format!("{url}?revision={revision}");
    }

    // BORROW: explicit .as_str() for request URL
    let response = client
        .get(url.as_str())
        .send()
        .await
        .map_err(|e| FetchError::Http(format!("resolve commit hash: {e}")))?;

    if !response.status().is_success() {
        return Err(FetchError::Http(format!(
            "resolve commit hash: HTTP {}",
            response.status()
        )));
    }

    let info: ApiCommitInfo = response
        .json()
        .await
        .map_err(|e| FetchError::Http(format!("resolve commit hash: {e}")))?;

    // Write refs file for future use.
    let refs_dir = crate::cache_layout::refs_dir(repo_dir);
    tokio::fs::create_dir_all(&refs_dir)
        .await
        .map_err(|e| FetchError::Io {
            path: refs_dir.clone(),
            source: e,
        })?;
    // BORROW: explicit .as_bytes() for String → &[u8] conversion
    tokio::fs::write(&ref_path, info.sha.as_bytes())
        .await
        .map_err(|e| FetchError::Io {
            path: ref_path,
            source: e,
        })?;

    Ok(info.sha)
}

/// Downloads a file via a simple GET (no Range headers) and writes to the `hf-hub` cache.
///
/// Used as a fallback when `hf-hub`'s `.get()` fails with HTTP 416 Range Not Satisfiable,
/// which happens for small git-stored files that don't support Range requests.
///
/// Reads the commit hash from the `refs/` file already written by `hf-hub` for other files
/// in the same repo, then downloads via a regular GET and writes directly to the snapshot
/// directory.
///
/// # Errors
///
/// Returns [`FetchError::Http`] on network failures.
/// Returns [`FetchError::Io`] on filesystem errors (including missing refs file).
pub(crate) async fn download_direct(
    client: &Client,
    repo_id: &str,
    revision: &str,
    filename: &str,
    cache_dir: &Path,
) -> Result<PathBuf, FetchError> {
    let repo_dir = crate::cache_layout::repo_dir(cache_dir, repo_id);

    // Resolve the commit hash (from refs file or HF API).
    let commit_hash = resolve_commit_hash(client, repo_id, revision, &repo_dir).await?;

    // Build the pointer path in the snapshot directory.
    let pointer_path = crate::cache_layout::pointer_path(&repo_dir, commit_hash.as_str(), filename);

    // If already cached, skip.
    if pointer_path.exists() {
        return Ok(pointer_path);
    }

    // Download the file content with a simple GET (client follows redirects).
    let url = build_download_url(repo_id, revision, filename);
    // BORROW: explicit .as_str() for request URL
    let response = client
        .get(url.as_str())
        .send()
        .await
        .map_err(|e| FetchError::Http(format!("direct download of {filename}: {e}")))?;

    if !response.status().is_success() {
        return Err(FetchError::Http(format!(
            "direct download of {filename}: HTTP {}",
            response.status()
        )));
    }

    let content = response
        .bytes()
        .await
        .map_err(|e| FetchError::Http(format!("direct download of {filename}: {e}")))?;

    // Create directory and write file directly to snapshot path.
    if let Some(parent) = pointer_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| FetchError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
    }

    tokio::fs::write(&pointer_path, &content)
        .await
        .map_err(|e| FetchError::Io {
            // BORROW: explicit .clone() for owned PathBuf
            path: pointer_path.clone(),
            source: e,
        })?;

    Ok(pointer_path)
}

/// Computes a relative path from `dst`'s parent to `src`, for symlink creation.
///
/// Mirrors `hf-hub`'s `make_relative()` logic.
fn make_relative(src: &Path, dst: &Path) -> PathBuf {
    let src_components: Vec<Component<'_>> = src.components().collect();
    let dst_parent = dst.parent().unwrap_or(dst);
    let dst_components: Vec<Component<'_>> = dst_parent.components().collect();

    // Find the common prefix length.
    let common_len = src_components
        .iter()
        .zip(dst_components.iter())
        .take_while(|(a, b)| a == b)
        .count();

    // Go up from dst to the common ancestor.
    let mut rel = PathBuf::new();
    for _ in common_len..dst_components.len() {
        rel.push(Component::ParentDir);
    }
    // Then down from the common ancestor to src.
    for comp in src_components.iter().skip(common_len) {
        rel.push(comp);
    }

    rel
}

/// Creates a symlink from `dst` pointing to `src`, or falls back to copy on Windows.
///
/// On Windows, if symlink creation fails (e.g., `SeCreateSymbolicLinkPrivilege` is
/// missing), the blob is **copied** rather than moved. This preserves the blob in
/// `blobs/<etag>` for cross-revision deduplication.
///
/// Diverges from `hf-hub`'s `symlink_or_rename()` which uses `rename` and destroys
/// the blob — see Finding 2 of the v0.9.5 security audit.
fn symlink_or_copy(src: &Path, dst: &Path) -> Result<(), std::io::Error> {
    if dst.exists() {
        return Ok(());
    }

    let rel_src = make_relative(src, dst);

    #[cfg(target_os = "windows")]
    {
        if std::os::windows::fs::symlink_file(&rel_src, dst).is_err() {
            // Copy rather than rename to preserve the blob for cross-revision deduplication.
            std::fs::copy(src, dst)?;
        }
    }

    #[cfg(target_family = "unix")]
    {
        // Tolerate EEXIST: a concurrent downloader may have created the symlink
        // between the dst.exists() check above and this call (TOCTOU race).
        if let Err(e) = std::os::unix::fs::symlink(rel_src, dst) {
            if e.kind() != std::io::ErrorKind::AlreadyExists {
                return Err(e);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::panic,
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing
    )]

    use super::*;

    #[test]
    fn test_repo_folder_name() {
        assert_eq!(
            crate::cache_layout::repo_folder_name("google/gemma-2-2b"),
            "models--google--gemma-2-2b"
        );
        assert_eq!(
            crate::cache_layout::repo_folder_name("RWKV/RWKV7-Goose-World3-1.5B-HF"),
            "models--RWKV--RWKV7-Goose-World3-1.5B-HF"
        );
    }

    #[test]
    fn test_build_download_url() {
        assert_eq!(
            build_download_url("google/gemma-2-2b", "main", "config.json"),
            "https://huggingface.co/google/gemma-2-2b/resolve/main/config.json"
        );
        assert_eq!(
            build_download_url("org/model", "refs/pr/42", "file.bin"),
            "https://huggingface.co/org/model/resolve/refs%2Fpr%2F42/file.bin"
        );
    }

    #[test]
    fn test_chunk_boundaries() {
        let total: u64 = 1000;
        let connections: usize = 4;
        let chunk_size = total / u64::try_from(connections).unwrap();

        let chunks: Vec<(u64, u64)> = (0..connections)
            .map(|i| {
                let idx = u64::try_from(i).unwrap();
                let start = idx * chunk_size;
                let end = if i == connections - 1 {
                    total - 1
                } else {
                    (idx + 1) * chunk_size - 1
                };
                (start, end)
            })
            .collect();

        assert_eq!(chunks[0], (0, 249));
        assert_eq!(chunks[1], (250, 499));
        assert_eq!(chunks[2], (500, 749));
        assert_eq!(chunks[3], (750, 999));
    }

    /// Default policy is keep-on-drop: a guard that is never marked corrupt
    /// must leave the file on disk when it goes out of scope. This is the
    /// behavior that lets a partial `.chunked.part` survive a transient
    /// timeout / Ctrl-C / panic and become resumable on the next run.
    #[test]
    fn temp_file_guard_keeps_file_on_drop_by_default() {
        let dir = std::env::temp_dir().join(format!("hf-fm-tempguard-keep-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("partial.chunked.part");
        std::fs::write(&path, b"some bytes").unwrap();
        assert!(path.exists());

        {
            let _guard = TempFileGuard::new(path.clone());
            // No mark_corrupt — guard drops in keep mode.
        }

        assert!(
            path.exists(),
            "default-drop should preserve the file at {}",
            path.display()
        );
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    /// Calling `mark_corrupt` flips the guard into wipe-on-drop mode: the
    /// file is removed when the guard goes out of scope. This is the path
    /// for confirmed-bad bytes (etag mismatch, total-size mismatch on
    /// resume), as opposed to "incomplete but valid" partials.
    #[test]
    fn temp_file_guard_wipes_file_after_mark_corrupt() {
        let dir = std::env::temp_dir().join(format!("hf-fm-tempguard-wipe-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("partial.chunked.part");
        std::fs::write(&path, b"corrupt bytes").unwrap();
        assert!(path.exists());

        {
            let mut guard = TempFileGuard::new(path.clone());
            guard.mark_corrupt();
        }

        assert!(
            !path.exists(),
            "mark_corrupt should wipe the file at {}",
            path.display()
        );
        std::fs::remove_dir(&dir).ok();
    }

    /// `mark_corrupt` on a guard whose file has already been moved away
    /// (the success path: `finalize_chunked_download` renamed temp → blob)
    /// must not panic — `remove_file` silently no-ops when the path is
    /// gone. Defensive sanity check on the Drop body.
    #[test]
    fn temp_file_guard_drop_is_safe_when_file_already_gone() {
        let dir = std::env::temp_dir().join(format!("hf-fm-tempguard-gone-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("never-existed.chunked.part");
        // Note: file is NOT created — simulating the post-rename state.

        {
            let mut guard = TempFileGuard::new(path.clone());
            guard.mark_corrupt();
            // Drop fires here — must not panic even though file is absent.
        }

        assert!(!path.exists());
        std::fs::remove_dir(&dir).ok();
    }
}
