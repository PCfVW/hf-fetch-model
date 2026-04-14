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
use tokio::task::JoinSet;

use serde::Deserialize;

use crate::error::FetchError;
use crate::progress::{self, ProgressEvent};
use crate::retry::{self, RetryPolicy};

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
/// Pre-allocates a `.chunked.part` temp file protected by a [`TempFileGuard`] — the
/// temp file is removed automatically on error or task abort (e.g., via
/// `JoinSet::abort_all()`), and committed only after successful finalization.
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
#[allow(clippy::too_many_arguments)]
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

    // Create directories and pre-allocate temp file.
    let temp_path = crate::cache_layout::temp_blob_path(&repo_dir, range_info.etag.as_str());
    prepare_temp_file(&blob_path, &pointer_path, &temp_path, total_size).await?;
    let mut temp_guard = TempFileGuard::new(temp_path.clone());

    // Compute chunk boundaries.
    // EXPLICIT: try_from for usize → u64 (infallible on 64-bit, safe fallback otherwise)
    let chunk_size = total_size / u64::try_from(connections).unwrap_or(1);
    let chunks: Vec<(usize, u64, u64)> = (0..connections)
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

    // Shared progress counter.
    let bytes_downloaded = Arc::new(AtomicU64::new(0));

    // Spawn chunk download tasks.
    let mut join_set = JoinSet::new();
    for (chunk_idx, start, end) in chunks {
        let task_client = client.clone();
        // BORROW: explicit .clone() for owned String
        let task_url = range_info.cdn_url.clone();
        // BORROW: explicit .clone() for owned PathBuf
        let task_temp = temp_path.clone();
        let task_policy = retry_policy.clone();
        let task_bytes = Arc::clone(&bytes_downloaded);
        let task_progress = on_progress.clone();
        // BORROW: explicit .clone() for owned String
        let task_filename = filename.clone();

        join_set.spawn(async move {
            download_chunk(
                task_client,
                task_url,
                task_temp,
                start,
                end,
                chunk_idx,
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
        // temp_guard drops here and removes the temp file.
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

    // Download and finalization succeeded — prevent guard from removing the file.
    temp_guard.commit();

    Ok(pointer_path)
}

/// RAII guard that removes a temp file on drop unless explicitly committed.
///
/// Ensures `.chunked.part` files are cleaned up even when a task is aborted
/// (e.g., via `JoinSet::abort_all()`), since `Drop` runs on abort.
struct TempFileGuard {
    /// Path to the temp file to remove on drop.
    path: PathBuf,
    /// Set to `true` after successful finalization to prevent removal.
    committed: bool,
}

impl TempFileGuard {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            committed: false,
        }
    }

    /// Marks the temp file as successfully finalized — `Drop` will not remove it.
    fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if !self.committed {
            // Sync remove is safe here: runs on the aborting thread, single syscall.
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Creates parent directories for blob and pointer paths, then pre-allocates a temp file.
async fn prepare_temp_file(
    blob_path: &std::path::Path,
    pointer_path: &std::path::Path,
    temp_path: &std::path::Path,
    total_size: u64,
) -> Result<(), FetchError> {
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

    let f = tokio::fs::File::create(temp_path)
        .await
        .map_err(|e| FetchError::Io {
            path: temp_path.to_path_buf(),
            source: e,
        })?;
    f.set_len(total_size).await.map_err(|e| FetchError::Io {
        path: temp_path.to_path_buf(),
        source: e,
    })?;

    Ok(())
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
#[allow(clippy::too_many_arguments)]
async fn download_chunk(
    client: Client,
    url: String,
    temp_path: PathBuf,
    start: u64,
    end: u64,
    chunk_idx: usize,
    retry_policy: &RetryPolicy,
    bytes_downloaded: &AtomicU64,
    on_progress: Option<&ProgressCallback>,
    filename: &str,
    total_size: u64,
    files_remaining: usize,
) -> Result<(), FetchError> {
    // BORROW: explicit .clone() for owned values in retry closure
    let url_owned = url.clone();
    let temp_owned = temp_path.clone();
    let filename_owned = filename.to_owned();

    retry::retry_async(retry_policy, retry::is_retryable, || {
        let task_client = client.clone();
        // BORROW: explicit .clone() for owned String
        let task_url = url_owned.clone();
        // BORROW: explicit .clone() for owned PathBuf
        let task_temp = temp_owned.clone();
        let task_filename = filename_owned.clone();

        async move {
            let range_header = format!("bytes={start}-{end}");
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

            // Open file and seek to chunk offset.
            let mut file = tokio::fs::OpenOptions::new()
                .write(true)
                .open(&task_temp)
                .await
                .map_err(|e| FetchError::Io {
                    path: task_temp.clone(),
                    source: e,
                })?;
            file.seek(SeekFrom::Start(start))
                .await
                .map_err(|e| FetchError::Io {
                    path: task_temp.clone(),
                    source: e,
                })?;

            // Stream bytes and write to file.
            let mut stream = response.bytes_stream();
            while let Some(chunk_result) = stream.next().await {
                let bytes = chunk_result.map_err(|e| FetchError::ChunkedDownload {
                    filename: task_filename.clone(),
                    reason: format!("chunk {chunk_idx} stream error: {e}"),
                })?;

                file.write_all(&bytes).await.map_err(|e| FetchError::Io {
                    path: task_temp.clone(),
                    source: e,
                })?;

                // Update shared progress counter.
                // EXPLICIT: try_from for usize → u64 (infallible on 64-bit, safe fallback otherwise)
                let added = u64::try_from(bytes.len()).unwrap_or(0);
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
            }

            file.flush().await.map_err(|e| FetchError::Io {
                path: task_temp,
                source: e,
            })?;

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
}
