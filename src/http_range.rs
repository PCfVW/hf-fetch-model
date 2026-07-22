// SPDX-License-Identifier: MIT OR Apache-2.0

//! HTTP Range `Read + Seek` substrate for remote inspection.
//!
//! [`RangeReader`] adapts a byte-range transport into [`std::io::Read`] +
//! [`std::io::Seek`], so format parsers that consume a reader (e.g.
//! `anamnesis::inspect_npz_from_reader`) can inspect a remote file's
//! metadata without downloading it. The transport is abstracted behind
//! [`RangeFetcher`]; production code uses [`HttpRangeFetcher`] (HTTP Range
//! requests against the `HuggingFace` CDN), and unit tests use an in-memory
//! fetcher — the buffering, budgeting, and `Seek` semantics are exercised
//! entirely offline.
//!
//! # Access pattern optimisations
//!
//! Tuned for archive-directory reads (`ZIP` central directory + per-entry
//! headers — the `NPZ` / `PTH` layout):
//!
//! - **Tail prefetch** — the first read landing in the last
//!   [`TAIL_PREFETCH_BYTES`] of the file fetches that whole tail once and
//!   caches it. The `ZIP` end-of-central-directory scan and the central
//!   directory itself are then served from memory.
//! - **Read-ahead** — short sequential reads are coalesced into
//!   [`READAHEAD_BYTES`]-sized window fetches, so parsing a 30-byte local
//!   header followed by a ~120-byte `NPY` header costs one request, not two.
//! - **No re-fetch** — reads covered by the cached tail or the current
//!   window never re-issue a request.
//!
//! # Safety budgets
//!
//! The reader enforces two hard caps — [`MAX_RANGE_REQUESTS`] requests and
//! [`MAX_TRANSFER_BUDGET`] bytes fetched — so a pathological or adversarial
//! archive layout (a seek storm, an absurdly large central directory) fails
//! fast with a clear error instead of degenerating into a full download.
//! [`RangeReader::with_limits`] overrides the defaults for callers with
//! different budgets.
//!
//! # Error channel
//!
//! `Read` / `Seek` must return [`std::io::Error`], which flattens the crate's
//! typed [`FetchError`]. The reader therefore stores the most recent fetch
//! failure; after a failed parse, [`RangeReader::take_last_error`] recovers
//! the typed error (e.g. an HTTP `403` that the CLI upgrades into a
//! gated-repo diagnosis).

use std::io::{self, Read, Seek, SeekFrom};
use std::time::Duration;

use serde::Serialize;

use crate::chunked;
use crate::error::FetchError;

// -----------------------------------------------------------------------
// Tuning constants
// -----------------------------------------------------------------------

/// Size of the cached file tail fetched by the first read near end-of-file.
///
/// 64 KiB covers the `ZIP` end-of-central-directory record (max comment
/// length 65 535 bytes) plus the central directory of typical `NPZ`
/// archives, so the whole directory scan costs one request.
pub const TAIL_PREFETCH_BYTES: u64 = 64 * 1024;

/// Minimum window size for a fetch serving a sequential read.
///
/// Short header-sized reads (tens of bytes) are widened to this size so
/// adjacent structures (`ZIP` local header + `NPY` header) arrive in one
/// request.
pub const READAHEAD_BYTES: u64 = 4 * 1024;

/// Default cap on total bytes fetched over the reader's lifetime.
///
/// Remote inspection of a well-formed archive transfers well under 1 MiB;
/// 32 MiB leaves two orders of magnitude of headroom (e.g. a many-thousand
/// entry central directory) while guaranteeing that no inspect silently
/// degenerates into a full-file download.
pub const MAX_TRANSFER_BUDGET: u64 = 32 * 1024 * 1024;

/// Default cap on the number of range fetches over the reader's lifetime.
///
/// A well-formed inspect needs a handful of requests; 256 tolerates a
/// fragmented directory layout while stopping request storms from
/// adversarial seek patterns.
pub const MAX_RANGE_REQUESTS: u32 = 256;

/// Wall-clock budget for a single HTTP range request (headers + body).
///
/// Bounds a stalled response body — the connect phase is separately bounded
/// by the client's TCP connect timeout.
const RANGE_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

// -----------------------------------------------------------------------
// Transport abstraction
// -----------------------------------------------------------------------

/// A transport that serves byte ranges of one remote file.
///
/// Implemented by [`HttpRangeFetcher`] for production and by in-memory
/// fetchers in tests. The contract: `fetch(start, end_inclusive)` returns
/// exactly `end_inclusive - start + 1` bytes of the file's content at
/// `start`, and `total_size` is the file's full length in bytes, known
/// up front.
pub trait RangeFetcher {
    /// Fetches the inclusive byte range `start..=end_inclusive`.
    ///
    /// # Errors
    ///
    /// Returns [`FetchError::Http`] when the transport fails (network
    /// error, unexpected HTTP status, validation mismatch).
    fn fetch(&mut self, start: u64, end_inclusive: u64) -> Result<Vec<u8>, FetchError>;

    /// Total size of the remote file in bytes.
    fn total_size(&self) -> u64;

    /// Requests spent by the transport outside [`RangeFetcher::fetch`]
    /// (e.g. the eager probe that resolved the file size and classified
    /// access). Included in [`RangeReader::stats`] so the reported request
    /// count is honest.
    fn extra_requests(&self) -> u32 {
        0
    }
}

/// Transfer statistics for one [`RangeReader`] lifetime.
///
/// Rendered by the CLI as provenance (e.g. `remote (6 range requests,
/// 136.0 KiB fetched)` — the live-measured cost of inspecting a 72 MiB
/// `NPZ`) — the on-screen proof that an inspect read metadata, not weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub struct RangeStats {
    /// Total HTTP requests issued: range fetches (attempts, including
    /// failed ones) plus the transport's probe requests.
    pub requests: u32,
    /// Total content bytes fetched across all successful range requests.
    /// Excludes HTTP header overhead and probe responses (single-byte
    /// probes; negligible).
    pub bytes_fetched: u64,
}

// -----------------------------------------------------------------------
// RangeReader core
// -----------------------------------------------------------------------

/// A cached contiguous extent of the remote file.
struct Extent {
    /// Absolute file offset of `data[0]`.
    start: u64,
    /// The fetched bytes.
    data: Vec<u8>,
}

impl Extent {
    /// One past the last absolute offset covered by this extent.
    fn end(&self) -> u64 {
        // CAST: usize → u64, buffer length always fits (usize ≤ 64 bits)
        #[allow(clippy::as_conversions)]
        self.start.saturating_add(self.data.len() as u64)
    }

    /// Whether `pos` falls inside this extent.
    fn contains(&self, pos: u64) -> bool {
        pos >= self.start && pos < self.end()
    }

    /// Copies bytes starting at absolute offset `pos` into `buf`, returning
    /// the count copied (`None` if `pos` is outside the extent).
    fn copy_at(&self, pos: u64, buf: &mut [u8]) -> Option<usize> {
        if !self.contains(pos) {
            return None;
        }
        let offset = usize::try_from(pos.checked_sub(self.start)?).ok()?;
        let available = self.data.len().checked_sub(offset)?;
        let n = buf.len().min(available);
        let src = self.data.get(offset..offset.checked_add(n)?)?;
        let dst = buf.get_mut(..n)?;
        dst.copy_from_slice(src);
        Some(n)
    }
}

/// `Read + Seek` over a [`RangeFetcher`], with tail caching, read-ahead,
/// and hard safety budgets.
///
/// See the [module docs](self) for the access-pattern and budget design.
pub struct RangeReader<F: RangeFetcher> {
    /// The transport serving byte ranges.
    fetcher: F,
    /// Current logical read position (may exceed the file size after a
    /// permissive seek; reads there return `Ok(0)`).
    pos: u64,
    /// Most recent read-ahead window.
    window: Option<Extent>,
    /// Cached file tail (`ZIP` end-of-central-directory region).
    tail: Option<Extent>,
    /// Range fetches attempted (successful or failed).
    requests: u32,
    /// Content bytes fetched across successful range fetches.
    bytes_fetched: u64,
    /// Lifetime cap on `requests`.
    max_requests: u32,
    /// Lifetime cap on `bytes_fetched`.
    max_transfer_bytes: u64,
    /// Most recent typed transport error (see [`Self::take_last_error`]).
    last_error: Option<FetchError>,
}

impl<F: RangeFetcher> RangeReader<F> {
    /// Creates a reader with the default safety budgets
    /// ([`MAX_RANGE_REQUESTS`], [`MAX_TRANSFER_BUDGET`]).
    #[must_use]
    pub const fn new(fetcher: F) -> Self {
        Self::with_limits(fetcher, MAX_RANGE_REQUESTS, MAX_TRANSFER_BUDGET)
    }

    /// Creates a reader with explicit safety budgets.
    ///
    /// `max_requests` caps the number of range fetches; `max_transfer_bytes`
    /// caps the total content bytes fetched. Exceeding either fails the
    /// offending read with a [`std::io::Error`] naming the cap.
    #[must_use]
    pub const fn with_limits(fetcher: F, max_requests: u32, max_transfer_bytes: u64) -> Self {
        Self {
            fetcher,
            pos: 0,
            window: None,
            tail: None,
            requests: 0,
            bytes_fetched: 0,
            max_requests,
            max_transfer_bytes,
            last_error: None,
        }
    }

    /// Transfer statistics so far (range fetches + transport probes).
    #[must_use]
    pub fn stats(&self) -> RangeStats {
        RangeStats {
            requests: self.requests.saturating_add(self.fetcher.extra_requests()),
            bytes_fetched: self.bytes_fetched,
        }
    }

    /// Total size of the remote file in bytes.
    #[must_use]
    pub fn total_size(&self) -> u64 {
        self.fetcher.total_size()
    }

    /// Takes the most recent typed transport error, if any.
    ///
    /// `Read` flattens [`FetchError`] into [`std::io::Error`] strings; after
    /// a parse fails, callers use this to recover the typed error (e.g. to
    /// let the CLI upgrade an HTTP `403` into a gated-repo diagnosis).
    #[must_use]
    pub const fn take_last_error(&mut self) -> Option<FetchError> {
        self.last_error.take()
    }

    /// Fetches `start..=end_inclusive` through the budget checks, recording
    /// stats and stashing typed errors.
    fn checked_fetch(&mut self, start: u64, end_inclusive: u64) -> io::Result<Vec<u8>> {
        let len = end_inclusive
            .checked_sub(start)
            .and_then(|d| d.checked_add(1))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("invalid range {start}..={end_inclusive}"),
                )
            })?;

        if self.requests >= self.max_requests {
            return Err(io::Error::other(format!(
                "range request cap exceeded ({} requests): pathological archive \
                 layout or seek storm; refusing further fetches",
                self.max_requests
            )));
        }
        if self.bytes_fetched.saturating_add(len) > self.max_transfer_bytes {
            return Err(io::Error::other(format!(
                "range transfer budget exceeded ({} bytes fetched, {len} more \
                 requested, cap {}): metadata inspection should never read \
                 this much; refusing further fetches",
                self.bytes_fetched, self.max_transfer_bytes
            )));
        }

        self.requests = self.requests.saturating_add(1);
        match self.fetcher.fetch(start, end_inclusive) {
            Ok(data) => {
                // CAST: usize → u64, buffer length always fits (usize ≤ 64 bits)
                #[allow(clippy::as_conversions)]
                let got = data.len() as u64;
                if got != len {
                    return Err(io::Error::other(format!(
                        "range fetcher returned {got} bytes for a {len}-byte \
                         range ({start}..={end_inclusive})"
                    )));
                }
                self.bytes_fetched = self.bytes_fetched.saturating_add(len);
                Ok(data)
            }
            Err(fetch_err) => {
                let io_err = io::Error::other(fetch_err.to_string());
                self.last_error = Some(fetch_err);
                Err(io_err)
            }
        }
    }

    /// Serves `buf` from the cached tail or window, if `pos` is covered.
    fn copy_cached(&self, buf: &mut [u8]) -> Option<usize> {
        if let Some(w) = &self.window {
            if let Some(n) = w.copy_at(self.pos, buf) {
                return Some(n);
            }
        }
        if let Some(t) = &self.tail {
            if let Some(n) = t.copy_at(self.pos, buf) {
                return Some(n);
            }
        }
        None
    }

    /// Advances the read position by `n` copied bytes.
    fn advance_pos(&mut self, n: usize) {
        // CAST: usize → u64, copy count bounded by buf length
        #[allow(clippy::as_conversions)]
        {
            self.pos = self.pos.saturating_add(n as u64);
        }
    }

    /// Copies from cache immediately after a fetch that must cover `pos`.
    ///
    /// A miss here means the fetch/cache bookkeeping is internally
    /// inconsistent; surfaced as an error rather than a panic.
    fn serve_from_cache_after_fetch(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.copy_cached(buf).ok_or_else(|| {
            io::Error::other(format!(
                "internal range cache inconsistency at offset {}",
                self.pos
            ))
        })?;
        self.advance_pos(n);
        Ok(n)
    }
}

impl<F: RangeFetcher> Read for RangeReader<F> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let total = self.fetcher.total_size();
        if self.pos >= total {
            return Ok(0); // EOF (also covers permissive past-end seeks)
        }

        // 1. Serve from cached extents (window, then tail) — no request.
        if let Some(n) = self.copy_cached(buf) {
            self.advance_pos(n);
            return Ok(n);
        }

        // 2. First read near end-of-file: prefetch and cache the tail.
        let tail_start = total.saturating_sub(TAIL_PREFETCH_BYTES);
        if self.tail.is_none() && self.pos >= tail_start {
            let data = self.checked_fetch(tail_start, total.saturating_sub(1))?;
            self.tail = Some(Extent {
                start: tail_start,
                data,
            });
            return self.serve_from_cache_after_fetch(buf);
        }

        // 3. Read-ahead window fetch at the current position.
        // CAST: usize → u64, buffer length always fits (usize ≤ 64 bits)
        #[allow(clippy::as_conversions)]
        let want = (buf.len() as u64).max(READAHEAD_BYTES);
        let mut end = self
            .pos
            .saturating_add(want)
            .saturating_sub(1)
            .min(total.saturating_sub(1));
        // Never overlap the cached tail — those bytes are already paid for.
        if let Some(t) = &self.tail {
            if self.pos < t.start {
                end = end.min(t.start.saturating_sub(1));
            }
        }
        let data = self.checked_fetch(self.pos, end)?;
        self.window = Some(Extent {
            start: self.pos,
            data,
        });
        self.serve_from_cache_after_fetch(buf)
    }
}

impl<F: RangeFetcher> Seek for RangeReader<F> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let total = self.fetcher.total_size();
        let target: i128 = match pos {
            SeekFrom::Start(p) => i128::from(p),
            SeekFrom::End(delta) => i128::from(total).saturating_add(i128::from(delta)),
            SeekFrom::Current(delta) => i128::from(self.pos).saturating_add(i128::from(delta)),
        };
        if target < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("seek to negative offset {target}"),
            ));
        }
        let new_pos = u64::try_from(target).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("seek offset {target} exceeds u64::MAX"),
            )
        })?;
        self.pos = new_pos;
        Ok(new_pos)
    }
}

// -----------------------------------------------------------------------
// HTTP transport
// -----------------------------------------------------------------------

/// [`RangeReader`] over [`HttpRangeFetcher`] — the production remote-inspect
/// substrate.
pub type HttpRangeReader = RangeReader<HttpRangeFetcher>;

/// HTTP Range transport against a `HuggingFace`-hosted file.
///
/// Opened via [`HttpRangeReader::open`]. One authenticated, redirect-free
/// probe (reusing the chunked-download probe) resolves the file size and
/// classifies access failures eagerly (gated repos, no Range support).
/// **Every range request then targets the HF `/resolve` URL** and follows
/// the redirect to a freshly-signed CDN URL; `reqwest` strips the
/// `Authorization` header on the cross-host redirect, so the `Bearer`
/// token is never sent to the CDN host.
///
/// Per-request re-resolution is **required**, not just convenient:
/// Xet-backed repos sign each CDN URL for the *exact* `Range` header that
/// minted it (a `ByteRange` policy condition in the signed URL), so a CDN
/// URL stored from the probe can never serve any other range. Re-resolving
/// also makes signed-URL expiry a non-issue — every request gets a fresh
/// signature. Discovered live against `google/gemma-scope-2b-pt-res`
/// (v0.11.0 dogfooding).
///
/// # Response validation (every request)
///
/// - Status must be `206 Partial Content`. A `200` (server ignoring the
///   `Range` header) is rejected **without reading the body** — the guard
///   against silently pulling the full file.
/// - `Content-Range` must match the requested range and the probed total
///   size.
/// - The body is streamed with a hard cap of the requested length;
///   over- or under-delivery is an error.
/// - The response `ETag` (when present) must stay identical across
///   requests — a mid-inspect change upstream aborts with a clear error.
pub struct HttpRangeFetcher {
    /// Runtime handle used to drive async `reqwest` calls from the
    /// blocking `Read` context (`spawn_blocking` thread).
    handle: tokio::runtime::Handle,
    /// Authenticated, redirect-following client. The auth header rides
    /// only to the HF host — `reqwest` removes sensitive headers when a
    /// redirect crosses hosts.
    client: reqwest::Client,
    /// The `HF` `/resolve` URL every range request targets.
    hf_url: String,
    /// Filename for error messages.
    filename: String,
    /// First `ETag` observed on a range response — later responses must
    /// match it (self-consistency across requests).
    response_etag: Option<String>,
    /// Total file size from the probe's `Content-Range`.
    total_size: u64,
    /// Probe requests spent (no-redirect probe + CDN size fetch).
    extra: u32,
}

impl HttpRangeReader {
    /// Opens a range reader over `filename` in `repo_id` at `revision`
    /// (default `main`).
    ///
    /// Performs the probe eagerly, so failures (missing file, gated repo,
    /// no Range support) surface here as typed errors rather than
    /// mid-parse `io::Error`s.
    ///
    /// Must be called from within a `tokio` runtime; the returned reader is
    /// then handed to a blocking context (`tokio::task::spawn_blocking`),
    /// where its `Read` / `Seek` impls drive async requests via the
    /// captured runtime handle.
    ///
    /// # Errors
    ///
    /// Returns [`FetchError::Http`] if the probe fails, the repository or
    /// file is inaccessible (including gated repos: `returned status
    /// 401/403` errors, which the CLI upgrades into a gated-repo
    /// diagnosis), or the server does not support Range requests.
    pub async fn open(
        repo_id: &str,
        revision: Option<&str>,
        filename: &str,
        token: Option<&str>,
    ) -> Result<Self, FetchError> {
        let fetcher = HttpRangeFetcher::open(repo_id, revision, filename, token).await?;
        Ok(RangeReader::new(fetcher))
    }
}

impl HttpRangeFetcher {
    /// Probes `filename` and constructs the transport (see
    /// [`HttpRangeReader::open`]).
    ///
    /// # Errors
    ///
    /// Returns [`FetchError::Http`] if the probe fails, the file is
    /// inaccessible, or the server does not support Range requests.
    async fn open(
        repo_id: &str,
        revision: Option<&str>,
        filename: &str,
        token: Option<&str>,
    ) -> Result<Self, FetchError> {
        let rev = revision.unwrap_or("main");
        let hf_url = chunked::build_download_url(repo_id, rev, filename);
        let client = chunked::build_client(token)?;

        let info = chunked::probe_range_support(
            client.clone(),
            hf_url.clone(),
            // BORROW: explicit String::from for Option<&str> → Option<String>
            token.map(String::from),
        )
        .await?;
        let Some(info) = info else {
            return Err(Self::classify_no_range_support(&client, &hf_url, filename).await);
        };

        Ok(Self {
            handle: tokio::runtime::Handle::current(),
            client,
            hf_url,
            // BORROW: explicit .to_owned() for the owned field
            filename: filename.to_owned(),
            response_etag: None,
            total_size: info.content_length,
            extra: 2, // no-redirect probe + CDN size fetch
        })
    }

    /// Distinguishes "no Range support" from an access failure after the
    /// probe declined.
    ///
    /// The chunked probe returns `None` for **any** non-redirect,
    /// non-`206` response — including a gated repo's `401`/`403`. One
    /// follow-up request recovers the real status so gated repos get the
    /// actionable diagnosis instead of a misleading "no Range support".
    async fn classify_no_range_support(
        client: &reqwest::Client,
        url: &str,
        filename: &str,
    ) -> FetchError {
        let result = client
            .get(url)
            .header(reqwest::header::RANGE, "bytes=0-0")
            .timeout(RANGE_REQUEST_TIMEOUT)
            .send()
            .await;
        match result {
            Ok(resp) => {
                let status = resp.status();
                if status.is_client_error() || status.is_server_error() {
                    FetchError::Http(format!(
                        "Range request for {filename} returned status {status}"
                    ))
                } else {
                    FetchError::Http(format!(
                        "server does not support Range requests for {filename}"
                    ))
                }
            }
            Err(e) => FetchError::Http(format!("failed to probe {filename}: {e}")),
        }
    }

    /// Issues one validated range request against the `/resolve` URL,
    /// following the redirect to a freshly-signed CDN URL.
    ///
    /// Returns the body plus the response `ETag` (cleaned), which the
    /// caller records for cross-request consistency.
    fn fetch_once(
        &self,
        start: u64,
        end_inclusive: u64,
    ) -> Result<(Vec<u8>, Option<String>), FetchError> {
        let expected_len = end_inclusive
            .checked_sub(start)
            .and_then(|d| d.checked_add(1))
            .ok_or_else(|| {
                FetchError::Http(format!(
                    "invalid range {start}..={end_inclusive} for {}",
                    self.filename
                ))
            })?;
        let expected_usize = usize::try_from(expected_len).map_err(|_| {
            FetchError::Http(format!(
                "range length {expected_len} exceeds addressable memory for {}",
                self.filename
            ))
        })?;

        let range_value = format!("bytes={start}-{end_inclusive}");
        // BORROW: explicit .as_str() instead of Deref coercion
        let filename = self.filename.as_str();
        let total_size = self.total_size;

        self.handle.block_on(async {
            let resp = self
                .client
                .get(self.hf_url.as_str())
                // BORROW: explicit .as_str() instead of Deref coercion
                .header(reqwest::header::RANGE, range_value.as_str())
                .timeout(RANGE_REQUEST_TIMEOUT)
                .send()
                .await
                .map_err(|e| {
                    FetchError::Http(format!("failed to send Range request for {filename}: {e}"))
                })?;

            let status = resp.status();
            if status == reqwest::StatusCode::OK {
                // Body deliberately not read: a 200 means the server ignored
                // the Range header and is offering the FULL file.
                return Err(FetchError::Http(format!(
                    "server ignored the Range header for {filename} (status 200 \
                     for bytes={start}-{end_inclusive}); refusing to read the full file"
                )));
            }
            if status != reqwest::StatusCode::PARTIAL_CONTENT {
                return Err(FetchError::Http(format!(
                    "Range request for {filename} returned status {status}"
                )));
            }

            let content_range = resp
                .headers()
                .get(reqwest::header::CONTENT_RANGE)
                .and_then(|v| v.to_str().ok())
                // BORROW: explicit .to_owned() — header value outlives the response borrow
                .map(str::to_owned)
                .ok_or_else(|| {
                    FetchError::Http(format!("missing Content-Range header for {filename}"))
                })?;
            let (cr_start, cr_end, cr_total) = parse_content_range(content_range.as_str())
                .ok_or_else(|| {
                    FetchError::Http(format!(
                        "invalid Content-Range header for {filename}: {content_range}"
                    ))
                })?;
            if cr_start != start || cr_end != end_inclusive || cr_total != total_size {
                return Err(FetchError::Http(format!(
                    "Content-Range mismatch for {filename}: requested \
                     bytes={start}-{end_inclusive} of {total_size}, server answered {content_range}"
                )));
            }

            let etag = resp
                .headers()
                .get(reqwest::header::ETAG)
                .and_then(|v| v.to_str().ok())
                .map(clean_etag);

            // Stream the body with a hard cap of the requested length.
            let mut data: Vec<u8> = Vec::with_capacity(expected_usize);
            let mut resp = resp;
            while let Some(chunk) = resp.chunk().await.map_err(|e| {
                FetchError::Http(format!("failed to read Range response for {filename}: {e}"))
            })? {
                if data.len().saturating_add(chunk.len()) > expected_usize {
                    return Err(FetchError::Http(format!(
                        "server sent more than the requested {expected_len} bytes \
                         for {filename} (bytes={start}-{end_inclusive}); aborting"
                    )));
                }
                data.extend_from_slice(&chunk);
            }
            if data.len() != expected_usize {
                return Err(FetchError::Http(format!(
                    "server returned {} bytes for a {expected_len}-byte range \
                     of {filename} (bytes={start}-{end_inclusive})",
                    data.len()
                )));
            }

            Ok((data, etag))
        })
    }

    /// Records / checks the response `ETag` for cross-request consistency.
    fn check_response_etag(&mut self, etag: Option<String>) -> Result<(), FetchError> {
        if let Some(current) = etag {
            match &self.response_etag {
                Some(previous) if *previous != current => {
                    return Err(FetchError::Http(format!(
                        "{} changed upstream during inspect (etag {previous} \
                         became {current})",
                        self.filename
                    )));
                }
                Some(_) => {} // EXPLICIT: etag unchanged — nothing to record
                None => self.response_etag = Some(current),
            }
        }
        Ok(())
    }
}

impl RangeFetcher for HttpRangeFetcher {
    fn fetch(&mut self, start: u64, end_inclusive: u64) -> Result<Vec<u8>, FetchError> {
        // Each fetch re-resolves through /resolve (fresh signed CDN URL),
        // so there is no stored-signature expiry to manage; failures
        // surface directly with their real HTTP status.
        let (data, etag) = self.fetch_once(start, end_inclusive)?;
        self.check_response_etag(etag)?;
        Ok(data)
    }

    fn total_size(&self) -> u64 {
        self.total_size
    }

    fn extra_requests(&self) -> u32 {
        self.extra
    }
}

/// Parses a `Content-Range: bytes S-E/T` value into `(S, E, T)`.
///
/// Returns `None` on any deviation from that exact form (including the
/// `bytes */T` unsatisfied-range form, which is never valid for a `206`).
fn parse_content_range(value: &str) -> Option<(u64, u64, u64)> {
    let rest = value.strip_prefix("bytes ")?;
    let (range, total) = rest.split_once('/')?;
    let (start, end) = range.split_once('-')?;
    Some((
        start.trim().parse().ok()?,
        end.trim().parse().ok()?,
        total.trim().parse().ok()?,
    ))
}

/// Normalises an `ETag` value: strips the weak-validator prefix and quotes.
///
/// Matches the probe's normalisation (`etag.replace('"', "")`), so probe
/// and response etags compare in the same representation.
fn clean_etag(raw: &str) -> String {
    raw.strip_prefix("W/").unwrap_or(raw).replace('"', "")
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

    use super::*;

    /// In-memory fetcher over a byte vector, logging every fetched range.
    struct InMemoryFetcher {
        data: Vec<u8>,
        calls: Vec<(u64, u64)>,
    }

    impl InMemoryFetcher {
        fn new(data: Vec<u8>) -> Self {
            Self {
                data,
                calls: Vec::new(),
            }
        }
    }

    impl RangeFetcher for InMemoryFetcher {
        fn fetch(&mut self, start: u64, end_inclusive: u64) -> Result<Vec<u8>, FetchError> {
            self.calls.push((start, end_inclusive));
            let s = usize::try_from(start).unwrap();
            let e = usize::try_from(end_inclusive).unwrap();
            self.data
                .get(s..=e)
                .map(<[u8]>::to_vec)
                .ok_or_else(|| FetchError::Http(format!("bad range {start}..={end_inclusive}")))
        }

        fn total_size(&self) -> u64 {
            u64::try_from(self.data.len()).unwrap()
        }
    }

    /// A fetcher that always fails with an HTTP-status-shaped error.
    struct FailingFetcher {
        size: u64,
    }

    impl RangeFetcher for FailingFetcher {
        fn fetch(&mut self, _start: u64, _end_inclusive: u64) -> Result<Vec<u8>, FetchError> {
            Err(FetchError::Http(
                "Range request for x.npz returned status 403 Forbidden".to_owned(),
            ))
        }

        fn total_size(&self) -> u64 {
            self.size
        }
    }

    /// A fetcher that returns fewer bytes than requested.
    struct ShortFetcher {
        size: u64,
    }

    impl RangeFetcher for ShortFetcher {
        fn fetch(&mut self, _start: u64, _end_inclusive: u64) -> Result<Vec<u8>, FetchError> {
            Ok(vec![0u8; 1])
        }

        fn total_size(&self) -> u64 {
            self.size
        }
    }

    fn sample_data(len: usize) -> Vec<u8> {
        // CAST: usize → u8, deliberate wrapping for a recognisable pattern
        #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
        (0..len).map(|i| (i % 251) as u8).collect()
    }

    // ---------- Seek semantics ----------

    #[test]
    fn seek_start_end_current_semantics() {
        let mut r = RangeReader::new(InMemoryFetcher::new(sample_data(1000)));
        assert_eq!(r.seek(SeekFrom::Start(10)).unwrap(), 10);
        assert_eq!(r.seek(SeekFrom::Current(5)).unwrap(), 15);
        assert_eq!(r.seek(SeekFrom::Current(-15)).unwrap(), 0);
        assert_eq!(r.seek(SeekFrom::End(0)).unwrap(), 1000);
        assert_eq!(r.seek(SeekFrom::End(-1000)).unwrap(), 0);
        // Past-EOF seek is permitted; reads there return 0.
        assert_eq!(r.seek(SeekFrom::End(50)).unwrap(), 1050);
        let mut buf = [0u8; 4];
        assert_eq!(r.read(&mut buf).unwrap(), 0);
    }

    #[test]
    fn seek_negative_is_invalid_input() {
        let mut r = RangeReader::new(InMemoryFetcher::new(sample_data(100)));
        let err = r.seek(SeekFrom::Current(-1)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        let err = r.seek(SeekFrom::End(-101)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn empty_file_reads_zero_with_no_requests() {
        let mut r = RangeReader::new(InMemoryFetcher::new(Vec::new()));
        let mut buf = [0u8; 8];
        assert_eq!(r.read(&mut buf).unwrap(), 0);
        assert_eq!(r.stats().requests, 0);
    }

    // ---------- Read-ahead and caching ----------

    #[test]
    fn sequential_small_reads_coalesce_into_one_window_fetch() {
        // 100 KiB file, reads far from the tail region.
        let mut r = RangeReader::new(InMemoryFetcher::new(sample_data(100 * 1024)));
        let mut buf = [0u8; 16];
        for i in 0..10 {
            r.read_exact(&mut buf).unwrap();
            assert_eq!(buf[0], sample_data(100 * 1024)[i * 16]);
        }
        // 160 bytes of sequential reads << 4 KiB read-ahead → one request.
        assert_eq!(r.stats().requests, 1);
        assert_eq!(r.fetcher.calls[0], (0, READAHEAD_BYTES - 1));
    }

    #[test]
    fn tail_region_read_prefetches_tail_once() {
        let size: u64 = 1024 * 1024; // 1 MiB
        let mut r = RangeReader::new(InMemoryFetcher::new(sample_data(
            usize::try_from(size).unwrap(),
        )));
        // Mimic the ZIP EOCD scan: seek near the end, read a few bytes.
        r.seek(SeekFrom::End(-22)).unwrap();
        let mut buf = [0u8; 22];
        r.read_exact(&mut buf).unwrap();
        assert_eq!(r.stats().requests, 1);
        assert_eq!(r.fetcher.calls[0], (size - TAIL_PREFETCH_BYTES, size - 1));
        // Every further read inside the tail is served from cache.
        r.seek(SeekFrom::End(-4096)).unwrap();
        let mut big = [0u8; 4096];
        r.read_exact(&mut big).unwrap();
        assert_eq!(r.stats().requests, 1);
        assert_eq!(r.stats().bytes_fetched, TAIL_PREFETCH_BYTES);
    }

    #[test]
    fn window_reuse_after_seek_back() {
        let mut r = RangeReader::new(InMemoryFetcher::new(sample_data(64 * 1024)));
        let mut buf = [0u8; 128];
        r.read_exact(&mut buf).unwrap();
        assert_eq!(r.stats().requests, 1);
        // Seek back inside the fetched window: no new request.
        r.seek(SeekFrom::Start(32)).unwrap();
        r.read_exact(&mut buf).unwrap();
        assert_eq!(r.stats().requests, 1);
        let expected = sample_data(64 * 1024);
        assert_eq!(&buf[..], &expected[32..160]);
    }

    #[test]
    fn window_fetch_never_overlaps_cached_tail() {
        let size: u64 = 200 * 1024;
        let mut r = RangeReader::new(InMemoryFetcher::new(sample_data(
            usize::try_from(size).unwrap(),
        )));
        // Prime the tail cache.
        r.seek(SeekFrom::End(-10)).unwrap();
        let mut small = [0u8; 10];
        r.read_exact(&mut small).unwrap();
        // Large read starting just before the tail: the window fetch must
        // stop at the tail boundary, and the rest is served from the tail.
        let tail_start = size - TAIL_PREFETCH_BYTES;
        r.seek(SeekFrom::Start(tail_start - 100)).unwrap();
        let mut big = vec![0u8; 4096];
        r.read_exact(&mut big).unwrap();
        assert_eq!(r.fetcher.calls[1], (tail_start - 100, tail_start - 1));
        let expected = sample_data(usize::try_from(size).unwrap());
        let from = usize::try_from(tail_start - 100).unwrap();
        assert_eq!(&big[..], &expected[from..from + 4096]);
    }

    // ---------- Safety budgets ----------

    #[test]
    fn request_cap_is_enforced() {
        let data = sample_data(10 * 1024 * 1024);
        let mut r = RangeReader::with_limits(InMemoryFetcher::new(data), 3, u64::MAX);
        let mut buf = [0u8; 8];
        // Three widely-spaced reads consume the three allowed requests
        // (spacing > READAHEAD_BYTES so no window reuse).
        for i in 0u64..3 {
            r.seek(SeekFrom::Start(i * 100 * 1024)).unwrap();
            r.read_exact(&mut buf).unwrap();
        }
        r.seek(SeekFrom::Start(1024 * 1024)).unwrap();
        let err = r.read(&mut buf).unwrap_err();
        assert!(
            err.to_string()
                .contains("range request cap exceeded (3 requests)"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn transfer_budget_is_enforced() {
        let data = sample_data(10 * 1024 * 1024);
        // Budget of 6 KiB: one 4 KiB window fits, the second does not.
        let mut r = RangeReader::with_limits(InMemoryFetcher::new(data), u32::MAX, 6 * 1024);
        let mut buf = [0u8; 8];
        r.read_exact(&mut buf).unwrap();
        r.seek(SeekFrom::Start(1024 * 1024)).unwrap();
        let err = r.read(&mut buf).unwrap_err();
        assert!(
            err.to_string().contains("range transfer budget exceeded"),
            "unexpected error: {err}"
        );
    }

    // ---------- Error channel ----------

    #[test]
    fn fetch_error_surfaces_as_io_and_is_recoverable_typed() {
        let mut r = RangeReader::new(FailingFetcher { size: 1024 });
        let mut buf = [0u8; 8];
        let err = r.read(&mut buf).unwrap_err();
        assert!(err.to_string().contains("returned status 403"));
        let typed = r.take_last_error().expect("typed error must be stored");
        assert!(matches!(typed, FetchError::Http(msg)
            if msg.contains("returned status 403 Forbidden")));
        // Taken once — the slot is now empty.
        assert!(r.take_last_error().is_none());
    }

    #[test]
    fn short_fetch_is_a_contract_error() {
        let mut r = RangeReader::new(ShortFetcher { size: 1024 * 1024 });
        let mut buf = [0u8; 8];
        let err = r.read(&mut buf).unwrap_err();
        assert!(
            err.to_string().contains("bytes for a"),
            "unexpected error: {err}"
        );
    }

    // ---------- Pure helpers ----------

    #[test]
    fn parse_content_range_accepts_the_exact_206_form() {
        assert_eq!(parse_content_range("bytes 0-7/1234"), Some((0, 7, 1234)));
        assert_eq!(
            parse_content_range("bytes 100-199/200"),
            Some((100, 199, 200))
        );
    }

    #[test]
    fn parse_content_range_rejects_deviant_forms() {
        assert_eq!(parse_content_range("bytes */1234"), None);
        assert_eq!(parse_content_range("bytes 0-7/*"), None);
        assert_eq!(parse_content_range("0-7/1234"), None);
        assert_eq!(parse_content_range("bytes 7/1234"), None);
        assert_eq!(parse_content_range(""), None);
    }

    #[test]
    fn clean_etag_strips_quotes_and_weak_prefix() {
        assert_eq!(clean_etag("\"abc123\""), "abc123");
        assert_eq!(clean_etag("W/\"abc123\""), "abc123");
        assert_eq!(clean_etag("abc123"), "abc123");
    }

    // ---------- NPZ end-to-end over the reader ----------

    /// Builds a minimal `NPY` v1.0 payload: magic, version, padded header
    /// dict, and zeroed data.
    fn npy_bytes(descr: &str, shape_literal: &str, data_len: usize) -> Vec<u8> {
        let dict =
            format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_literal}, }}");
        // Pad so (magic 6 + version 2 + len 2 + header) % 64 == 0, per spec.
        let unpadded = 10 + dict.len() + 1; // +1 for the trailing '\n'
        let padding = (64 - unpadded % 64) % 64;
        let header_len = dict.len() + padding + 1;
        let mut out = Vec::with_capacity(10 + header_len + data_len);
        out.extend_from_slice(b"\x93NUMPY\x01\x00");
        out.extend_from_slice(&u16::try_from(header_len).unwrap().to_le_bytes());
        out.extend_from_slice(dict.as_bytes());
        out.extend(std::iter::repeat_n(b' ', padding));
        out.push(b'\n');
        out.extend(std::iter::repeat_n(0u8, data_len));
        out
    }

    /// Builds a stored (uncompressed) ZIP archive from `(name, payload)`
    /// entries: local headers, central directory, EOCD. CRC fields are
    /// zero — the inspect path never reads entry data, so they are unused.
    fn stored_zip(entries: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut offsets = Vec::new();
        for (name, payload) in entries {
            offsets.push(u32::try_from(out.len()).unwrap());
            let size = u32::try_from(payload.len()).unwrap();
            out.extend_from_slice(&0x0403_4b50u32.to_le_bytes()); // LFH sig
            out.extend_from_slice(&20u16.to_le_bytes()); // version needed
            out.extend_from_slice(&0u16.to_le_bytes()); // flags
            out.extend_from_slice(&0u16.to_le_bytes()); // method: stored
            out.extend_from_slice(&0u32.to_le_bytes()); // mod time + date
            out.extend_from_slice(&0u32.to_le_bytes()); // CRC-32 (unused)
            out.extend_from_slice(&size.to_le_bytes()); // compressed size
            out.extend_from_slice(&size.to_le_bytes()); // uncompressed size
            out.extend_from_slice(&u16::try_from(name.len()).unwrap().to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes()); // extra len
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(payload);
        }
        let cd_offset = u32::try_from(out.len()).unwrap();
        for ((name, payload), lfh_offset) in entries.iter().zip(&offsets) {
            let size = u32::try_from(payload.len()).unwrap();
            out.extend_from_slice(&0x0201_4b50u32.to_le_bytes()); // CDFH sig
            out.extend_from_slice(&20u16.to_le_bytes()); // version made by
            out.extend_from_slice(&20u16.to_le_bytes()); // version needed
            out.extend_from_slice(&0u16.to_le_bytes()); // flags
            out.extend_from_slice(&0u16.to_le_bytes()); // method: stored
            out.extend_from_slice(&0u32.to_le_bytes()); // mod time + date
            out.extend_from_slice(&0u32.to_le_bytes()); // CRC-32 (unused)
            out.extend_from_slice(&size.to_le_bytes()); // compressed size
            out.extend_from_slice(&size.to_le_bytes()); // uncompressed size
            out.extend_from_slice(&u16::try_from(name.len()).unwrap().to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes()); // extra len
            out.extend_from_slice(&0u16.to_le_bytes()); // comment len
            out.extend_from_slice(&0u16.to_le_bytes()); // disk number
            out.extend_from_slice(&0u16.to_le_bytes()); // internal attrs
            out.extend_from_slice(&0u32.to_le_bytes()); // external attrs
            out.extend_from_slice(&lfh_offset.to_le_bytes());
            out.extend_from_slice(name.as_bytes());
        }
        let cd_size = u32::try_from(out.len()).unwrap() - cd_offset;
        let n = u16::try_from(entries.len()).unwrap();
        out.extend_from_slice(&0x0605_4b50u32.to_le_bytes()); // EOCD sig
        out.extend_from_slice(&0u16.to_le_bytes()); // this disk
        out.extend_from_slice(&0u16.to_le_bytes()); // CD start disk
        out.extend_from_slice(&n.to_le_bytes()); // entries this disk
        out.extend_from_slice(&n.to_le_bytes()); // entries total
        out.extend_from_slice(&cd_size.to_le_bytes());
        out.extend_from_slice(&cd_offset.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // comment len
        out
    }

    #[test]
    fn npz_inspect_over_range_reader_reads_metadata_not_data() {
        // Two arrays; the big one carries 600 KiB of data the inspect
        // must never fetch.
        let npz = stored_zip(&[
            ("w_enc.npy", npy_bytes("<f4", "(2, 3)", 2 * 3 * 4)),
            ("b_dec.npy", npy_bytes("<f4", "(150, 1024)", 600 * 1024)),
        ]);
        let total = u64::try_from(npz.len()).unwrap();
        let mut reader = RangeReader::new(InMemoryFetcher::new(npz));

        let info = anamnesis::inspect_npz_from_reader(&mut reader)
            .expect("synthetic NPZ must inspect cleanly");

        assert_eq!(info.tensors.len(), 2);
        let names: Vec<&str> = info.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"w_enc"), "names: {names:?}");
        assert!(names.contains(&"b_dec"), "names: {names:?}");
        let b_dec = info.tensors.iter().find(|t| t.name == "b_dec").unwrap();
        assert_eq!(b_dec.shape, vec![150, 1024]);

        // The efficiency property this module exists for: metadata-only
        // transfer, a small number of requests, no weight data fetched.
        let stats = reader.stats();
        assert!(
            stats.requests <= 8,
            "expected a handful of range requests, got {}",
            stats.requests
        );
        assert!(
            stats.bytes_fetched < total / 4,
            "fetched {} of {total} bytes — inspect must not read tensor data",
            stats.bytes_fetched
        );
    }
}
