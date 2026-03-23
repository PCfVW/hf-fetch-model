# Download Performance Roadmap

**Date:** March 23, 2026
**Status:** Proposed
**Context:** Benchmark of hf-fm vs Python `hf download` on 3 models (1/5/13.5 GiB) showed identical throughput (~40 MiB/s) on a ~300 Mbps consumer connection. Both tools saturate the available bandwidth. hf-fm's multi-connection advantage cannot materialize when the pipe is the bottleneck. This roadmap identifies improvements validated by comparing the Python huggingface_hub source code (v1.3.5, httpx-based) with hf-fm's Rust implementation.

**Benchmark conditions (2026-03-23):** WiFi, ~300 Mbps, Windows 11. WiFi signal varied between runs (noted for the large model result). Results show parity, not advantage.

---

## Validated Improvements

### 1. Reduce per-file probe overhead

**Current:** Each large file (above chunk threshold) makes 2 HTTP requests before downloading: one to HF endpoint (no-redirect, extracts etag/commit), one to CDN (extracts Content-Range/file size). For a repo with 10 large files at concurrency 4, this adds 20 round-trips of latency.

**Python comparison:** Python makes 1 HEAD request for metadata, then 1 GET per file. No probe step.

**Improvement:** Batch-probe all large files upfront before acquiring semaphore permits, or combine probing with the metadata fetch that already happens via `list_repo_files_with_metadata()`. Cache `RangeInfo` results so files don't re-probe.

**Impact:** ~500ms–2s faster startup, especially on high-latency connections.

**Location:** `src/chunked.rs` `probe_range_support()` (lines 121-208), `src/download.rs` `dispatch_download()` (lines 383-396).

### 2. Tune reqwest client configuration

**Current:** `build_client()` and `build_no_redirect_client()` use bare `Client::builder()` defaults. No explicit pool size, no keepalive tuning.

**Python comparison:** Python also uses library defaults (httpx, ~100 connections/host). Neither side explicitly tunes TCP settings.

**Improvement:** Add explicit configuration to reqwest builder:
```rust
Client::builder()
    .pool_max_idle_per_host(16)     // sized for connections_per_file (default 8)
    .pool_idle_timeout(Duration::from_secs(30))
    .tcp_nodelay(true)              // disable Nagle's algorithm
    .default_headers(headers)
    .build()
```

**Impact:** Marginal but measurable for many concurrent Range requests. `tcp_nodelay` helps small request/response exchanges (probes, chunk headers).

**Location:** `src/chunked.rs` `build_client()` (lines 90-107), `build_no_redirect_client()` (lines 66-83).

### 3. Add streaming timeout per chunk

**Current:** `download_chunk()` streams via `response.bytes_stream()` with no per-chunk timeout. If the stream stalls, the task hangs until TCP timeout (~10-20 minutes).

**Python comparison:** Python has a 10-second read timeout plus 5 retries with resume from last byte position.

**Improvement:** Wrap the streaming loop in `tokio::time::timeout()`:
```rust
tokio::time::timeout(Duration::from_secs(60), async {
    while let Some(chunk) = stream.next().await { ... }
}).await
```
On timeout, retry the chunk from the last written offset (resume within chunk).

**Impact:** Faster recovery from stalled connections. Currently a stalled chunk blocks the entire file indefinitely.

**Location:** `src/chunked.rs` `download_chunk()` (lines 446-548).

### 4. Rate-limit awareness (429 + Ratelimit headers)

**Current:** hf-fm retries on HTTP errors with exponential backoff (300ms–10s) but does not parse rate-limit headers.

**Python comparison:** Python parses IETF `Ratelimit` headers (`Ratelimit: "api";r=0;t=55`) and waits the exact reset time + 1s on 429 responses. This avoids wasting retries during rate-limit windows.

**Improvement:** In the retry loop, detect 429 responses, parse `Ratelimit` or `Retry-After` headers, and wait the server-specified duration instead of generic backoff.

**Impact:** Faster recovery from rate limiting (wait exactly the right time instead of potentially too short or too long). More relevant for batch/CI workflows that hit rate limits.

**Location:** `src/retry.rs` (lines 60-107), `src/chunked.rs` retry loops.

### ~~5. Disk space check before download~~ — DONE (v0.9.0)

Implemented in `src/download.rs` `check_disk_space()`. Shows current cache size, projected size after download, and available disk space before every download. Warns if space is tight (<10% margin) or insufficient. Uses the `fs2` crate for cross-platform available space queries.

### 6. Resume within chunk

**Current:** If a chunk partially downloads then fails, the entire chunk is re-downloaded from the start.

**Python comparison:** Python tracks `resume_size` and sends `Range: bytes={resume_size}-` on retry, appending to the partial file.

**Improvement:** Track bytes written per chunk. On retry, seek to last written position and send `Range: bytes={written_offset}-{end}` instead of `Range: bytes={start}-{end}`.

**Impact:** Faster recovery on large chunks over unstable connections. A 128 MiB chunk that fails at 120 MiB currently wastes 120 MiB of re-download.

**Location:** `src/chunked.rs` `download_chunk()` retry loop (lines 487-548).

---

### 7. Speed probe and download ETA

**Current:** No way to estimate download time before committing to a download. Users downloading a 13 GiB model on an unknown connection have no idea if it will take 30 seconds or 30 minutes.

**Improvement:** Add `hf-fm speedtest` command (or `--probe-speed` flag on download):
1. Download a small Range chunk (e.g., 10 MiB) from a known public file on the HF CDN
2. Measure sustained throughput
3. Report speed and, if a `REPO_ID` is given, estimate download time:

```
$ hf-fm speedtest google/gemma-2-2b-it --preset safetensors
  HuggingFace CDN: 42.3 MiB/s (338 Mbps)
  google/gemma-2-2b-it (safetensors preset): 4.89 GiB
  Estimated download time: ~1 min 58s
```

Could also integrate with `--dry-run` to show ETA alongside the file listing.

**Impact:** Better UX for users on unfamiliar networks. Helps decide "download now or wait for office WiFi."

---

## Not Validated (investigated but no clear improvement)

| Idea | Finding |
|------|---------|
| HTTP/2 multiplexing | Neither Python nor Rust explicitly enables HTTP/2. HF CDN (Cloudflare) negotiates it automatically. Explicit `http2_prior_knowledge()` would skip negotiation but risk breaking non-HTTP/2 proxies. |
| Higher default concurrency | Python uses 8 workers (file-level). hf-fm auto-tunes to 4-8 based on file distribution. Current auto-tuning is already adaptive. |
| Compression (Accept-Encoding) | Python explicitly disables compression (`Accept-Encoding: identity`) because it needs accurate Content-Length for cache validation. hf-fm should follow the same pattern. |
| Xet storage | Python supports HF's Xet chunk-based storage for >50GB files. This is a proprietary protocol requiring `hf_xet` package. Out of scope for hf-fm unless HF makes it a standard. |

---

## Benchmark Data (2026-03-23)

| Model | Size | `hf download` (Python) | `hf-fm` (Rust) | Ratio | Notes |
|-------|------|----------------------|----------------|-------|-------|
| `casperhansen/llama-3.2-1b-instruct-awq` | 1.0 GiB | 30.2s | 29.4s | 1.0x | Single safetensors file |
| `google/gemma-2-2b-it` | 4.9 GiB | 121.4s | 121.8s | 1.0x | 2 shards |
| `mistralai/Mistral-7B-v0.1` | 13.5 GiB | 263.4s | 347.2s | 0.76x | WiFi signal weaker during hf-fm run |

**Conclusion:** On a ~300 Mbps consumer connection, both tools saturate the pipe. Multi-connection advantage would show on faster links (1+ Gbps) where a single TCP stream can't fill the bandwidth. The improvements in this roadmap focus on reducing overhead, improving resilience, and better UX — not raw throughput on bandwidth-limited connections.
