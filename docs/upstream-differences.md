# Upstream differences

Where hf-fetch-model diverges from Python's `huggingface_hub` and `hf_transfer`, and why. When upstream catches up on a point, the entry is removed.

**Last reviewed:** April 2026, against `huggingface_hub` 0.34.x and `hf_transfer` 0.1.9.

---

## CDN signed URL expiry detection

Python's `huggingface_hub` and `hf_transfer` do not parse `X-Amz-Expires` from CDN signed URLs. If a chunked download outlasts the URL's validity window (typically 1 hour), chunk requests silently fail with HTTP 403. Retries use the same expired URL and fail deterministically.

hf-fetch-model parses the expiry from the URL, estimates download time at a conservative 5 MB/s per connection, and re-probes for a fresh URL before starting if the margin is insufficient. Non-AWS CDNs (GCS, Cloudflare) return `None` and the re-probe path is silently skipped.

**Files:** `chunked.rs` (`parse_cdn_expiry`), `download.rs` (`download_single_file_chunked`).

---

## Windows blob deduplication

The HF cache uses symlinks from `snapshots/<hash>/<filename>` to `blobs/<etag>`. On Windows without `SeCreateSymbolicLinkPrivilege` (the default for non-admin users), both `hf-hub` 0.5 and Python's `huggingface_hub` fall back to `rename`, which moves the blob out of `blobs/` and destroys the deduplication property. A second revision of the same model triggers a full re-download.

hf-fetch-model falls back to `copy`, preserving the blob in `blobs/<etag>` for cross-revision deduplication. The trade-off is doubled disk usage for the pointer file, but the blob is never lost.

**File:** `chunked.rs` (`symlink_or_copy`).

---

## POSIX symlink race tolerance

When two concurrent downloaders race to create the same pointer symlink (same etag, different revision downloads), `symlink()` returns `EEXIST`. Both `hf-hub` and `huggingface_hub` propagate this as a hard error. hf-fetch-model catches `EEXIST` and silently succeeds, since the file is correctly cached.

**File:** `chunked.rs` (`symlink_or_copy`).

---

## Temp file cleanup on task abort

When a download batch times out, `JoinSet::abort_all()` cancels in-flight tasks. Each chunked download pre-allocates a `.chunked.part` file at the full file size. Neither `hf-hub` nor `hf_transfer` cleans up these files on abort — they persist until manual removal.

hf-fetch-model wraps the temp file in an RAII `TempFileGuard` that removes it on drop, including on task abort. The guard uses synchronous `std::fs::remove_file` in `Drop`, which is safe for a single syscall on the aborting thread.

**File:** `chunked.rs` (`TempFileGuard`).

---

## TCP connect timeout

`hf-hub`'s `reqwest::Client` and Python's `requests.Session` set no explicit TCP connect timeout. A dead peer (SYN sent, no SYN-ACK) stalls indefinitely until the OS-level TCP timeout fires (typically 2+ minutes on Linux, longer on Windows).

hf-fetch-model sets a 30-second connect timeout on both HTTP clients (`build_client`, `build_no_redirect_client`), bounding the handshake for all download, probe, inspect, and info operations.

**File:** `chunked.rs` (`CONNECT_TIMEOUT`).

---

## Disk space check without cache scan

Python's `huggingface_hub` checks available disk space via `shutil.disk_usage()` without scanning the cache. hf-fetch-model previously scanned the entire cache (`cache_summary()`) on the Tokio thread before every download to display a "cache X GiB -> Y GiB" line. This is now removed — the display shows download size, available space, and projected remaining space, matching the Python approach and eliminating a multi-second blocking stall on large caches.

**File:** `download.rs` (`check_disk_space`).

---

## Shared HTTP client for metadata requests

Python's `huggingface_hub` uses `requests.Session` which maintains a connection pool. `hf-hub` 0.5's internal API creates clients per-request. hf-fetch-model's `list_repo_files_with_metadata` now accepts a shared `&reqwest::Client`, reusing TCP connections and TLS sessions across metadata, download, and inspect operations.

**File:** `repo.rs` (`list_repo_files_with_metadata`), `chunked.rs` (`build_client`).
