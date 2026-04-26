// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-chunk progress sidecar for resumeable chunked downloads.
//!
//! When a chunked download is interrupted — timeout-induced future drop,
//! Ctrl-C, panic, or a retryable chunk error — the `.chunked.part` temp
//! file is preserved by [`crate::chunked::TempFileGuard`] and accompanied
//! by a `.chunked.part.state` JSON sidecar describing how many bytes each
//! chunk completed.
//!
//! On the next invocation, [`crate::chunked`] reuses the partial when the
//! sidecar's `(schema_version, etag, total_size, connections)` quadruple
//! matches the current [`crate::chunked::RangeInfo`]; otherwise it
//! discards both files and starts fresh.
//!
//! The sidecar is updated atomically (write-tmp + rename) every
//! `CHECKPOINT_BYTES` of progress per chunk. On successful finalize the
//! sidecar is removed by [`crate::chunked`].

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::FetchError;

/// Schema version embedded in every sidecar.
///
/// Bumped whenever the on-disk JSON shape changes incompatibly. Older
/// versions of `hf-fetch-model` reading a newer sidecar discard it via
/// [`ChunkedState::is_compatible_with`] and start the download fresh.
pub(crate) const SCHEMA_VERSION: u32 = 1;

/// Per-chunk progress entry inside [`ChunkedState`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ChunkProgress {
    /// Chunk index in `[0, connections)`. Stable across invocations.
    pub idx: usize,
    /// First byte (inclusive) this chunk owns in the file.
    pub start: u64,
    /// Last byte (inclusive) this chunk owns in the file.
    pub end: u64,
    /// Bytes successfully written to disk for this chunk so far,
    /// counted from `start`. Resume offset = `start + completed`;
    /// completion length = `end + 1 - start`.
    pub completed: u64,
}

impl ChunkProgress {
    /// Returns `true` when this chunk has written all bytes in `[start, end]`.
    #[must_use]
    pub(crate) const fn is_complete(&self) -> bool {
        // `end - start + 1` is the chunk's full length (inclusive range).
        self.completed >= self.end.saturating_sub(self.start).saturating_add(1)
    }
}

/// On-disk state of a chunked download in progress, serialized as JSON
/// at `{etag}.chunked.part.state`.
///
/// Each field is also a resume invariant: a mismatch between any of
/// `(schema_version, etag, total_size, connections)` and the current
/// download configuration discards the sidecar (see
/// [`Self::is_compatible_with`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ChunkedState {
    /// Sidecar schema version. Compared against [`SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Remote etag the partial file was downloaded against. A different
    /// etag means the upstream file changed — the bytes on disk are no
    /// longer valid for the current download.
    pub etag: String,
    /// Full file size in bytes. Must match the current `Content-Length`.
    pub total_size: u64,
    /// Number of parallel connections used when the partial was started.
    /// Connections divide the byte range; resuming with a different count
    /// would mis-align the chunk boundaries.
    pub connections: usize,
    /// Per-chunk completion offsets, one entry per `connections` slot.
    pub chunks: Vec<ChunkProgress>,
}

impl ChunkedState {
    /// Builds a fresh state for a brand-new download.
    ///
    /// `chunks` is a slice of `(idx, start, end)` boundaries; the
    /// `completed` field of each entry starts at `0`.
    #[must_use]
    pub(crate) fn new_fresh(
        etag: String,
        total_size: u64,
        connections: usize,
        chunks: &[(usize, u64, u64)],
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            etag,
            total_size,
            connections,
            chunks: chunks
                .iter()
                .map(|&(idx, start, end)| ChunkProgress {
                    idx,
                    start,
                    end,
                    completed: 0,
                })
                .collect(),
        }
    }

    /// Returns `true` when the on-disk sidecar can be resumed for the
    /// current download — i.e. when every resume invariant matches.
    ///
    /// The four invariants are checked in order: schema version (newer
    /// versions discard older), etag (upstream file unchanged), total
    /// size (sanity), and chunk count (boundary alignment).
    #[must_use]
    pub(crate) fn is_compatible_with(
        &self,
        etag: &str,
        total_size: u64,
        connections: usize,
    ) -> bool {
        self.schema_version == SCHEMA_VERSION
            && self.etag == etag
            && self.total_size == total_size
            && self.connections == connections
            && self.chunks.len() == connections
    }

    /// Reads and parses the sidecar at `path`.
    ///
    /// # Returns
    ///
    /// * `Ok(Some(state))` when the file exists and parses successfully.
    /// * `Ok(None)` when the file is absent OR present-but-unparseable
    ///   (treated as "no usable resume state — start fresh"; the caller
    ///   should overwrite the corrupt file with a fresh state).
    /// * `Err(...)` only on hard I/O errors (permission denied,
    ///   disk read failure).
    ///
    /// # Errors
    ///
    /// Returns [`FetchError::Io`](crate::FetchError::Io) on read failures
    /// other than file-not-found.
    pub(crate) async fn load(path: &Path) -> Result<Option<Self>, FetchError> {
        match tokio::fs::read_to_string(path).await {
            Ok(text) => Ok(serde_json::from_str(text.as_str()).ok()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(FetchError::Io {
                path: path.to_path_buf(),
                source: e,
            }),
        }
    }

    /// Writes this state to `path` atomically (write-tmp + rename).
    ///
    /// The rename is atomic on POSIX and effectively atomic on Windows
    /// (`MoveFileEx`), so a process crash mid-save leaves either the
    /// previous valid sidecar or the new one — never a half-written
    /// blend that would fail to parse on a future load.
    ///
    /// # Errors
    ///
    /// Returns [`FetchError::ChunkedDownload`](crate::FetchError::ChunkedDownload)
    /// if JSON serialization fails (would indicate a programmer bug —
    /// every field of [`ChunkedState`] is plain-old-data).
    /// Returns [`FetchError::Io`](crate::FetchError::Io) on filesystem
    /// errors during the temp write or rename.
    pub(crate) async fn save_atomic(&self, path: &Path) -> Result<(), FetchError> {
        let json = serde_json::to_string(self).map_err(|e| FetchError::ChunkedDownload {
            // BORROW: explicit display() for owned String
            filename: path.display().to_string(),
            reason: format!("failed to serialize chunked-state sidecar: {e}"),
        })?;
        let tmp = path.with_extension("state.tmp");
        tokio::fs::write(&tmp, json.as_bytes())
            .await
            .map_err(|e| FetchError::Io {
                // BORROW: explicit .clone() for owned PathBuf
                path: tmp.clone(),
                source: e,
            })?;
        tokio::fs::rename(&tmp, path)
            .await
            .map_err(|e| FetchError::Io {
                path: path.to_path_buf(),
                source: e,
            })?;
        Ok(())
    }

    /// Removes the sidecar at `path`. Idempotent — a missing file is not
    /// an error. Called from the success path in
    /// [`crate::chunked::download_chunked`] after the blob rename.
    ///
    /// # Errors
    ///
    /// Returns [`FetchError::Io`](crate::FetchError::Io) on hard I/O
    /// errors (permission denied, etc.). File-not-found is silently
    /// ignored.
    pub(crate) async fn remove(path: &Path) -> Result<(), FetchError> {
        match tokio::fs::remove_file(path).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(FetchError::Io {
                path: path.to_path_buf(),
                source: e,
            }),
        }
    }
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

    fn sample_chunks() -> Vec<(usize, u64, u64)> {
        vec![(0, 0, 99), (1, 100, 199), (2, 200, 299)]
    }

    #[test]
    fn new_fresh_initializes_completed_to_zero() {
        let chunks = sample_chunks();
        let state = ChunkedState::new_fresh("etag-abc".to_owned(), 300, 3, &chunks);

        assert_eq!(state.schema_version, SCHEMA_VERSION);
        assert_eq!(state.etag, "etag-abc");
        assert_eq!(state.total_size, 300);
        assert_eq!(state.connections, 3);
        assert_eq!(state.chunks.len(), 3);
        for chunk in &state.chunks {
            assert_eq!(chunk.completed, 0);
        }
    }

    #[test]
    fn json_round_trip_is_lossless() {
        let chunks = sample_chunks();
        let mut state = ChunkedState::new_fresh("etag-xyz".to_owned(), 300, 3, &chunks);
        state.chunks[0].completed = 50;
        state.chunks[1].completed = 100;
        state.chunks[2].completed = 25;

        let json = serde_json::to_string(&state).unwrap();
        let decoded: ChunkedState = serde_json::from_str(json.as_str()).unwrap();

        assert_eq!(state, decoded);
    }

    #[test]
    fn is_compatible_with_matches_on_all_invariants() {
        let chunks = sample_chunks();
        let state = ChunkedState::new_fresh("etag-1".to_owned(), 300, 3, &chunks);

        assert!(state.is_compatible_with("etag-1", 300, 3));
    }

    #[test]
    fn is_compatible_with_rejects_etag_mismatch() {
        let chunks = sample_chunks();
        let state = ChunkedState::new_fresh("etag-1".to_owned(), 300, 3, &chunks);

        assert!(!state.is_compatible_with("etag-2", 300, 3));
    }

    #[test]
    fn is_compatible_with_rejects_total_size_mismatch() {
        let chunks = sample_chunks();
        let state = ChunkedState::new_fresh("etag-1".to_owned(), 300, 3, &chunks);

        assert!(!state.is_compatible_with("etag-1", 600, 3));
    }

    #[test]
    fn is_compatible_with_rejects_connections_mismatch() {
        let chunks = sample_chunks();
        let state = ChunkedState::new_fresh("etag-1".to_owned(), 300, 3, &chunks);

        assert!(!state.is_compatible_with("etag-1", 300, 8));
    }

    #[test]
    fn is_compatible_with_rejects_schema_version_mismatch() {
        let chunks = sample_chunks();
        let mut state = ChunkedState::new_fresh("etag-1".to_owned(), 300, 3, &chunks);
        state.schema_version = SCHEMA_VERSION + 1;

        assert!(!state.is_compatible_with("etag-1", 300, 3));
    }

    #[test]
    fn chunk_is_complete_at_full_length() {
        let chunk = ChunkProgress {
            idx: 0,
            start: 100,
            end: 199,
            completed: 100,
        };
        assert!(chunk.is_complete());
    }

    #[test]
    fn chunk_is_not_complete_one_byte_short() {
        let chunk = ChunkProgress {
            idx: 0,
            start: 100,
            end: 199,
            completed: 99,
        };
        assert!(!chunk.is_complete());
    }

    #[tokio::test]
    async fn load_returns_none_when_file_missing() {
        let dir = std::env::temp_dir().join(format!("hf-fm-state-missing-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("absent.chunked.part.state");

        let loaded = ChunkedState::load(&path).await.unwrap();
        assert!(loaded.is_none());

        std::fs::remove_dir(&dir).ok();
    }

    #[tokio::test]
    async fn load_returns_none_when_file_is_corrupt() {
        let dir = std::env::temp_dir().join(format!("hf-fm-state-corrupt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("corrupt.chunked.part.state");
        std::fs::write(&path, b"this is not JSON {{{").unwrap();

        let loaded = ChunkedState::load(&path).await.unwrap();
        assert!(
            loaded.is_none(),
            "corrupt sidecar should be treated as None"
        );

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[tokio::test]
    async fn save_then_load_recovers_exact_state() {
        let dir =
            std::env::temp_dir().join(format!("hf-fm-state-roundtrip-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rt.chunked.part.state");

        let chunks = sample_chunks();
        let mut original = ChunkedState::new_fresh("etag-rt".to_owned(), 300, 3, &chunks);
        original.chunks[1].completed = 42;

        original.save_atomic(&path).await.unwrap();
        let loaded = ChunkedState::load(&path).await.unwrap().unwrap();

        assert_eq!(original, loaded);

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[tokio::test]
    async fn save_atomic_does_not_leave_tmp_behind() {
        let dir = std::env::temp_dir().join(format!("hf-fm-state-tmp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("clean.chunked.part.state");

        let chunks = sample_chunks();
        let state = ChunkedState::new_fresh("etag-clean".to_owned(), 300, 3, &chunks);
        state.save_atomic(&path).await.unwrap();

        let tmp = path.with_extension("state.tmp");
        assert!(
            !tmp.exists(),
            "tmp file must be renamed away, found {tmp:?}"
        );
        assert!(path.exists());

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[tokio::test]
    async fn remove_is_idempotent_when_file_missing() {
        let dir = std::env::temp_dir().join(format!("hf-fm-state-rm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("never-existed.chunked.part.state");

        // Calling remove() on a missing file must succeed.
        ChunkedState::remove(&path).await.unwrap();

        std::fs::remove_dir(&dir).ok();
    }

    #[tokio::test]
    async fn remove_deletes_existing_sidecar() {
        let dir =
            std::env::temp_dir().join(format!("hf-fm-state-rm-existing-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("existing.chunked.part.state");

        let chunks = sample_chunks();
        let state = ChunkedState::new_fresh("etag-rm".to_owned(), 300, 3, &chunks);
        state.save_atomic(&path).await.unwrap();
        assert!(path.exists());

        ChunkedState::remove(&path).await.unwrap();
        assert!(!path.exists(), "sidecar should be gone after remove()");

        std::fs::remove_dir(&dir).ok();
    }
}
