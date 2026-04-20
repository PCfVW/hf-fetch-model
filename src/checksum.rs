// SPDX-License-Identifier: MIT OR Apache-2.0

//! SHA256 checksum verification for downloaded files.
//!
//! Computes a file's SHA256 digest and compares it against
//! the expected hash from `HuggingFace` repository metadata.

use std::path::Path;

use sha2::{Digest, Sha256};

use crate::error::FetchError;

/// Verifies a downloaded file's SHA256 against the expected hex digest.
///
/// # Errors
///
/// Returns [`FetchError::Checksum`] if the hashes do not match.
/// Returns [`FetchError::Io`] if the file cannot be read.
pub async fn verify_sha256(
    path: &Path,
    filename: &str,
    expected_hex: &str,
) -> Result<(), FetchError> {
    // BORROW: explicit .to_path_buf() for owned PathBuf in error context
    let path_owned = path.to_path_buf();
    let actual_hex = compute_sha256(path).await.map_err(|e| FetchError::Io {
        path: path_owned,
        source: e,
    })?;

    if actual_hex != expected_hex {
        // BORROW: explicit .to_owned() for &str → owned String fields
        return Err(FetchError::Checksum {
            filename: filename.to_owned(),
            expected: expected_hex.to_owned(),
            actual: actual_hex,
        });
    }

    Ok(())
}

/// Computes the SHA256 hex digest of a file.
///
/// Reads the file in chunks to avoid loading the entire file into memory,
/// which is important for large model weight files (multi-GB).
///
/// # Errors
///
/// Returns [`std::io::Error`] if the file cannot be read.
async fn compute_sha256(path: &Path) -> Result<String, std::io::Error> {
    // Read file in 8 KiB chunks on a blocking thread to avoid
    // blocking the async runtime with synchronous I/O.
    // BORROW: explicit .to_path_buf() — owned PathBuf needed to move into closure
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || {
        use std::fmt::Write as _;
        use std::io::Read;

        let mut file = std::fs::File::open(&path)?;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 8192];

        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            if let Some(chunk) = buffer.get(..bytes_read) {
                hasher.update(chunk);
            }
        }

        // Manual lowercase-hex encoding: `sha2` 0.11 returns
        // `hybrid_array::Array<u8, _>`, which (unlike `generic_array::GenericArray`)
        // does not implement `fmt::LowerHex`, so `format!("{digest:x}")` no longer
        // compiles. `write!` into `String` is infallible; the `Result` is discarded.
        let digest = hasher.finalize();
        let mut hex = String::with_capacity(digest.len() * 2);
        for &b in &digest {
            let _ = write!(&mut hex, "{b:02x}");
        }
        Ok(hex)
    })
    .await
    .map_err(std::io::Error::other)?
}

#[cfg(test)]
mod tests {
    #![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use std::io::Write;

    #[tokio::test]
    async fn test_compute_sha256_known_value() {
        // SHA256("hello\n") = 5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03
        let dir = std::env::temp_dir().join("hf_fetch_model_test_sha256");
        let _ = std::fs::create_dir_all(&dir);
        let file_path = dir.join("hello.txt");
        {
            let mut f = std::fs::File::create(&file_path).unwrap();
            f.write_all(b"hello\n").unwrap();
        }

        let hex = compute_sha256(&file_path).await.unwrap();
        assert_eq!(
            hex,
            "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_verify_sha256_match() {
        let dir = std::env::temp_dir().join("hf_fetch_model_test_verify");
        let _ = std::fs::create_dir_all(&dir);
        let file_path = dir.join("verify.txt");
        {
            let mut f = std::fs::File::create(&file_path).unwrap();
            f.write_all(b"hello\n").unwrap();
        }

        let result = verify_sha256(
            &file_path,
            "verify.txt",
            "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03",
        )
        .await;
        assert!(result.is_ok());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_verify_sha256_mismatch() {
        let dir = std::env::temp_dir().join("hf_fetch_model_test_mismatch");
        let _ = std::fs::create_dir_all(&dir);
        let file_path = dir.join("mismatch.txt");
        {
            let mut f = std::fs::File::create(&file_path).unwrap();
            f.write_all(b"hello\n").unwrap();
        }

        let result = verify_sha256(&file_path, "mismatch.txt", "0000000000000000").await;
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
