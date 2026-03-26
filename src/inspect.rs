// SPDX-License-Identifier: MIT OR Apache-2.0

//! Safetensors header inspection (local and remote).
//!
//! Reads tensor metadata (names, shapes, dtypes, byte offsets) from
//! `.safetensors` files without downloading full weight data. Supports
//! cache-first resolution with HTTP Range request fallback.
//!
//! The primary types are [`TensorInfo`] (per-tensor metadata),
//! [`SafetensorsHeaderInfo`] (parsed header), and [`ShardedIndex`]
//! (shard-to-tensor mapping for sharded models).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::cache;
use crate::chunked;
use crate::error::FetchError;

// -----------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------

/// Metadata for a single tensor from a `.safetensors` header.
///
/// This is hf-fetch-model's own type — lightweight, no quantization logic.
/// Consumers (e.g., anamnesis) map this into their own richer types.
#[derive(Debug, Clone, Serialize)]
pub struct TensorInfo {
    /// Tensor name (e.g., `"model.layers.0.self_attn.q_proj.weight"`).
    pub name: String,
    /// Element dtype string as it appears in the header (e.g., `"F8_E4M3"`, `"BF16"`).
    pub dtype: String,
    /// Tensor shape (e.g., `[7168, 7168]`).
    pub shape: Vec<usize>,
    /// Byte offset range `[start, end)` within the data section of the file.
    pub data_offsets: (u64, u64),
}

impl TensorInfo {
    /// Total number of elements (product of shape dimensions).
    ///
    /// Returns `1` for a scalar (empty shape).
    #[must_use]
    pub fn num_elements(&self) -> u64 {
        self.shape.iter().fold(1u64, |acc, &d| {
            // CAST: usize → u64, dimension values fit in u64
            #[allow(clippy::as_conversions)]
            let dim = d as u64;
            acc.saturating_mul(dim)
        })
    }

    /// Byte length of the tensor data (`end - start`).
    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.data_offsets.1.saturating_sub(self.data_offsets.0)
    }

    /// Bytes per element for the tensor's dtype, if recognized.
    ///
    /// Returns `None` for unknown dtype strings. Recognized dtypes:
    ///
    /// | Dtype string | Bytes | Notes |
    /// |-------------|-------|-------|
    /// | `"BOOL"` | 1 | |
    /// | `"U8"`, `"I8"` | 1 | |
    /// | `"F8_E4M3"`, `"F8_E5M2"` | 1 | FP8 variants |
    /// | `"U16"`, `"I16"`, `"F16"`, `"BF16"` | 2 | |
    /// | `"U32"`, `"I32"`, `"F32"` | 4 | |
    /// | `"U64"`, `"I64"`, `"F64"` | 8 | |
    #[must_use]
    pub fn dtype_bytes(&self) -> Option<usize> {
        // BORROW: explicit .as_str() instead of Deref coercion
        match self.dtype.as_str() {
            "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => Some(1),
            "U16" | "I16" | "F16" | "BF16" => Some(2),
            "U32" | "I32" | "F32" => Some(4),
            "U64" | "I64" | "F64" => Some(8),
            _ => None,
        }
    }
}

/// Parsed `.safetensors` header metadata.
#[derive(Debug, Clone, Serialize)]
pub struct SafetensorsHeaderInfo {
    /// All tensors in the header, in the order they appear in the JSON.
    pub tensors: Vec<TensorInfo>,
    /// Raw `__metadata__` entries, if present.
    ///
    /// For quantized models, this typically contains entries like
    /// `quant_method`, `bits`, `group_size` that consumers like anamnesis
    /// use to distinguish GPTQ from AWQ without downloading weights.
    pub metadata: Option<HashMap<String, String>>,
    /// Size of the JSON header in bytes.
    pub header_size: u64,
    /// Total file size in bytes (header + data), if known.
    ///
    /// **Source:** for local files, from `std::fs::metadata().len()`. For HTTP
    /// Range requests, extracted from the `Content-Range` response header of
    /// the first request (`bytes 0-7/TOTAL` → `TOTAL`). This is free — no
    /// extra request needed.
    pub file_size: Option<u64>,
}

impl SafetensorsHeaderInfo {
    /// Total parameter count across all tensors.
    #[must_use]
    pub fn total_params(&self) -> u64 {
        self.tensors
            .iter()
            .map(TensorInfo::num_elements)
            .fold(0u64, u64::saturating_add)
    }

    /// Returns tensors matching a dtype string (e.g., `"F8_E4M3"`).
    #[must_use]
    pub fn tensors_with_dtype(&self, dtype: &str) -> Vec<&TensorInfo> {
        self.tensors
            .iter()
            // BORROW: explicit .as_str() instead of Deref coercion
            .filter(|t| t.dtype.as_str() == dtype)
            .collect()
    }
}

/// The source from which a header was read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InspectSource {
    /// Read from local cache (no network).
    Cached,
    /// Fetched via HTTP Range requests.
    Remote,
}

/// Parsed `model.safetensors.index.json` for a sharded model.
#[derive(Debug, Clone, Serialize)]
pub struct ShardedIndex {
    /// Mapping from tensor name to shard filename.
    pub weight_map: HashMap<String, String>,
    /// Ordered list of unique shard filenames.
    pub shards: Vec<String>,
    /// Raw metadata from the index, if present.
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// PEFT adapter configuration parsed from `adapter_config.json`.
///
/// Contains the key fields that identify an adapter: the PEFT type,
/// base model, `LoRA` rank and scaling parameters, and target modules.
/// All fields are optional because adapter configs vary across PEFT methods.
#[derive(Debug, Clone, Serialize)]
pub struct AdapterConfig {
    /// PEFT method type (e.g., `"LORA"`, `"ADALORA"`, `"IA3"`).
    pub peft_type: Option<String>,
    /// The base model this adapter was trained on.
    pub base_model_name_or_path: Option<String>,
    /// `LoRA` rank (the `r` parameter). Only meaningful for `LoRA`-family methods.
    pub r: Option<u32>,
    /// `LoRA` alpha scaling factor. Only meaningful for `LoRA`-family methods.
    pub lora_alpha: Option<f64>,
    /// List of model modules targeted by the adapter.
    pub target_modules: Vec<String>,
    /// Task type the adapter was trained for (e.g., `"CAUSAL_LM"`).
    pub task_type: Option<String>,
}

// -----------------------------------------------------------------------
// JSON parsing
// -----------------------------------------------------------------------

/// Raw tensor entry as it appears in the safetensors JSON header.
#[derive(serde::Deserialize)]
struct RawTensorEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (u64, u64),
}

/// Parsed tensor list and optional metadata from a safetensors header.
type ParsedHeader = (Vec<TensorInfo>, Option<HashMap<String, String>>);

/// Parses the safetensors JSON header bytes into tensor metadata.
///
/// Extracts the `__metadata__` key separately (if present).
fn parse_header_json(json_bytes: &[u8], filename: &str) -> Result<ParsedHeader, FetchError> {
    let raw: HashMap<String, serde_json::Value> =
        serde_json::from_slice(json_bytes).map_err(|e| FetchError::SafetensorsHeader {
            filename: filename.to_owned(),
            reason: format!("failed to parse header JSON: {e}"),
        })?;

    let mut metadata: Option<HashMap<String, String>> = None;
    let mut tensors = Vec::new();

    for (key, value) in &raw {
        if key == "__metadata__" {
            if let Some(obj) = value.as_object() {
                let mut meta_map = HashMap::new();
                for (mk, mv) in obj {
                    // BORROW: explicit .to_string() for Value → String (strips quotes from strings)
                    let v_str = if let Some(s) = mv.as_str() {
                        s.to_owned()
                    } else {
                        mv.to_string()
                    };
                    // BORROW: explicit .clone() for owned String
                    meta_map.insert(mk.clone(), v_str);
                }
                metadata = Some(meta_map);
            }
            continue;
        }

        let entry: RawTensorEntry =
            serde_json::from_value(value.clone()).map_err(|e| FetchError::SafetensorsHeader {
                filename: filename.to_owned(),
                reason: format!("failed to parse tensor \"{key}\": {e}"),
            })?;

        tensors.push(TensorInfo {
            // BORROW: explicit .clone() for owned String
            name: key.clone(),
            dtype: entry.dtype,
            shape: entry.shape,
            data_offsets: entry.data_offsets,
        });
    }

    // Sort by data offset start to preserve file order.
    tensors.sort_by_key(|t| t.data_offsets.0);

    Ok((tensors, metadata))
}

// -----------------------------------------------------------------------
// Cache resolution
// -----------------------------------------------------------------------

/// Resolves a cached file path for a given repo, revision, and filename.
///
/// Returns `None` if the file is not in the local cache.
fn resolve_cached_path(repo_id: &str, revision: &str, filename: &str) -> Option<PathBuf> {
    let cache_dir = cache::hf_cache_dir().ok()?;
    let repo_folder = chunked::repo_folder_name(repo_id);
    let repo_dir = cache_dir.join(&repo_folder);
    let commit_hash = cache::read_ref(&repo_dir, revision)?;
    let cached_path = repo_dir.join("snapshots").join(commit_hash).join(filename);
    if cached_path.exists() {
        Some(cached_path)
    } else {
        None
    }
}

// -----------------------------------------------------------------------
// Local file reading
// -----------------------------------------------------------------------

/// Inspects a single `.safetensors` file's header from a local file path.
///
/// Reads the first `8 + header_size` bytes from disk. Does not read tensor data.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the file cannot be read.
/// Returns [`FetchError::SafetensorsHeader`] if the header is malformed.
pub fn inspect_safetensors_local(path: &Path) -> Result<SafetensorsHeaderInfo, FetchError> {
    use std::io::Read;

    let file_size = std::fs::metadata(path)
        .map_err(|e| FetchError::Io {
            path: path.to_path_buf(),
            source: e,
        })?
        .len();

    // BORROW: explicit .to_string_lossy() for Path → str conversion
    let filename = path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().to_string(),
    );

    let mut file = std::fs::File::open(path).map_err(|e| FetchError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;

    // Read 8-byte header length prefix (little-endian u64).
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf).map_err(|e| FetchError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let header_size = u64::from_le_bytes(len_buf);

    // Sanity check: header cannot be larger than the file.
    if header_size.saturating_add(8) > file_size {
        return Err(FetchError::SafetensorsHeader {
            filename,
            reason: format!("header length {header_size} exceeds file size {file_size}"),
        });
    }

    // Read the JSON header.
    // CAST: u64 → usize, header size bounded by file size (checked above)
    #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
    let json_len = header_size as usize;
    let mut json_buf = vec![0u8; json_len];
    file.read_exact(&mut json_buf).map_err(|e| FetchError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;

    // BORROW: explicit .as_str() instead of Deref coercion
    let (tensors, metadata) = parse_header_json(&json_buf, filename.as_str())?;

    Ok(SafetensorsHeaderInfo {
        tensors,
        metadata,
        header_size,
        file_size: Some(file_size),
    })
}

// -----------------------------------------------------------------------
// Remote fetching (HTTP Range requests)
// -----------------------------------------------------------------------

/// Fetches safetensors header bytes via two HTTP Range requests.
///
/// 1. `Range: bytes=0-7` → 8-byte header length (little-endian `u64`)
/// 2. `Range: bytes=8-{8+length-1}` → JSON header
///
/// Returns `(json_bytes, total_file_size)`. The file size is extracted from
/// the `Content-Range` header of the first request.
async fn fetch_header_bytes(
    client: &reqwest::Client,
    url: &str,
    filename: &str,
) -> Result<(Vec<u8>, Option<u64>), FetchError> {
    // Request 1: 8-byte length prefix.
    let resp1 = client
        .get(url)
        .header(reqwest::header::RANGE, "bytes=0-7")
        .send()
        .await
        .map_err(|e| {
            FetchError::Http(format!("failed to fetch header length for {filename}: {e}"))
        })?;

    if !resp1.status().is_success() && resp1.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        return Err(FetchError::Http(format!(
            "Range request for {filename} returned status {}",
            resp1.status()
        )));
    }

    // Extract total file size from Content-Range: bytes 0-7/{total}
    let file_size = resp1
        .headers()
        .get(reqwest::header::CONTENT_RANGE)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.split('/').next_back())
        .and_then(|s| s.parse::<u64>().ok());

    let len_bytes = resp1.bytes().await.map_err(|e| {
        FetchError::Http(format!("failed to read header length for {filename}: {e}"))
    })?;

    if len_bytes.len() < 8 {
        return Err(FetchError::SafetensorsHeader {
            filename: filename.to_owned(),
            reason: format!(
                "expected 8 bytes for length prefix, got {}",
                len_bytes.len()
            ),
        });
    }

    // INDEX: first 8 bytes guaranteed by length check above
    #[allow(clippy::indexing_slicing)]
    let header_size = u64::from_le_bytes([
        len_bytes[0],
        len_bytes[1],
        len_bytes[2],
        len_bytes[3],
        len_bytes[4],
        len_bytes[5],
        len_bytes[6],
        len_bytes[7],
    ]);

    // Request 2: JSON header.
    let range_end = 8u64.saturating_add(header_size).saturating_sub(1);
    let range_header = format!("bytes=8-{range_end}");
    let resp2 = client
        .get(url)
        // BORROW: explicit .as_str() instead of Deref coercion
        .header(reqwest::header::RANGE, range_header.as_str())
        .send()
        .await
        .map_err(|e| {
            FetchError::Http(format!("failed to fetch header JSON for {filename}: {e}"))
        })?;

    if !resp2.status().is_success() && resp2.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        return Err(FetchError::Http(format!(
            "Range request for {filename} header JSON returned status {}",
            resp2.status()
        )));
    }

    let json_bytes = resp2
        .bytes()
        .await
        .map_err(|e| FetchError::Http(format!("failed to read header JSON for {filename}: {e}")))?;

    Ok((json_bytes.to_vec(), file_size))
}

// -----------------------------------------------------------------------
// Public API: single-file inspection
// -----------------------------------------------------------------------

/// Inspects a single `.safetensors` file's header (cache-first).
///
/// Checks the local HF cache first. If the file is cached, reads the header
/// from disk with zero network requests. Otherwise, falls back to two HTTP
/// Range requests (8-byte length prefix + JSON header). Does not download
/// tensor data in either case.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the Range requests fail.
/// Returns [`FetchError::SafetensorsHeader`] if the header is malformed.
pub async fn inspect_safetensors(
    repo_id: &str,
    filename: &str,
    token: Option<&str>,
    revision: Option<&str>,
) -> Result<(SafetensorsHeaderInfo, InspectSource), FetchError> {
    let rev = revision.unwrap_or("main");

    // Try local cache first.
    if let Some(cached_path) = resolve_cached_path(repo_id, rev, filename) {
        let info = inspect_safetensors_local(&cached_path)?;
        return Ok((info, InspectSource::Cached));
    }

    // Fall back to HTTP Range requests.
    let client = chunked::build_client(token)?;
    let url = chunked::build_download_url(repo_id, rev, filename);

    // BORROW: explicit .as_str() instead of Deref coercion
    let (json_bytes, file_size) = fetch_header_bytes(&client, url.as_str(), filename).await?;

    // CAST: usize → u64, JSON buffer length is always small
    #[allow(clippy::as_conversions)]
    let header_size = json_bytes.len() as u64;

    let (tensors, metadata) = parse_header_json(&json_bytes, filename)?;

    Ok((
        SafetensorsHeaderInfo {
            tensors,
            metadata,
            header_size,
            file_size,
        },
        InspectSource::Remote,
    ))
}

/// Inspects a single `.safetensors` file from cache only.
///
/// Resolves the file in the local HF cache using the given `repo_id`,
/// `revision`, and `filename`. Returns an error if the file is not cached.
///
/// # Errors
///
/// Returns [`FetchError::SafetensorsHeader`] if the file is not in the cache.
/// Returns [`FetchError::Io`] if the cached file cannot be read.
/// Returns [`FetchError::SafetensorsHeader`] if the header is malformed.
pub fn inspect_safetensors_cached(
    repo_id: &str,
    filename: &str,
    revision: Option<&str>,
) -> Result<SafetensorsHeaderInfo, FetchError> {
    let rev = revision.unwrap_or("main");

    let cached_path = resolve_cached_path(repo_id, rev, filename).ok_or_else(|| {
        FetchError::SafetensorsHeader {
            filename: filename.to_owned(),
            reason: format!("file not found in local cache for {repo_id} ({rev})"),
        }
    })?;

    inspect_safetensors_local(&cached_path)
}

// -----------------------------------------------------------------------
// Public API: multi-file inspection
// -----------------------------------------------------------------------

/// Inspects all `.safetensors` files in a repository (cache-first per file).
///
/// Fetches the file listing via `list_repo_files_with_metadata()`, then
/// inspects each `.safetensors` file's header via [`inspect_safetensors()`].
/// For each file, checks the local cache first and only makes HTTP Range
/// requests on cache miss. Returns full per-shard headers in filename order.
///
/// For a lightweight summary of sharded models (tensor counts per shard
/// without fetching individual headers), use [`fetch_shard_index()`] instead.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the metadata or Range requests fail.
pub async fn inspect_repo_safetensors(
    repo_id: &str,
    token: Option<&str>,
    revision: Option<&str>,
) -> Result<Vec<(String, SafetensorsHeaderInfo, InspectSource)>, FetchError> {
    let files = crate::repo::list_repo_files_with_metadata(repo_id, token, revision).await?;

    let safetensors_files: Vec<String> = files
        .into_iter()
        .filter(|f| f.filename.ends_with(".safetensors"))
        .map(|f| f.filename)
        .collect();

    if safetensors_files.is_empty() {
        return Ok(Vec::new());
    }

    let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(4));
    let mut handles = Vec::new();

    for filename in safetensors_files {
        let sem = semaphore.clone();
        let repo = repo_id.to_owned();
        let tok = token.map(str::to_owned);
        let rev = revision.map(str::to_owned);

        handles.push(tokio::spawn(async move {
            let _permit = sem
                .acquire()
                .await
                .map_err(|e| FetchError::Http(format!("semaphore error: {e}")))?;
            // BORROW: explicit .as_deref() for Option<String> → Option<&str>
            let (info, source) =
                inspect_safetensors(&repo, &filename, tok.as_deref(), rev.as_deref()).await?;
            Ok::<_, FetchError>((filename, info, source))
        }));
    }

    let mut results = Vec::new();
    for handle in handles {
        let result = handle
            .await
            .map_err(|e| FetchError::Http(format!("task join error: {e}")))?;
        results.push(result?);
    }

    results.sort_by(|a, b| a.0.cmp(&b.0));

    Ok(results)
}

/// Inspects all `.safetensors` files in a cached repository (no network).
///
/// Walks the snapshot directory and inspects each `.safetensors` file's
/// header from local disk. Returns results in filename order.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the cache directory cannot be read.
/// Returns [`FetchError::SafetensorsHeader`] if any header is malformed.
pub fn inspect_repo_safetensors_cached(
    repo_id: &str,
    revision: Option<&str>,
) -> Result<Vec<(String, SafetensorsHeaderInfo)>, FetchError> {
    let rev = revision.unwrap_or("main");
    let cache_dir = cache::hf_cache_dir()?;
    let repo_folder = chunked::repo_folder_name(repo_id);
    let repo_dir = cache_dir.join(&repo_folder);

    let Some(commit_hash) = cache::read_ref(&repo_dir, rev) else {
        return Ok(Vec::new());
    };

    let snapshot_dir = repo_dir.join("snapshots").join(commit_hash);
    if !snapshot_dir.exists() {
        return Ok(Vec::new());
    }

    let mut results = Vec::new();
    collect_safetensors_recursive(&snapshot_dir, "", &mut results)?;
    results.sort_by(|a, b| a.0.cmp(&b.0));

    Ok(results)
}

/// Recursively finds and inspects `.safetensors` files in a snapshot directory.
fn collect_safetensors_recursive(
    dir: &Path,
    prefix: &str,
    results: &mut Vec<(String, SafetensorsHeaderInfo)>,
) -> Result<(), FetchError> {
    let entries = std::fs::read_dir(dir).map_err(|e| FetchError::Io {
        path: dir.to_path_buf(),
        source: e,
    })?;

    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        // BORROW: explicit .to_string_lossy() for OsString → str conversion
        let name = entry.file_name().to_string_lossy().to_string();

        if path.is_dir() {
            let child_prefix = if prefix.is_empty() {
                name
            } else {
                format!("{prefix}/{name}")
            };
            collect_safetensors_recursive(&path, &child_prefix, results)?;
        } else if name.ends_with(".safetensors") {
            let filename = if prefix.is_empty() {
                name
            } else {
                format!("{prefix}/{name}")
            };
            let info = inspect_safetensors_local(&path)?;
            results.push((filename, info));
        }
    }

    Ok(())
}

// -----------------------------------------------------------------------
// Shard index
// -----------------------------------------------------------------------

/// Raw JSON structure of `model.safetensors.index.json`.
#[derive(serde::Deserialize)]
struct RawShardIndex {
    weight_map: HashMap<String, String>,
    #[serde(default)]
    metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Fetches and parses the shard index for a sharded `.safetensors` model (cache-first).
///
/// Returns `Ok(None)` if the repo has no `model.safetensors.index.json` (i.e.,
/// the model is not sharded or uses a single `.safetensors` file).
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the index fetch fails.
/// Returns [`FetchError::SafetensorsHeader`] if the index JSON is malformed.
pub async fn fetch_shard_index(
    repo_id: &str,
    token: Option<&str>,
    revision: Option<&str>,
) -> Result<Option<ShardedIndex>, FetchError> {
    let rev = revision.unwrap_or("main");
    let index_filename = "model.safetensors.index.json";

    // Try local cache first.
    if let Some(cached_path) = resolve_cached_path(repo_id, rev, index_filename) {
        let content = std::fs::read_to_string(&cached_path).map_err(|e| FetchError::Io {
            path: cached_path,
            source: e,
        })?;
        let index = parse_shard_index_json(&content, repo_id)?;
        return Ok(Some(index));
    }

    // Fall back to HTTP.
    let client = chunked::build_client(token)?;
    let url = chunked::build_download_url(repo_id, rev, index_filename);

    // BORROW: explicit .as_str() instead of Deref coercion
    let response =
        client.get(url.as_str()).send().await.map_err(|e| {
            FetchError::Http(format!("failed to fetch shard index for {repo_id}: {e}"))
        })?;

    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }

    if !response.status().is_success() {
        return Err(FetchError::Http(format!(
            "shard index request for {repo_id} returned status {}",
            response.status()
        )));
    }

    let content = response
        .text()
        .await
        .map_err(|e| FetchError::Http(format!("failed to read shard index for {repo_id}: {e}")))?;

    let index = parse_shard_index_json(&content, repo_id)?;
    Ok(Some(index))
}

/// Fetches the shard index from cache only (no network).
///
/// Returns `Ok(None)` if the index file is not cached.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the cached file cannot be read.
/// Returns [`FetchError::SafetensorsHeader`] if the index JSON is malformed.
pub fn fetch_shard_index_cached(
    repo_id: &str,
    revision: Option<&str>,
) -> Result<Option<ShardedIndex>, FetchError> {
    let rev = revision.unwrap_or("main");
    let index_filename = "model.safetensors.index.json";

    let Some(cached_path) = resolve_cached_path(repo_id, rev, index_filename) else {
        return Ok(None);
    };

    let content = std::fs::read_to_string(&cached_path).map_err(|e| FetchError::Io {
        path: cached_path,
        source: e,
    })?;

    let index = parse_shard_index_json(&content, repo_id)?;
    Ok(Some(index))
}

/// Parses shard index JSON into a `ShardedIndex`.
fn parse_shard_index_json(content: &str, repo_id: &str) -> Result<ShardedIndex, FetchError> {
    let raw: RawShardIndex =
        serde_json::from_str(content).map_err(|e| FetchError::SafetensorsHeader {
            filename: "model.safetensors.index.json".to_owned(),
            reason: format!("failed to parse shard index for {repo_id}: {e}"),
        })?;

    // Collect unique shard filenames in sorted order.
    let mut shard_set: Vec<String> = raw.weight_map.values().cloned().collect();
    shard_set.sort();
    shard_set.dedup();

    Ok(ShardedIndex {
        weight_map: raw.weight_map,
        shards: shard_set,
        metadata: raw.metadata,
    })
}

// -----------------------------------------------------------------------
// Param formatting helper
// -----------------------------------------------------------------------

/// Formats a parameter count with a compact suffix (e.g., `927.0M`, `1.02B`).
#[must_use]
pub fn format_params(count: u64) -> String {
    // CAST: u64 → f64, precision loss acceptable; value is a display-only scalar
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let val = count as f64;

    if count >= 1_000_000_000 {
        format!("{:.2}B", val / 1_000_000_000.0)
    } else if count >= 1_000_000 {
        format!("{:.1}M", val / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}K", val / 1_000.0)
    } else {
        count.to_string()
    }
}

// -----------------------------------------------------------------------
// Adapter config
// -----------------------------------------------------------------------

/// Raw JSON structure of `adapter_config.json`.
#[derive(serde::Deserialize)]
struct RawAdapterConfig {
    #[serde(default)]
    peft_type: Option<String>,
    #[serde(default)]
    base_model_name_or_path: Option<String>,
    #[serde(default)]
    r: Option<u32>,
    #[serde(default)]
    lora_alpha: Option<f64>,
    #[serde(default)]
    target_modules: Option<AdapterTargetModules>,
    #[serde(default)]
    task_type: Option<String>,
}

/// `target_modules` in adapter configs can be a list of strings or a single string.
#[derive(serde::Deserialize)]
#[serde(untagged)]
enum AdapterTargetModules {
    /// A list of module name strings.
    List(Vec<String>),
    /// A single module name string.
    Single(String),
}

/// Fetches and parses `adapter_config.json` for a PEFT adapter repository (cache-first).
///
/// Returns `Ok(None)` if the file does not exist (HTTP 404), meaning the
/// repository is not a PEFT adapter.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the request fails (other than 404).
/// Returns [`FetchError::SafetensorsHeader`] if the JSON is malformed.
pub async fn fetch_adapter_config(
    repo_id: &str,
    token: Option<&str>,
    revision: Option<&str>,
) -> Result<Option<AdapterConfig>, FetchError> {
    let rev = revision.unwrap_or("main");
    let config_filename = "adapter_config.json";

    // Try local cache first.
    if let Some(cached_path) = resolve_cached_path(repo_id, rev, config_filename) {
        let content = std::fs::read_to_string(&cached_path).map_err(|e| FetchError::Io {
            path: cached_path,
            source: e,
        })?;
        let config = parse_adapter_config_json(&content, repo_id)?;
        return Ok(Some(config));
    }

    // Fall back to HTTP.
    let client = chunked::build_client(token)?;
    let url = chunked::build_download_url(repo_id, rev, config_filename);

    // BORROW: explicit .as_str() instead of Deref coercion
    let response = client.get(url.as_str()).send().await.map_err(|e| {
        FetchError::Http(format!("failed to fetch adapter config for {repo_id}: {e}"))
    })?;

    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }

    if !response.status().is_success() {
        return Err(FetchError::Http(format!(
            "adapter config request for {repo_id} returned status {}",
            response.status()
        )));
    }

    let content = response.text().await.map_err(|e| {
        FetchError::Http(format!("failed to read adapter config for {repo_id}: {e}"))
    })?;

    let config = parse_adapter_config_json(&content, repo_id)?;
    Ok(Some(config))
}

/// Fetches the adapter config from cache only (no network).
///
/// Returns `Ok(None)` if the file is not cached.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the cached file cannot be read.
/// Returns [`FetchError::SafetensorsHeader`] if the JSON is malformed.
pub fn fetch_adapter_config_cached(
    repo_id: &str,
    revision: Option<&str>,
) -> Result<Option<AdapterConfig>, FetchError> {
    let rev = revision.unwrap_or("main");
    let config_filename = "adapter_config.json";

    let Some(cached_path) = resolve_cached_path(repo_id, rev, config_filename) else {
        return Ok(None);
    };

    let content = std::fs::read_to_string(&cached_path).map_err(|e| FetchError::Io {
        path: cached_path,
        source: e,
    })?;

    let config = parse_adapter_config_json(&content, repo_id)?;
    Ok(Some(config))
}

/// Parses adapter config JSON into an [`AdapterConfig`].
fn parse_adapter_config_json(content: &str, repo_id: &str) -> Result<AdapterConfig, FetchError> {
    let raw: RawAdapterConfig =
        serde_json::from_str(content).map_err(|e| FetchError::SafetensorsHeader {
            filename: "adapter_config.json".to_owned(),
            reason: format!("failed to parse adapter config for {repo_id}: {e}"),
        })?;

    let target_modules = match raw.target_modules {
        Some(AdapterTargetModules::List(v)) => v,
        Some(AdapterTargetModules::Single(s)) => vec![s],
        None => Vec::new(),
    };

    Ok(AdapterConfig {
        peft_type: raw.peft_type,
        base_model_name_or_path: raw.base_model_name_or_path,
        r: raw.r,
        lora_alpha: raw.lora_alpha,
        target_modules,
        task_type: raw.task_type,
    })
}
