// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model family discovery and search via the `HuggingFace` Hub API.
//!
//! Queries the HF Hub for popular models, extracts `model_type` metadata,
//! compares against locally cached families, and fetches model card metadata.

use std::collections::BTreeMap;
use std::hash::BuildHasher;

use serde::Deserialize;

use crate::error::FetchError;

/// A model found by searching the `HuggingFace` Hub.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The repository identifier (e.g., `"RWKV/RWKV7-Goose-World3-1.5B-HF"`).
    pub model_id: String,
    /// Total download count.
    pub downloads: u64,
}

/// A model family discovered from the `HuggingFace` Hub.
#[derive(Debug, Clone)]
pub struct DiscoveredFamily {
    /// The `model_type` identifier (e.g., `"gpt_neox"`, `"llama"`).
    pub model_type: String,
    /// The most-downloaded representative model for this family.
    pub top_model: String,
    /// Download count of the representative model.
    pub downloads: u64,
}

/// JSON response structure for an individual model from the HF API.
#[derive(Debug, Deserialize)]
struct ApiModelEntry {
    #[serde(rename = "modelId")]
    model_id: String,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    config: Option<ApiConfig>,
}

/// The `config` object embedded in a model API response.
#[derive(Debug, Deserialize)]
struct ApiConfig {
    model_type: Option<String>,
}

/// Access control status of a model on the `HuggingFace` Hub.
///
/// Some models require users to accept license terms before downloading.
/// The gating mode determines whether approval is automatic or manual.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GateStatus {
    /// No gate — anyone can download without restrictions.
    Open,
    /// Automatic approval after the user accepts terms on the Hub.
    Auto,
    /// Manual approval by the model author after the user requests access.
    Manual,
}

impl GateStatus {
    /// Returns `true` if the model requires accepting terms before download.
    #[must_use]
    pub const fn is_gated(&self) -> bool {
        matches!(self, Self::Auto | Self::Manual)
    }
}

impl std::fmt::Display for GateStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Open => write!(f, "open"),
            Self::Auto => write!(f, "auto"),
            Self::Manual => write!(f, "manual"),
        }
    }
}

/// Metadata from a `HuggingFace` model card.
///
/// Extracted from the single-model API endpoint
/// (`GET /api/models/{owner}/{model}`). All fields are optional
/// because model cards may omit any of them.
#[derive(Debug, Clone)]
pub struct ModelCardMetadata {
    /// SPDX license identifier (e.g., `"apache-2.0"`).
    pub license: Option<String>,
    /// Pipeline tag (e.g., `"text-generation"`).
    pub pipeline_tag: Option<String>,
    /// Tags associated with the model (e.g., `["pytorch", "safetensors"]`).
    pub tags: Vec<String>,
    /// Library name (e.g., `"transformers"`, `"vllm"`).
    pub library_name: Option<String>,
    /// Languages the model supports (e.g., `["en", "fr"]`).
    pub languages: Vec<String>,
    /// Access control status (open, auto-gated, or manually gated).
    pub gated: GateStatus,
}

/// JSON response for a single model from `GET /api/models/{model_id}`.
#[derive(Debug, Deserialize)]
struct ApiModelDetail {
    #[serde(default)]
    pipeline_tag: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    library_name: Option<String>,
    #[serde(default)]
    gated: ApiGated,
    #[serde(default, rename = "cardData")]
    card_data: Option<ApiCardData>,
}

/// The `cardData` sub-object (parsed YAML front matter from the model README).
#[derive(Debug, Deserialize)]
struct ApiCardData {
    #[serde(default)]
    license: Option<String>,
    #[serde(default)]
    language: Option<ApiLanguage>,
}

/// Languages in `cardData` can be a single string or a list of strings.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiLanguage {
    Single(String),
    Multiple(Vec<String>),
}

/// The `gated` field can be `false` (boolean) or a string like `"auto"` / `"manual"`.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiGated {
    Bool(bool),
    Mode(String),
}

impl Default for ApiGated {
    fn default() -> Self {
        Self::Bool(false)
    }
}

const PAGE_SIZE: usize = 100;
const HF_API_BASE: &str = "https://huggingface.co/api/models";

/// Queries the `HuggingFace` Hub API for top models by downloads
/// and returns families not present in the local cache.
///
/// # Arguments
///
/// * `local_families` — Set of `model_type` values already cached locally.
/// * `max_models` — Maximum number of models to scan (paginated in batches of 100).
///
/// # Errors
///
/// Returns [`FetchError::Http`] if any API request fails.
pub async fn discover_new_families<S: BuildHasher>(
    local_families: &std::collections::HashSet<String, S>,
    max_models: usize,
) -> Result<Vec<DiscoveredFamily>, FetchError> {
    let client = reqwest::Client::new();
    let mut remote_families: BTreeMap<String, (String, u64)> = BTreeMap::new();
    let mut offset: usize = 0;

    while offset < max_models {
        let limit = PAGE_SIZE.min(max_models.saturating_sub(offset));
        let url = format!(
            "{HF_API_BASE}?config=true&sort=downloads&direction=-1&limit={limit}&offset={offset}"
        );

        let response = client
            .get(url.as_str()) // BORROW: explicit .as_str()
            .send()
            .await
            .map_err(|e| FetchError::Http(e.to_string()))?;

        if !response.status().is_success() {
            return Err(FetchError::Http(format!(
                "HF API returned status {}",
                response.status()
            )));
        }

        let models: Vec<ApiModelEntry> = response
            .json()
            .await
            .map_err(|e| FetchError::Http(e.to_string()))?;

        if models.is_empty() {
            break;
        }

        for model in &models {
            // BORROW: explicit .as_ref() and .as_str() for Option<String>
            let model_type = model.config.as_ref().and_then(|c| c.model_type.as_deref());

            if let Some(mt) = model_type {
                remote_families
                    .entry(mt.to_owned())
                    .or_insert_with(|| (model.model_id.clone(), model.downloads));
            }
        }

        offset = offset.saturating_add(models.len());
    }

    // Filter to families not already cached locally
    // BORROW: explicit .as_str() instead of Deref coercion
    let discovered: Vec<DiscoveredFamily> = remote_families
        .into_iter()
        .filter(|(mt, _)| !local_families.contains(mt.as_str()))
        .map(|(model_type, (top_model, downloads))| DiscoveredFamily {
            model_type,
            top_model,
            downloads,
        })
        .collect();

    Ok(discovered)
}

/// Normalizes common quantization synonyms in a search query so that
/// variant spellings (e.g., `"8bit"`, `"8-bit"`, `"int8"`) produce
/// consistent results.
#[must_use]
fn normalize_quantization_terms(query: &str) -> String {
    /// Synonym groups: all variants map to the first (canonical) form.
    const SYNONYMS: &[(&[&str], &str)] = &[
        (&["8bit", "8-bit", "int8"], "8-bit"),
        (&["4bit", "4-bit", "int4"], "4-bit"),
        (&["fp8", "float8"], "fp8"),
    ];

    query
        .split_whitespace()
        .map(|token| {
            // BORROW: explicit .to_lowercase() for case-insensitive comparison
            let lower = token.to_lowercase();
            for &(variants, canonical) in SYNONYMS {
                // BORROW: explicit .as_str() instead of Deref coercion
                if variants.contains(&lower.as_str()) {
                    // BORROW: explicit .to_owned() for &str → owned String
                    return (*canonical).to_owned();
                }
            }
            // BORROW: explicit .to_owned() for &str → owned String
            token.to_owned()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Searches the `HuggingFace` Hub for models matching a query string.
///
/// Optionally filters by `library` framework (e.g., `"transformers"`, `"peft"`)
/// and/or `pipeline` task tag (e.g., `"text-generation"`). These filters are
/// applied server-side by the `HuggingFace` API.
///
/// Common quantization synonyms (`"8bit"` / `"8-bit"` / `"int8"`,
/// `"4bit"` / `"4-bit"` / `"int4"`, `"fp8"` / `"float8"`) are normalized
/// before querying the API so that variant spellings return consistent results.
///
/// Results are sorted by download count (most popular first).
///
/// # Arguments
///
/// * `query` — Free-text search string (e.g., `"RWKV-7"`, `"llama 3"`).
/// * `limit` — Maximum number of results to return.
/// * `library` — Optional library filter (e.g., `"peft"`, `"transformers"`).
/// * `pipeline` — Optional pipeline tag filter (e.g., `"text-generation"`).
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the API request fails.
pub async fn search_models(
    query: &str,
    limit: usize,
    library: Option<&str>,
    pipeline: Option<&str>,
) -> Result<Vec<SearchResult>, FetchError> {
    let normalized = normalize_quantization_terms(query);
    let client = reqwest::Client::new();

    // BORROW: explicit .as_str() instead of Deref coercion
    let mut query_params: Vec<(&str, &str)> = vec![
        ("search", normalized.as_str()),
        ("sort", "downloads"),
        ("direction", "-1"),
    ];
    if let Some(lib) = library {
        query_params.push(("library", lib));
    }
    if let Some(pipe) = pipeline {
        query_params.push(("pipeline_tag", pipe));
    }

    let response = client
        .get(HF_API_BASE)
        .query(&query_params)
        .query(&[("limit", limit)])
        .send()
        .await
        .map_err(|e| FetchError::Http(e.to_string()))?;

    if !response.status().is_success() {
        return Err(FetchError::Http(format!(
            "HF API returned status {}",
            response.status()
        )));
    }

    let models: Vec<ApiModelEntry> = response
        .json()
        .await
        .map_err(|e| FetchError::Http(e.to_string()))?;

    let results = models
        .into_iter()
        .map(|m| SearchResult {
            model_id: m.model_id,
            downloads: m.downloads,
        })
        .collect();

    Ok(results)
}

/// Fetches model card metadata for a specific model from the `HuggingFace` Hub.
///
/// Queries `GET https://huggingface.co/api/models/{model_id}` and extracts
/// license, pipeline tag, tags, library name, and languages from the response.
///
/// # Arguments
///
/// * `model_id` — The full model identifier (e.g., `"mistralai/Ministral-3-3B-Instruct-2512"`).
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the API request fails or the model is not found.
pub async fn fetch_model_card(model_id: &str) -> Result<ModelCardMetadata, FetchError> {
    let client = reqwest::Client::new();
    let url = format!("{HF_API_BASE}/{model_id}");

    let response = client
        .get(url.as_str()) // BORROW: explicit .as_str()
        .send()
        .await
        .map_err(|e| FetchError::Http(e.to_string()))?;

    if !response.status().is_success() {
        return Err(FetchError::Http(format!(
            "HF API returned status {} for model {model_id}",
            response.status()
        )));
    }

    let detail: ApiModelDetail = response
        .json()
        .await
        .map_err(|e| FetchError::Http(e.to_string()))?;

    let (license, languages) = if let Some(card) = detail.card_data {
        let langs = match card.language {
            Some(ApiLanguage::Single(s)) => vec![s],
            Some(ApiLanguage::Multiple(v)) => v,
            None => Vec::new(),
        };
        (card.license, langs)
    } else {
        (None, Vec::new())
    };

    let gated = match detail.gated {
        ApiGated::Bool(false) => GateStatus::Open,
        ApiGated::Mode(ref mode) if mode.eq_ignore_ascii_case("manual") => GateStatus::Manual,
        ApiGated::Bool(true) | ApiGated::Mode(_) => GateStatus::Auto,
    };

    Ok(ModelCardMetadata {
        license,
        pipeline_tag: detail.pipeline_tag,
        tags: detail.tags,
        library_name: detail.library_name,
        languages,
        gated,
    })
}

/// Fetches the raw README text for a `HuggingFace` model repository.
///
/// Downloads `README.md` from the repository at the given revision.
/// Returns `Ok(None)` if the file does not exist (HTTP 404).
///
/// # Arguments
///
/// * `model_id` — The full model identifier (e.g., `"mistralai/Ministral-3-3B-Instruct-2512"`).
/// * `revision` — Git revision to fetch (defaults to `"main"` when `None`).
/// * `token` — Optional authentication token.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the request fails (other than 404).
pub async fn fetch_readme(
    model_id: &str,
    revision: Option<&str>,
    token: Option<&str>,
) -> Result<Option<String>, FetchError> {
    let rev = revision.unwrap_or("main");
    let url = crate::chunked::build_download_url(model_id, rev, "README.md");
    let client = crate::chunked::build_client(token)?;

    let response = client
        .get(url.as_str()) // BORROW: explicit .as_str() instead of Deref coercion
        .send()
        .await
        .map_err(|e| FetchError::Http(format!("failed to fetch README for {model_id}: {e}")))?;

    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }

    if !response.status().is_success() {
        return Err(FetchError::Http(format!(
            "README request for {model_id} returned status {}",
            response.status()
        )));
    }

    let text = response
        .text()
        .await
        .map_err(|e| FetchError::Http(format!("failed to read README for {model_id}: {e}")))?;

    Ok(Some(text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_8bit_variants() {
        assert_eq!(normalize_quantization_terms("AWQ 8bit"), "AWQ 8-bit");
        assert_eq!(normalize_quantization_terms("AWQ 8-bit"), "AWQ 8-bit");
        assert_eq!(normalize_quantization_terms("AWQ int8"), "AWQ 8-bit");
        assert_eq!(normalize_quantization_terms("AWQ INT8"), "AWQ 8-bit");
    }

    #[test]
    fn normalize_4bit_variants() {
        assert_eq!(normalize_quantization_terms("GPTQ 4bit"), "GPTQ 4-bit");
        assert_eq!(normalize_quantization_terms("GPTQ INT4"), "GPTQ 4-bit");
        assert_eq!(normalize_quantization_terms("GPTQ 4-bit"), "GPTQ 4-bit");
    }

    #[test]
    fn normalize_fp8_variants() {
        assert_eq!(normalize_quantization_terms("FP8"), "fp8");
        assert_eq!(normalize_quantization_terms("float8"), "fp8");
        assert_eq!(normalize_quantization_terms("fp8"), "fp8");
    }

    #[test]
    fn normalize_passthrough() {
        assert_eq!(normalize_quantization_terms("llama 3"), "llama 3");
        assert_eq!(normalize_quantization_terms("RWKV-7"), "RWKV-7");
    }
}
