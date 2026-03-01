// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model family discovery via the `HuggingFace` Hub API.
//!
//! Queries the HF Hub for popular models, extracts `model_type` metadata,
//! and compares against locally cached families.

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

/// Searches the `HuggingFace` Hub for models matching a query string.
///
/// Results are sorted by download count (most popular first).
///
/// # Arguments
///
/// * `query` — Free-text search string (e.g., `"RWKV-7"`, `"llama 3"`).
/// * `limit` — Maximum number of results to return.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the API request fails.
pub async fn search_models(query: &str, limit: usize) -> Result<Vec<SearchResult>, FetchError> {
    let client = reqwest::Client::new();

    let url = format!("{HF_API_BASE}?search={query}&sort=downloads&direction=-1&limit={limit}");

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

    let results = models
        .into_iter()
        .map(|m| SearchResult {
            model_id: m.model_id,
            downloads: m.downloads,
        })
        .collect();

    Ok(results)
}
