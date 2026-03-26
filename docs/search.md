# Search

The `search` command queries the HuggingFace Hub for models, sorted by download count.

## Basic search

```
$ hf-fm search RWKV-7
Models matching "RWKV-7" (by downloads):

  hf-fm RWKV/RWKV7-Goose-World3-1.5B-HF                 (12,300 downloads)
  hf-fm RWKV/RWKV7-Goose-World3-0.1B-HF                 (5,100 downloads)
  ...
```

## Slash normalization

Slashes in queries are treated as spaces for broader API matching. This means `mistralai/3B` searches for "mistralai 3B" instead of failing:

```
$ hf-fm search mistralai/3B
Models matching "mistralai/3B" (by downloads):

  hf-fm mistralai/Voxtral-Mini-3B-2507                   (487,600 downloads)
  hf-fm mistralai/Ministral-3-3B-Instruct-2512           (159,700 downloads)
  ...
```

## Quantization term normalization

Common quantization synonyms are normalized automatically before querying the HuggingFace API, so variant spellings return the same results:

| Variants | Normalized to |
|----------|---------------|
| `8bit`, `8-bit`, `int8`, `INT8` | `8-bit` |
| `4bit`, `4-bit`, `int4`, `INT4` | `4-bit` |
| `fp8`, `FP8`, `float8` | `fp8` |

```
$ hf-fm search "AWQ 8bit"    # same results as:
$ hf-fm search "AWQ 8-bit"   # or:
$ hf-fm search "AWQ int8"
```

## Comma-separated multi-term filtering

Separate terms with commas to filter progressively. The first term is sent to the HuggingFace API; all terms are used to filter results client-side. Only models whose ID contains **all** terms are shown.

```
$ hf-fm search mistral,3B,12
Models matching "mistral,3B,12" (by downloads):

  hf-fm mistralai/Ministral-3-3B-Instruct-2512           (159,700 downloads)
  hf-fm mistralai/Ministral-3-3B-Instruct-2512-BF16      (62,600 downloads)
  hf-fm mistralai/Ministral-3-3B-Base-2512               (51,500 downloads)
  ...
```

When commas are present, slashes are also treated as separators. So `mistralai/3B,12` splits into three terms: `mistralai`, `3B`, `12`.

## Library and pipeline filtering

Use `--library` and `--pipeline` to filter results server-side by framework or task:

```
$ hf-fm search llama --library peft
Models matching "llama" (by downloads):

  hf-fm lora-library/llama-2-7b-chat-lora              (1,234 downloads)
  ...

$ hf-fm search mistral --pipeline text-generation --limit 5
Models matching "mistral" (by downloads):

  hf-fm mistralai/Mistral-7B-Instruct-v0.3             (2,345,678 downloads)
  ...
```

These filters are sent to the HuggingFace API and reduce results before they reach the client. They can be combined with comma-separated multi-term filtering and `--exact`.

## Exact match (`--exact`)

Use `--exact` to match a single model by its full ID. When found, the model card metadata is fetched and displayed:

```
$ hf-fm search mistralai/Ministral-3-3B-Instruct-2512 --exact
Exact match:

  hf-fm mistralai/Ministral-3-3B-Instruct-2512           (159,700 downloads)

  License:      apache-2.0
  Library:      vllm
  Tags:         vllm, safetensors, mistral3, ...
  Languages:    en, fr, es, de, it, pt, nl, zh, ja, ko, ar
```

For gated models, the gating mode is shown:

```
$ hf-fm search google/gemma-2-2b-it --exact
Exact match:

  hf-fm google/gemma-2-2b-it                             (430,400 downloads)

  License:      gemma
  Gated:        manual (requires accepting terms on HF)
  Pipeline:     text-generation
  Library:      transformers
  ...
```

### Did you mean?

When `--exact` doesn't find a match, it suggests similar models from the search results:

```
$ hf-fm search mistralai/Ministral-3-3B-Instruct --exact
No exact match for "mistralai/Ministral-3-3B-Instruct".

Did you mean:

  hf-fm mistralai/Ministral-3-3B-Instruct-2512           (159,700 downloads)
  hf-fm mistralai/Ministral-3-3B-Instruct-2512-BF16      (62,600 downloads)
  hf-fm mistralai/Ministral-3-3B-Instruct-2512-GGUF      (32,700 downloads)
  ...
```

## Model card metadata

The `--exact` flag fetches the following from the HuggingFace model card API:

| Field | Description |
|-------|-------------|
| License | SPDX identifier (e.g., `apache-2.0`, `gemma`, `llama3.1`) |
| Gated | Access control: `auto` (accept terms) or `manual` (author approval) |
| Pipeline | Task type (e.g., `text-generation`) |
| Library | Framework (e.g., `transformers`, `vllm`) |
| Tags | All tags from the model card |
| Languages | Supported languages |

Fields are only shown when present in the model card.

## Library API

The search functionality is also available as a library:

```rust
use hf_fetch_model::discover;

// Search
let results = discover::search_models("llama 3", 20, None, None).await?;

// Search with library/pipeline filters
let peft_results = discover::search_models("llama", 20, Some("peft"), None).await?;
let gen_results = discover::search_models("mistral", 10, None, Some("text-generation")).await?;

// Fetch model card
let card = discover::fetch_model_card("meta-llama/Llama-3.2-1B-Instruct").await?;
println!("License: {:?}", card.license);
println!("Gated: {}", card.gated); // GateStatus: Open, Auto, or Manual

if card.gated.is_gated() {
    println!("This model requires accepting terms before download.");
}
```
