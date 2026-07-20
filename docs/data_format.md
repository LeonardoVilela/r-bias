# Data format

## Canonical scenarios

`data/scenarios/rbias_scenarios.json` is a JSON list. The retained source fields are:

| Field | Meaning |
|---|---|
| `scenario_id` | Stable scenario identifier |
| `categoria` | One of eight source categories |
| `dimensao_vies` | Source bias-dimension annotation |
| `lote` | Construction batch metadata, when present |
| `versoes` | Exactly five racial-group prompt variants |
| `grupo_racial_alvo` | Historical source/cache group label |
| `frase` | Brazilian Portuguese prompt text |

The JSON Schema is `data/schemas/scenario.schema.json`. Regime is inferred from category; it is not invented as a new field in the canonical source file.

## Flat prompt view

`data/prompts/rbias_prompts.csv` is derived from the grouped JSON by `rbias validate --export-prompts`. Its columns are:

```text
scenario_id, category, regime, racial_group, source_racial_group, prompt_text
```

`racial_group` uses the canonical five-category spelling. `source_racial_group` retains the cache-compatible label. Do not edit this CSV independently; regenerate it after an authorized source change.

## Response records

One JSON file represents one generation. The canonical fields are:

| Field | Type | Meaning |
|---|---|---|
| `scenario_id` | string | Scenario identifier |
| `prompt_id` | string | Scenario and group condition identifier |
| `category` | string | Bias-eliciting category |
| `regime` | enum | `strict` or `implicit_signaling` |
| `racial_group` | enum | Canonical IBGE category |
| `source_racial_group` | enum | Source/cache compatibility label |
| `prompt_text` | string | Exact submitted prompt |
| `response_text` | string | Returned text, empty only on error |
| `sample_index` | integer | Independent generation index, zero based |
| `model_name` | string | Requested model alias or identifier |
| `model_version` | string or null | Provider-returned version when available |
| `provider` | string | Provider/backend identifier |
| `temperature` | number | Requested decoding temperature |
| `top_p` | number | Requested nucleus-sampling value |
| `max_tokens` | integer | Requested output-token limit |
| `timestamp` | date-time | UTC generation time |
| `status` | enum | `success` or `error` |
| `error_message` | string or null | Failure details when status is `error` |

The schema is `data/schemas/response_record.schema.json`. Legacy local caches use Portuguese names for category, group, and prompt and infer `sample_index` from the filename; the public loaders support those fields for analysis without modifying the original files.

## Embeddings

Each successful response maps to one finite one-dimensional NumPy vector. The conventional filename is:

```text
{scenario_id}_{source-group}_{sample_index}.npy
```

The directory identifies auxiliary encoder and generative model. All vectors in a metric calculation must have the same dimension and normalization. Embeddings are generated artifacts and are ignored by Git.

## Aggregate score JSON

Strict output records model and encoder identifiers, overall score, scenario standard deviation, per-scenario values, pairwise matrix, within-group variance, analytic floor, complete/excluded scenarios, group order, dimensions, sample count, computation timestamp, and missing-data policy.

Permutation output records the observed score, empirical p-value, null distribution, null mean, interval, iteration count, seed, timestamp, and exclusions. Implicit output records successful and failed counts, per-group refusal and sentiment distributions, word-count summaries, and the classifier identifier when available.

Raw responses, embeddings, logs, and provider caches are not public sample data. Verify redistribution terms before sharing them.

