# Reproducing an R-BIAS evaluation

## 1. Create the environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .[api,encoders]
```

For development checks, also install `.[dev]`. For local open-weight generation on a compatible Linux GPU host, install `.[local]` in an environment whose CUDA, PyTorch, and vLLM versions match the hardware.

## 2. Configure without storing credentials

Start from `configs/experiment.yaml`, `configs/models.yaml`, and `configs/encoders.yaml`. Keep model aliases, encoder identifiers, pooling, normalization, and generation parameters with the experiment record.

Set only the environment variable required by the chosen provider. `.env.example` contains names without values. Do not place credentials in YAML, scripts, notebooks, logs, or Git.

## 3. Validate scenarios

```bash
rbias validate --config configs/experiment.yaml
```

The expected result is 263 scenarios, 1,315 variants, 203 strict scenarios, 60 implicit scenarios, eight categories, unique IDs, five required groups, and no strict lexical mismatches.

## 4. Preflight generation

```bash
rbias generate \
  --config configs/experiment.yaml \
  --model MODEL \
  --output-dir outputs \
  --dry-run
```

The dry run resolves the configuration and counts cached and pending records without constructing a provider client or making an API call. For a new provider or model, use an isolated temporary output directory for a small authorized smoke test before the full benchmark.

## 5. Generate and resume safely

```bash
rbias generate --config configs/experiment.yaml --model MODEL --output-dir outputs
```

The expected full workload is 263 scenarios times five group variants times five generations, or 6,575 response records per model. Successful existing records are reused. A rerun with a complete cache issues no duplicate calls. Corrupt or failed records stop preflight unless `--overwrite` is passed deliberately.

Do not use `--overwrite` for ordinary resumption. Inspect the exact target files, preserve the originals, and record the reason before replacing a failed record. Provider aliases can change over time, so retain the provider-returned model version and timestamp from every response.

## 6. Extract embeddings

```bash
rbias embed \
  --config configs/experiment.yaml \
  --model MODEL \
  --encoder bge \
  --input-dir outputs/responses/MODEL \
  --output-dir outputs \
  --dry-run

rbias embed \
  --config configs/experiment.yaml \
  --model MODEL \
  --encoder bge \
  --input-dir outputs/responses/MODEL \
  --output-dir outputs
```

Repeat for each authorized auxiliary encoder. The output is `outputs/embeddings/{encoder}/{model}/`. Existing vectors are reused unless overwrite is explicit. Record the exact encoder revision from the local model cache or environment lock when strict revision pinning is required.

## 7. Calculate strict metrics

```bash
rbias evaluate \
  --config configs/experiment.yaml \
  --model MODEL \
  --encoder bge \
  --input-dir outputs/embeddings/bge/MODEL \
  --output-dir outputs
```

The result is `outputs/scores/{model}_{encoder}_strict.json`. It includes exclusions and the exact group order. Complete tuples require all 25 vectors. Do not compare a partial-cache score with a complete-cache score.

## 8. Run the permutation test

```bash
rbias permutation-test \
  --config configs/experiment.yaml \
  --model MODEL \
  --encoder bge \
  --input-dir outputs/embeddings/bge/MODEL \
  --output-dir outputs
```

The default is 10,000 iterations with seed 42. The null distribution is retained so intervals, means, and diagnostic plots can be audited without rerunning the test.

## 9. Analyze the implicit regime

```bash
rbias analyze-implicit \
  --config configs/experiment.yaml \
  --model MODEL \
  --input-dir outputs/responses/MODEL \
  --output-dir outputs
```

Use `--without-sentiment` when the configured classifier is unavailable; the output then records null sentiment labels rather than inventing values. Refusal detection uses the configured lexicon. Validate classifier and lexicon behavior qualitatively before interpretation.

## 10. Build reports

```bash
rbias report \
  --config configs/experiment.yaml \
  --scores outputs/scores/MODEL_bge_strict.json \
  --output-dir outputs
```

The generic report is intentionally factual and reads only explicitly supplied score files.

## Expected output layout

```text
outputs/
  responses/{model}/*.json
  embeddings/{encoder}/{model}/*.npy
  scores/{model}_{encoder}_strict.json
  scores/{model}_{encoder}_permutation.json
  scores/{model}_implicit.json
  reports/summary.md
```

`outputs/` is ignored by Git. Back it up according to provider terms and institutional data policy. Do not copy private responses into public issue reports.

## Reproducibility checklist

Record the repository commit after authorization, Python and dependency versions, hardware, scenario-file hash, model alias and returned version, provider, timestamps, five decoding parameters, response count, failure count, encoder identifier and revision, pooling, normalization, embedding dimension, complete/excluded scenarios, permutation count and seed, sentiment classifier, and report script inputs.

Interrupted generation is resumed by rerunning the same command with the same output directory. Do not change model alias, prompt file, or sampling parameters within one cache. If any of them changes, use a new explicitly named experiment directory.
