# Adding a language model

## 1. Identify the exact interface

Determine whether the model uses Anthropic, OpenAI-compatible, Google, or local vLLM generation. Confirm the provider's accepted decoding parameters. Do not assume that a marketing alias is an immutable model version; preserve the version returned by the provider when available.

## 2. Add one model entry

Add a direct entry to `configs/models.yaml`:

```yaml
models:
  example-model:
    provider: openai
    environment_variable: EXAMPLE_API_KEY
```

If an OpenAI-compatible provider requires a base URL, add the public endpoint to that entry. Never add the credential. Local models require an exact model identifier and, where needed, tensor-parallel and quantization settings that have been verified on the target hardware.

If a new provider API is necessary, add the smallest backend implementation in `src/rbias/generation.py`. It must return response text and an exact model-version string or a documented null value, translate accepted sampling parameters explicitly, and remain import-lazy so unrelated providers do not become mandatory dependencies.

## 3. Preflight without calls

```bash
rbias generate \
  --config configs/experiment.yaml \
  --model example-model \
  --output-dir outputs/example-model-evaluation \
  --dry-run
```

Review the resolved scenario file, call count, model entry, and sampling settings. A full run requires 6,575 successful records.

## 4. Run a separately authorized smoke test

Use a temporary experiment copy or a dedicated testing output directory. Do not mix smoke-test records with the full cache. Check that records contain non-empty text, sample index, provider, returned version, timestamp, and requested parameters. Confirm that rerunning the same command reuses successful records.

## 5. Complete the encoder and analysis stages

Run every approved auxiliary encoder and the same strict, permutation, and implicit analyses used for the comparison set. Cross-encoder correlations must use only models with complete data under both encoders. Do not place a model in a paper ranking until cache completeness, version metadata, and result licensing have been checked.

## 6. Document limitations

Record provider parameter constraints, safety-filter behavior, regional availability, local quantization, context limits, rate limits, and any failed or excluded scenarios. Do not make unsupported claims about discrimination from an aggregate score.

