# R-BIAS

R-BIAS is a Brazilian Portuguese benchmark and evaluation framework for measuring race-conditioned disparities in language-model outputs. It evaluates both explicit racial attribution and implicit racial signaling across the five racial categories used by the Brazilian census.

R-BIAS measures semantic disparity. A larger score means that an auxiliary encoder places responses to race-conditioned variants farther apart on average. It does not, by itself, establish normative bias, discriminatory intent, or harmful discrimination. Those conclusions require contextual and qualitative analysis in addition to statistical evidence.

## Motivation and Brazilian context

Most counterfactual fairness benchmarks reduce race to a Black--White comparison. That binary omits central features of Brazilian racial classification and social experience. R-BIAS therefore works with the five IBGE categories: `branco`, `preto`, `pardo`, `amarelo`, and `indígena`. The `pardo` category cannot be treated as a residual midpoint, and the `indígena` and `amarelo` categories are not interchangeable with categories used in United States datasets.

The source dataset retains two historical cache labels, `negro/preto` and `asiatico/amarelo`. They map to the IBGE categories `preto` and `amarelo`; they do not create additional groups. The mapping is documented in `data/README.md` and preserved so previous scenario identifiers and cache filenames remain traceable.

## Benchmark composition

The checked-in benchmark contains:

- 263 base scenarios;
- five racial-group variants per scenario;
- 1,315 prompt variants in total;
- eight bias-eliciting categories;
- 203 scenarios in the strict counterfactual regime;
- 60 scenarios in the implicit signaling regime.

In the strict counterfactual regime, race is stated explicitly. The five prompts in a tuple must be lexically identical apart from the racial marker and necessary gender or number agreement. This regime supports the primary R-BIAS metric.

In the implicit signaling regime, race is not stated directly. Prompt variants use culturally racialized names or contextual references. These features represent perceived racial coding in Brazilian usage; they do not determine the actual race of a real person and must not be used for individual racial classification. The implicit regime is analyzed separately because its lexical variation is part of the intervention.

## Strict R-BIAS metric

For scenario \(i\), racial group \(g\), and independent generation \(k\), let \(e_{i,g,k}\) be the response representation from one auxiliary encoder. With \(N\) generations, the racial-group centroid is

\[
\mu_{i,g} = \frac{1}{N}\sum_{k=1}^{N} e_{i,g,k}.
\]

The global scenario centroid is \(\bar{\mu}_i = \frac{1}{5}\sum_g \mu_{i,g}\). The scenario disparity is

\[
\sigma_i^2 = \frac{1}{5}\sum_g \lVert \mu_{i,g} - \bar{\mu}_i \rVert_2^2,
\]

and `Score_strict` is the mean of \(\sigma_i^2\) over complete strict scenarios. Larger values indicate greater race-conditioned semantic disparity in the auxiliary representation space. Scores are only directly comparable when encoder, normalization, sampling parameters, and missing-data rules match.

The pairwise decomposition reports the mean Euclidean distance between every pair of racial-group centroids as a symmetric 5 by 5 matrix. It shows which comparisons contribute to an aggregate score without reducing the Brazilian setting to a single axis.

The stochastic baseline estimates variation caused by independent sampling within the same prompt condition. The implementation reports raw within-group variance and an analytic centroid-scale floor. The 10,000-iteration permutation test independently shuffles racial labels among the 25 response vectors in each scenario and compares the observed score with the resulting equality-null distribution. A small empirical p-value is evidence against exchangeability under that test; it is not a universal fairness judgment.

R-BIAS uses independently trained auxiliary encoders to evaluate the robustness of observed disparities. They must not be described as neutral. The configured encoders are documented in `configs/encoders.yaml`, including pooling, prefixes, and normalization.

BERTimbau and PTT5 are listed separately as intrinsic WEAT/SEAT/SBS evaluation targets. They do not join the auxiliary response-encoder set unless a later documented analysis explicitly approves that role.

## Repository structure

```text
configs/     Experiment, model, and encoder definitions
data/        Canonical scenarios, derived prompts, schemas, and synthetic samples
docs/        Methodology, formats, reproduction, and model-integration guidance
examples/    Minimal synthetic evaluation
notebooks/   Notebook inventory and release policy
src/rbias/   Canonical implementation
tests/       Synthetic unit and integration checks
```

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

Install provider and encoder dependencies only when needed:

```bash
python -m pip install -e .[api,encoders]
```

Local open-weight generation through vLLM is a separate GPU-oriented installation:

```bash
python -m pip install -e .[local]
```

No credential file is read by the public implementation. Copy `.env.example` to a local, ignored `.env` only if your environment tooling requires it, or export variables directly. The pipeline itself reads provider credentials from environment variables.

## Validate the benchmark

```bash
rbias validate --config configs/experiment.yaml
```

To regenerate the auditable flat prompt view:

```bash
rbias validate \
  --config configs/experiment.yaml \
  --export-prompts data/prompts/rbias_prompts.csv \
  --overwrite
```

## Minimal reproducible example

The example uses invented arrays and makes no network calls:

```bash
python examples/minimal_evaluation.py
```

Equivalent Python:

```python
import numpy as np

from rbias.metrics import strict_rbias_metrics
from rbias.statistics import permutation_test

embeddings = np.random.default_rng(7).normal(size=(3, 5, 5, 8))
metrics = strict_rbias_metrics(embeddings)
inference = permutation_test(embeddings, n_permutations=100, seed=42)

print(metrics["score_strict"])
print(inference.p_value)
```

## Main commands

Every command accepts an experiment configuration, prints its resolved values, and requires an explicit output directory when it writes experiment artifacts.

```text
rbias validate
rbias generate
rbias embed
rbias evaluate
rbias permutation-test
rbias analyze-implicit
rbias report
```

Generation and embedding support `--dry-run`. Existing successful response caches are reused. An existing corrupt, failed, or conflicting output is not replaced unless `--overwrite` is explicit.

A typical sequence is:

```bash
rbias generate --config configs/experiment.yaml --model MODEL --output-dir outputs --dry-run
rbias generate --config configs/experiment.yaml --model MODEL --output-dir outputs
rbias embed --config configs/experiment.yaml --model MODEL --encoder bge \
  --input-dir outputs/responses/MODEL --output-dir outputs
rbias evaluate --config configs/experiment.yaml --model MODEL --encoder bge \
  --input-dir outputs/embeddings/bge/MODEL --output-dir outputs
rbias permutation-test --config configs/experiment.yaml --model MODEL --encoder bge \
  --input-dir outputs/embeddings/bge/MODEL --output-dir outputs
rbias analyze-implicit --config configs/experiment.yaml --model MODEL \
  --input-dir outputs/responses/MODEL --output-dir outputs
```

See `docs/reproduction.md` for cache layout, resumption, version metadata, and GPU-specific notes.

## Data format

The canonical grouped scenario JSON uses the existing Portuguese field names `scenario_id`, `categoria`, `dimensao_vies`, `versoes`, `grupo_racial_alvo`, and `frase`. The derived CSV and new response schema use explicit English field names. A successful response record includes scenario, prompt, category, regime, racial group, generation index, provider, exact returned model version, decoding parameters, timestamp, status, and response text.

Failed and empty generations are recorded with status and error metadata. They are excluded from semantic and implicit-analysis denominators and counted explicitly. Strict evaluation requires all five groups and all five requested generations; incomplete tuples are listed in `excluded_scenarios`.

Full schemas and compatibility notes are in `docs/data_format.md`.

## Reproducing experiments and adding models

Reproduction requires the exact scenario file, model alias and provider-returned version, timestamp, sampling settings, response cache, auxiliary encoder identifier, pooling and normalization settings, and permutation seed. Follow `docs/reproduction.md`; do not regenerate an existing response cache merely to rebuild metrics.

To add a model, add one entry to `configs/models.yaml`, set only the documented environment variable, run a dry run, then complete a small smoke test before a full 6,575-response run. Instructions and provider constraints are in `docs/adding_a_model.md`.

## Limitations

R-BIAS is specific to Brazilian Portuguese prompts and a bounded set of constructed scenarios. It does not measure all forms of racism, fairness, safety, factuality, or downstream harm. Semantic distance can arise from style, verbosity, refusal behavior, lexical choices, or encoder geometry. Prompt sensitivity, decoding randomness, provider-side changes, and model-version changes can affect results.

The implicit signaling regime cannot isolate race as cleanly as the strict regime because the contextual signal itself changes. Names and contextual references are historically and socially contingent, can be perceived differently across regions and communities, and never identify a real person's race.

Auxiliary encoders have their own limitations. Cross-encoder agreement is a robustness check, not proof that a measurement is free of representation bias. Benchmark rankings must not be interpreted as universal rankings of model fairness. See `BENCHMARK_CARD.md` for further limitations and unresolved validation items.

## Ethical and responsible use

Use results to investigate controlled race-conditioned differences, not to classify people or justify discriminatory decisions. Review aggregate findings together with pairwise, qualitative, refusal, sentiment, and missing-data analyses. Do not infer harmful discrimination from a score without a documented normative argument and domain context.

`RESPONSIBLE_USE.md` prohibits or strongly discourages racial profiling, stereotype generation, discriminatory targeting, protected-attribute ranking, and biological essentialism. Model outputs and third-party datasets may have separate terms and redistribution limits.

## Citation

If you use R-BIAS, please cite the accompanying paper and this software release. `CITATION.cff` intentionally contains placeholders for authors and publication metadata that must be completed before release. No DOI, venue, repository URL, or publication status is claimed here.

## Licenses

- Original code under `src/` and `tests/`: Apache License 2.0 (`LICENSE`).
- Original benchmark scenarios, prompts, schemas, and documentation under `data/` and `docs/`: CC BY 4.0 (`DATA_LICENSE`), subject to author review before release.
- Third-party dependencies and datasets: their original licenses.
- Model outputs: applicable provider terms; they are not automatically relicensed by this repository.

`NOTICE` defines the path-level mapping and unresolved licensing checks.

## Project status

This is a local release candidate. The benchmark structure and core strict metric are implemented and covered by synthetic tests. Human content-validation documentation, author metadata, citation details, and third-party data licensing remain `TODO` items. Nothing in this repository should be treated as publicly released until the owner explicitly authorizes publication.

AI-assisted tools were used only to support repository organization and documentation editing. The research design, benchmark construction, implementation decisions, experiments, analyses, and conclusions are the responsibility of the authors.
