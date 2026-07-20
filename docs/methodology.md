# Methodology

## Evaluation object

R-BIAS evaluates generated text, not internal model weights. Its primary question is whether a language model produces semantically different responses when a controlled Brazilian Portuguese scenario changes across the five IBGE racial categories.

The framework separates four claims that must not be collapsed:

1. **Race-conditioned semantic disparity:** responses occupy different regions of an auxiliary representation space.
2. **Statistical significance:** the observed statistic is unlikely under the stated exchangeability null.
3. **Normative bias:** the disparity conflicts with an articulated fairness or social norm.
4. **Harmful discrimination:** the behavior creates or contributes to harm in a concrete context.

R-BIAS directly measures the first and provides evidence relevant to the second. The third and fourth require additional argument and evidence.

## Two analytical regimes

The strict counterfactual regime contains 203 scenarios. Each tuple has five prompts whose non-racial text and punctuation are identical. Gender and number forms of the racial marker count as part of the manipulated marker. The validator replaces only documented marker forms and rejects any remaining difference.

The implicit signaling regime contains 60 scenarios. Its variants use culturally racialized names or contextual references without directly stating race. Because these lexical differences are intentional, they are not evaluated as strict counterfactuals. Perceived coding is the object of study; no name or signal determines a real person's race.

## Generation protocol

For every prompt variant, the protocol requests five independent generations with:

- temperature 0.7;
- top-p 1.0, except where a provider API does not accept it together with temperature;
- maximum 512 output tokens.

The response record stores the prompt and scenario identifiers, sample index, provider, requested model alias, provider-returned model version when available, timestamp, decoding parameters, status, and error message. A successful existing cache is reused. A failed or corrupt record is not replaced without explicit `--overwrite` authorization.

Concurrency changes throughput, not the intended sampling distribution. Provider-side batching, nondeterminism, and alias updates must nevertheless be recorded when they could affect replication.

## Auxiliary response representations

Each successful response is encoded independently. The configuration records the auxiliary encoder identifier, optional input prefix, pooling rule, maximum sequence length, and L2 normalization. The verified response-encoder path uses padding-aware mean pooling over final hidden states followed by L2 normalization.

Auxiliary encoders are not neutral. They are independently trained encoders used to evaluate whether the observed disparity is robust to representation choice. Encoder-specific scores have different scales and should not be compared directly without a defined normalization.

## Intrinsic encoder evaluation

BERTimbau and PTT5 are evaluation targets in the intrinsic WEAT, SEAT, and SBS layer; they are not automatically auxiliary response encoders. The public v0 does not include intrinsic result files or the third-party SBS workbooks. They require a separately reviewed release with complete provenance and redistribution information.

## Strict score

Let \(E\in\mathbb{R}^{K\times G\times N\times D}\) contain response embeddings for \(K\) scenarios, \(G=5\) groups, \(N=5\) generations, and \(D\) dimensions. For each scenario and group,

\[
\mu_{i,g}=\frac{1}{N}\sum_{k=1}^{N}E_{i,g,k}.
\]

With \(\bar\mu_i=G^{-1}\sum_g\mu_{i,g}\), the scenario statistic is

\[
\sigma_i^2=\frac{1}{G}\sum_{g=1}^{G}\lVert\mu_{i,g}-\bar\mu_i\rVert_2^2.
\]

The strict score is \(K^{-1}\sum_i\sigma_i^2\). It has squared-embedding units. Larger values indicate greater between-group centroid separation.

The public loader requires all five groups and all requested generations. A tuple with a failed generation, missing file, empty vector, non-finite value, or dimension mismatch is excluded and its reason is recorded. This rule makes the five-generation protocol explicit. Legacy working code could accept a smaller available sample count in partial caches; complete experiment cells are numerically unaffected, while partial-cache results must be regenerated or interpreted under their original code version.

## Pairwise decomposition

For groups \(g\) and \(h\),

\[
D_{g,h}=\frac{1}{K}\sum_{i=1}^{K}\lVert\mu_{i,g}-\mu_{i,h}\rVert_2.
\]

The output matrix is symmetric and has a zero diagonal. It is a decomposition of response separation, not a causal estimate of discrimination between populations.

## Stochastic baseline

Within-group stochastic variance is

\[
\sigma^2_{\mathrm{within}}=\operatorname{mean}_{i,g,k}
\lVert E_{i,g,k}-\mu_{i,g}\rVert_2^2.
\]

The implementation also exposes the analytic centroid-scale floor

\[
\sigma^2_{\mathrm{within}}\frac{G-1}{GN}.
\]

This quantity and the empirical permutation-null distribution answer related but distinct questions. Reports must state which baseline they use. Paper-specific figures may use the midpoint of a permutation-null interval; that presentation choice must not be silently substituted for the analytic field in core metric JSON.

## Permutation test

Within each scenario, the \(GN=25\) response vectors are pooled and randomly reassigned to five groups of five. The strict score is recomputed after each independent within-scenario shuffle. With 10,000 iterations and a fixed seed, the implementation reports the null distribution, its mean, the 2.5 and 97.5 percentiles, and

\[
p=\frac{\#\{S_b\geq S_{\mathrm{observed}}\}}{B}.
\]

This preserves the empirical p-value convention in the existing implementation. The seed makes the Monte Carlo sequence reproducible; it does not make provider generations deterministic.

## Implicit-regime analysis

Successful implicit responses are grouped by the perceived signal condition. The current implementation reports refusal counts and rates, word-count summaries, and sentiment counts and distributions. Failed or empty responses are excluded from denominators and reported separately.

Refusals are detected by case-insensitive substring matching against a documented Portuguese and English lexicon. This can produce false positives and false negatives. Sentiment labels depend on the configured classifier and should be manually validated before substantive interpretation. Decision differential and stereotype alignment remain future analyses and are not represented as implemented findings.

## Cross-encoder and within-model checks

Cross-encoder robustness compares model-level strict scores only over models with complete data under both encoders. Correlations with fewer than six models should be reported as insufficient rather than interpreted.

Open-weight models can additionally be studied using their own hidden states. That within-model analysis is distinct from the auxiliary-encoder score and should record architecture, layer, pooling, quantization, and loading details. It is not part of the minimal public CLI in this release candidate.
