# R-BIAS benchmark card

## Benchmark summary

R-BIAS is a Brazilian Portuguese benchmark and evaluation framework for measuring race-conditioned semantic disparity in language-model outputs. It contains controlled prompt tuples spanning the five racial categories used by the Brazilian census and separates explicit racial attribution from implicit racial signaling.

The benchmark measures whether model responses differ semantically across prompt variants. It does not automatically determine whether a response is normatively biased, discriminatory, or harmful. Those questions require qualitative analysis and a deployment-specific theory of harm.

## Intended use

R-BIAS is intended for language-model evaluation, Responsible AI research, safeguard analysis, robustness measurement, replication, and methodological research on counterfactual generation. It can support comparisons across model versions when all prompts, decoding settings, auxiliary encoders, and missing-data rules are held constant.

## Out-of-scope use

The benchmark is not a race classifier, a diagnostic of individuals, a universal fairness ranking, or a substitute for domain-specific impact assessment. It must not be used for racial profiling, protected-attribute inference, discriminatory targeting, stereotype generation, or biological essentialism. See `RESPONSIBLE_USE.md`.

## Brazilian racial context and terminology

R-BIAS uses `branco`, `preto`, `pardo`, `amarelo`, and `indígena`. These are social and census categories situated in Brazilian history. They are not biological types and do not map cleanly to racial taxonomies used in other countries.

The source file preserves historical experiment labels `negro/preto` and `asiatico/amarelo`, mapped respectively to `preto` and `amarelo`. This compatibility layer preserves cache traceability; it does not expand the five-category taxonomy.

The Black--White binary is insufficient here because it would omit `pardo`, `amarelo`, and `indígena` comparisons and could hide disparities that do not lie primarily on a `preto`--`branco` axis.

## Dataset composition

The canonical source JSON has been structurally verified to contain:

| Component | Count |
|---|---:|
| Base scenarios | 263 |
| Racial-group variants per scenario | 5 |
| Prompt variants | 1,315 |
| Bias-eliciting categories | 8 |
| Strict counterfactual scenarios | 203 |
| Implicit signaling scenarios | 60 |

The six strict categories are occupational association, counterfactual pairs, health and care, sentiment and perception, decision making, and violence and criminality. The two implicit categories are merit and record and implicit stereotype.

## Scenario construction

The retained construction script documents an LLM-assisted drafting process using the model alias `claude-opus-4-7` and a structured prompt requiring five Brazilian racial-group variants. Scenario identifiers were added during generation, and the grouped JSON was used by the response pipeline.

The current repository verifies JSON structure, unique scenario identifiers, known categories, exactly five required groups, non-empty prompts, and strict lexical symmetry after replacing documented gender and number forms of the racial marker. All 203 current strict scenarios pass this automated check.

TODO before public release: document who reviewed scenario content, review criteria, reviewer demographics or expertise where appropriate, adjudication, revisions, and any inter-reviewer agreement. A locally verifiable human-validation record was not found, so this card does not claim completed human validation.

## Strict counterfactual regime

Strict tuples explicitly state racial identity. Their five prompts are required to be identical apart from the racial marker and necessary morphological agreement. Five independent generations are collected per prompt with temperature 0.7, top-p 1.0, and a maximum of 512 output tokens.

Responses are mapped to sentence representations using independently trained auxiliary encoders. For each scenario and group, the mean response representation is its racial-group centroid. The scenario statistic is the mean squared Euclidean distance of the five group centroids from their global scenario centroid. `Score_strict` averages that statistic across complete scenarios.

## Implicit signaling regime

Implicit variants avoid direct racial labels and vary culturally racialized names or contextual references. These signals concern perceived racial coding in Brazilian usage. Names do not determine a real person's race, and the benchmark must not be used to classify individuals.

Because the prompt text itself differs, the strict centroid-variance interpretation does not transfer unchanged. The current pipeline reports refusal rates, sentiment distributions, and response lengths by signaled group. Future decision differential and stereotype-alignment analyses are documented as planned analyses, not completed features.

## Metric definitions

For scenario \(i\), group \(g\), and generation \(k\), let \(e_{i,g,k}\) be the auxiliary-encoder response vector. The group centroid is \(\mu_{i,g}=N^{-1}\sum_k e_{i,g,k}\), the global scenario centroid is \(\bar{\mu}_i=5^{-1}\sum_g\mu_{i,g}\), and the strict scenario statistic is

\[
\sigma_i^2 = \frac{1}{5}\sum_g \lVert\mu_{i,g}-\bar{\mu}_i\rVert_2^2.
\]

The strict R-BIAS score is the mean of \(\sigma_i^2\). Larger values mean larger semantic disparity in the selected auxiliary representation space.

The pairwise matrix contains mean Euclidean distances \(D_{g,h}=K^{-1}\sum_i\lVert\mu_{i,g}-\mu_{i,h}\rVert_2\). It is symmetric with a zero diagonal.

Within-group stochastic variance measures squared response distance from each racial-group centroid before averaging. An analytic floor converts this sample-scale variance to centroid scale. Separately, the permutation test shuffles racial labels within each scenario over 10,000 iterations and reports a one-sided empirical p-value and null interval.

Auxiliary encoders are not assumed to be neutral. They are independently trained encoders used to evaluate robustness, with identifiers, pooling, prefixes, and normalization recorded in configuration. Results may change with the encoder.

## Validation and quality checks

The repository includes synthetic tests for strict tuple validation, required groups, duplicate IDs, centroids, zero and positive disparity, pairwise symmetry, within-group variance, deterministic permutations, failed generations, and configuration validation. The full dataset validator checks the expected counts and exact counterfactual structure.

Production result validation, qualitative output review, sentiment-model validation, and any human agreement study require separate author records. They are not inferred from aggregate score files.

## Known limitations and risks

- Constructed prompts cannot represent every Brazilian context or form of racism.
- The five census categories are internally heterogeneous and context dependent.
- Semantic distance can reflect style, verbosity, refusals, or encoder geometry rather than harmful treatment.
- The implicit regime combines contextual lexical variation with race-conditioned behavior.
- Name and context signals can be regionally unstable and can reinforce stereotypes if presented without care.
- Provider-side model updates can change outputs under an unchanged alias.
- Rankings can change across model versions, decoding parameters, missing-data rules, and auxiliary encoders.
- A statistically significant disparity is not equivalent to a large, harmful, or normatively unacceptable effect.

Benchmark rankings must not be interpreted as a universal measure of model fairness.

## Human-subject considerations

The canonical R-BIAS scenarios are constructed prompts rather than records about identified people. No direct human-subject study or private annotation dataset is included in this release candidate. TODO: the authors must confirm whether any scenario review or output annotation constituted human-subject research and document the applicable ethics or consent process before release.

Third-party datasets used in earlier intrinsic analyses are not included here and retain their original governance and licensing conditions.

## Model-output and data licensing

Original code and benchmark materials use the split described in `LICENSE`, `DATA_LICENSE`, and `NOTICE`. Provider model outputs are not automatically relicensed and should not be distributed until provider terms, privacy, harmful content, and research consent are reviewed. Model weights and private caches are excluded.

## Maintenance policy

Changes to prompts, group mappings, regime assignments, formulas, pooling, normalization, or missing-data rules must be versioned and documented because they can affect numerical results. Provider aliases should not be treated as immutable versions. Maintainer identities, security contact, release cadence, and deprecation policy are TODO items for the repository owner.

## Citation

If you use R-BIAS, please cite the accompanying paper and this software release. Complete the placeholders in `CITATION.cff` before publication; no DOI or venue is asserted in this release candidate.

