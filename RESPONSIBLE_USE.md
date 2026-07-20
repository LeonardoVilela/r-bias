# Responsible use

R-BIAS is intended for controlled research on race-conditioned semantic disparity in Brazilian Portuguese language-model outputs. Appropriate uses include model evaluation, safeguard analysis, robustness studies, replication, measurement research, and carefully scoped audits in which affected communities and deployment context are considered.

The benchmark must not be used to infer or classify the race of real people. Names, neighborhoods, cultural references, language varieties, appearance descriptions, and other signals do not determine a person's race. The implicit regime studies perceived racial coding and model behavior under that coding.

Do not use R-BIAS to:

- build racial-profiling or surveillance systems;
- infer protected characteristics of individuals;
- optimize discriminatory advertising, credit, employment, insurance, policing, health, education, housing, or political targeting;
- generate, expand, or operationalize racial stereotypes;
- rank individuals by race or other protected characteristics;
- justify biological essentialism from social or census categories;
- conceal discriminatory decision rules behind an aggregate benchmark score;
- claim that a model, provider, or organization is discriminatory solely from a semantic-distance result.

A responsible analysis reports model version, timestamp, decoding parameters, scenario coverage, missing responses, auxiliary encoders, pooling and normalization, permutation settings, pairwise results, and material qualitative limitations. Statistical significance should be reported separately from effect magnitude. Normative or harm claims require an explicit argument and relevant domain evidence.

Benchmark rankings are not universal measures of model fairness. Results can change across model versions, providers, decoding settings, prompt revisions, and auxiliary encoders. If R-BIAS is used in a consequential audit, involve domain experts and people familiar with Brazilian racial relations, document disagreements, and avoid releasing private or harmful model outputs without review.

This document is responsible-use guidance. It does not add restrictions to the Apache-2.0 or CC BY 4.0 licenses.

