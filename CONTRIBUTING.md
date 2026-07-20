# Contributing

R-BIAS is research software. Contributions should make the scientific assumptions easier to inspect, reproduce, or test without silently changing the benchmark definition.

Before editing a metric, dataset mapping, prompt, category, racial-group label, pooling rule, missing-data rule, or statistical procedure, open a discussion with the maintainers and state whether existing numerical outputs could change. Formula changes require a documented methodological rationale, synthetic tests with manually verifiable values, and a changelog entry.

For code changes:

1. Create an isolated Python environment.
2. Install `python -m pip install -e .[dev]` and any feature extras required by the change.
3. Add or update focused synthetic tests.
4. Run `pytest`, `ruff check .`, `ruff format --check .`, and `mypy src/rbias`.
5. Run `rbias validate --config configs/experiment.yaml` when data loading is affected.
6. Describe data, model, provider, or output-licensing implications.

Never include credentials, private response caches, model weights, personal paths, private URLs, or non-redistributable data. Use invented examples in tests. Do not regenerate cached responses as part of a code review.

Documentation should distinguish race-conditioned semantic disparity, statistical significance, normative bias, and harmful discrimination. Do not describe auxiliary encoders as neutral or present benchmark ranks as universal fairness rankings.

Author identity, contributor agreement policy, and release workflow are TODO items for the repository owner.

