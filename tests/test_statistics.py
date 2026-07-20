from __future__ import annotations

import numpy as np

from rbias.statistics import permutation_test


def test_permutation_test_is_deterministic_with_fixed_seed() -> None:
    embeddings = np.random.default_rng(3).normal(size=(4, 5, 3, 6))
    first = permutation_test(embeddings, n_permutations=40, seed=42)
    second = permutation_test(embeddings, n_permutations=40, seed=42)
    np.testing.assert_array_equal(first.null_distribution, second.null_distribution)
    assert first.observed_score == second.observed_score
    assert first.p_value == second.p_value


def test_strong_group_separation_exceeds_permutation_null() -> None:
    embeddings = np.zeros((10, 5, 5, 4))
    for group in range(5):
        embeddings[:, group] = group * 10.0
    result = permutation_test(embeddings, n_permutations=100, seed=7)
    assert result.p_value < 0.05

