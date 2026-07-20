from __future__ import annotations

import numpy as np
import pytest

from rbias.metrics import (
    between_group_variance,
    pairwise_racial_group_distance,
    racial_group_centroids,
    strict_rbias_metrics,
    within_group_variance,
)


def test_centroid_calculation_is_manually_verifiable() -> None:
    samples = np.array([[[0.0], [2.0]], [[2.0], [4.0]]])
    np.testing.assert_allclose(racial_group_centroids(samples), [[1.0], [3.0]])


def test_zero_disparity_for_identical_group_responses() -> None:
    embeddings = np.ones((3, 5, 4, 2))
    assert strict_rbias_metrics(embeddings)["score_strict"] == pytest.approx(0.0)


def test_positive_disparity_for_separated_groups() -> None:
    embeddings = np.zeros((2, 5, 3, 1))
    for group in range(5):
        embeddings[:, group, :, 0] = group
    assert strict_rbias_metrics(embeddings)["score_strict"] == pytest.approx(2.0)
    assert between_group_variance(np.arange(5, dtype=float).reshape(5, 1)) == pytest.approx(2.0)


def test_pairwise_matrix_is_symmetric_with_zero_diagonal() -> None:
    centroids = np.array([[[0.0], [1.0], [2.0], [3.0], [4.0]]])
    matrix = pairwise_racial_group_distance(centroids)
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(np.diag(matrix), 0.0)
    assert matrix[0, 4] == pytest.approx(4.0)


def test_within_group_variance() -> None:
    embeddings = np.array([[[[0.0], [2.0]], [[3.0], [3.0]]]])
    assert within_group_variance(embeddings) == pytest.approx(0.5)


def test_invalid_or_nonfinite_vectors_fail_clearly() -> None:
    with pytest.raises(ValueError, match="finite"):
        strict_rbias_metrics(np.array([[[[np.nan]]]]))

