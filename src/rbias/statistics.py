from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import strict_rbias_metrics


@dataclass(frozen=True)
class PermutationResult:
    observed_score: float
    p_value: float
    null_distribution: np.ndarray
    seed: int

    @property
    def null_mean(self) -> float:
        return float(self.null_distribution.mean())

    @property
    def null_ci_95(self) -> tuple[float, float]:
        low, high = np.percentile(self.null_distribution, [2.5, 97.5])
        return float(low), float(high)


def permutation_test(
    response_embeddings: np.ndarray,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> PermutationResult:
    """Shuffle racial labels within each scenario and return an empirical p-value.

    Input shape is ``(scenarios, groups, samples, dimensions)``. Each iteration
    independently shuffles all group-by-sample response vectors within every
    scenario, restores equally sized groups, and recomputes the mean
    between-centroid variance. The one-sided empirical p-value is the fraction
    of permuted scores greater than or equal to the observed score, matching the
    existing R-BIAS implementation. Failed or incomplete tuples must be removed
    before calling this function.
    """
    embeddings = np.asarray(response_embeddings, dtype=np.float64)
    if embeddings.ndim != 4 or 0 in embeddings.shape or not np.isfinite(embeddings).all():
        raise ValueError("response_embeddings must be a finite non-empty (K, G, N, D) array")
    if not isinstance(n_permutations, int) or n_permutations <= 0:
        raise ValueError("n_permutations must be a positive integer")

    observed = strict_rbias_metrics(embeddings)["score_strict"]
    scenarios, groups, samples, dimensions = embeddings.shape
    combined = embeddings.reshape(scenarios, groups * samples, dimensions)
    base_indices = np.tile(np.arange(groups * samples), (scenarios, 1))
    scenario_indices = np.arange(scenarios)[:, None]
    null_distribution = np.empty(n_permutations, dtype=np.float64)
    rng = np.random.default_rng(seed)

    for index in range(n_permutations):
        shuffled_indices = rng.permuted(base_indices, axis=1)
        shuffled = combined[scenario_indices, shuffled_indices].reshape(
            scenarios, groups, samples, dimensions
        )
        centroids = shuffled.mean(axis=2)
        global_centroids = centroids.mean(axis=1, keepdims=True)
        per_scenario = np.square(centroids - global_centroids).sum(axis=2).mean(axis=1)
        null_distribution[index] = per_scenario.mean()

    p_value = float((null_distribution >= observed).mean())
    return PermutationResult(observed, p_value, null_distribution, seed)
