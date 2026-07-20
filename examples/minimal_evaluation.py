from __future__ import annotations

import numpy as np

from rbias.metrics import strict_rbias_metrics
from rbias.statistics import permutation_test


def main() -> None:
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(3, 5, 5, 8))
    metrics = strict_rbias_metrics(embeddings)
    inference = permutation_test(embeddings, n_permutations=100, seed=42)
    print(f"Score_strict: {metrics['score_strict']:.6f}")
    print(f"Permutation p-value: {inference.p_value:.4f}")


if __name__ == "__main__":
    main()
