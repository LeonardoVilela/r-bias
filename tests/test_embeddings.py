from __future__ import annotations

from pathlib import Path

import numpy as np

from rbias.dataset import load_scenarios
from rbias.embeddings import load_complete_strict_embeddings


def test_missing_embedding_excludes_the_tuple(tmp_path: Path) -> None:
    scenarios = load_scenarios(
        Path(__file__).parents[1] / "data" / "samples" / "synthetic_scenarios.json"
    )
    try:
        load_complete_strict_embeddings(scenarios, tmp_path, n_samples=1)
    except ValueError as error:
        assert "No complete strict tuples" in str(error)
    else:
        raise AssertionError("Expected incomplete data to be rejected")


def test_complete_synthetic_embedding_tuple_loads(tmp_path: Path) -> None:
    scenarios = load_scenarios(
        Path(__file__).parents[1] / "data" / "samples" / "synthetic_scenarios.json"
    )
    groups = ["negro-preto", "branco", "pardo", "indigena", "asiatico-amarelo"]
    for index, group in enumerate(groups):
        np.save(tmp_path / f"synthetic-strict-001_{group}_0.npy", np.array([index, 1.0]))
    batch = load_complete_strict_embeddings(scenarios, tmp_path, n_samples=1)
    assert batch.values.shape == (1, 5, 1, 2)
    assert batch.scenario_ids == ("synthetic-strict-001",)
