from __future__ import annotations

from pathlib import Path

import pytest

from rbias.config import load_experiment_config


def test_configuration_paths_and_protocol_are_valid() -> None:
    path = Path(__file__).parents[1] / "examples" / "sample_config.yaml"
    config = load_experiment_config(path)
    assert config.scenarios_path.is_absolute()
    assert config.generation["n_samples"] == 5
    assert config.generation["temperature"] == 0.7
    assert config.generation["top_p"] == 1.0
    assert config.generation["max_tokens"] == 512
    assert config.permutation["n_permutations"] == 100


def test_invalid_top_p_fails_clearly(tmp_path: Path) -> None:
    source = (Path(__file__).parents[1] / "examples" / "sample_config.yaml").read_text(
        encoding="utf-8"
    )
    path = tmp_path / "invalid.yaml"
    path.write_text(source.replace("top_p: 1.0", "top_p: 0.0"), encoding="utf-8")
    with pytest.raises(ValueError, match="top_p"):
        load_experiment_config(path)

