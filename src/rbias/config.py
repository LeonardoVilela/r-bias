from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    config_path: Path
    name: str
    scenarios_path: Path
    models_path: Path
    encoders_path: Path
    expected_scenarios: int
    expected_prompt_variants: int
    expected_strict_scenarios: int
    expected_implicit_scenarios: int
    generation: dict[str, Any]
    permutation: dict[str, Any]
    sentiment: dict[str, Any]
    refusal_lexicon: tuple[str, ...]

    def printable(self) -> dict[str, Any]:
        values = asdict(self)
        for key in ("config_path", "scenarios_path", "models_path", "encoders_path"):
            values[key] = str(values[key])
        values["refusal_lexicon"] = list(self.refusal_lexicon)
        return values


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return loaded


def _positive_int(section: dict[str, Any], name: str) -> int:
    value = section.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    raw = _read_yaml(config_path)
    experiment = raw.get("experiment")
    generation = raw.get("generation")
    permutation = raw.get("permutation")
    sentiment = raw.get("sentiment")
    if not isinstance(experiment, dict):
        raise ValueError("Missing mapping: experiment")
    if not isinstance(generation, dict):
        raise ValueError("Missing mapping: generation")
    if not isinstance(permutation, dict):
        raise ValueError("Missing mapping: permutation")
    if not isinstance(sentiment, dict):
        raise ValueError("Missing mapping: sentiment")

    temperature = generation.get("temperature")
    top_p = generation.get("top_p")
    if not isinstance(temperature, (int, float)) or temperature < 0:
        raise ValueError("generation.temperature must be non-negative")
    if not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
        raise ValueError("generation.top_p must be in (0, 1]")
    for name in ("n_samples", "max_tokens", "concurrency", "max_retries"):
        _positive_int(generation, name)
    _positive_int(permutation, "n_permutations")
    if not isinstance(permutation.get("seed"), int):
        raise ValueError("permutation.seed must be an integer")
    for name in ("primary", "fallback"):
        if not isinstance(sentiment.get(name), str) or not sentiment[name].strip():
            raise ValueError(f"sentiment.{name} must be a model identifier")

    base = config_path.parent

    def resolved(name: str) -> Path:
        value = experiment.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"experiment.{name} must be a path")
        return (base / value).resolve()

    lexicon = raw.get("refusal_lexicon")
    if not isinstance(lexicon, list) or not all(isinstance(item, str) and item for item in lexicon):
        raise ValueError("refusal_lexicon must be a non-empty list of strings")

    return ExperimentConfig(
        config_path=config_path,
        name=str(experiment.get("name") or "rbias"),
        scenarios_path=resolved("scenarios"),
        models_path=resolved("models"),
        encoders_path=resolved("encoders"),
        expected_scenarios=_positive_int(experiment, "expected_scenarios"),
        expected_prompt_variants=_positive_int(experiment, "expected_prompt_variants"),
        expected_strict_scenarios=_positive_int(experiment, "expected_strict_scenarios"),
        expected_implicit_scenarios=_positive_int(experiment, "expected_implicit_scenarios"),
        generation=dict(generation),
        permutation=dict(permutation),
        sentiment=dict(sentiment),
        refusal_lexicon=tuple(item.lower() for item in lexicon),
    )


def load_registry(path: str | Path, section: str) -> dict[str, dict[str, Any]]:
    raw = _read_yaml(Path(path))
    registry = raw.get(section)
    if not isinstance(registry, dict) or not registry:
        raise ValueError(f"Missing non-empty mapping {section!r} in {path}")
    if not all(isinstance(value, dict) for value in registry.values()):
        raise ValueError(f"Every entry in {section!r} must be a mapping")
    return registry


def format_resolved_config(config: ExperimentConfig) -> str:
    return yaml.safe_dump(config.printable(), allow_unicode=True, sort_keys=False)
