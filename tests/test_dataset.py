from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from rbias.dataset import load_scenarios, summarize_scenarios, validate_scenarios


def synthetic_scenarios() -> list[dict]:
    path = Path(__file__).parents[1] / "data" / "samples" / "synthetic_scenarios.json"
    return load_scenarios(path)


def issue_codes(scenarios: list[dict]) -> set[str]:
    return {issue.code for issue in validate_scenarios(scenarios)}


def test_valid_strict_counterfactual_tuple_and_expected_variants() -> None:
    scenarios = synthetic_scenarios()
    assert validate_scenarios(scenarios) == []
    assert summarize_scenarios(scenarios) == {
        "scenarios": 2,
        "prompt_variants": 10,
        "strict_scenarios": 1,
        "implicit_scenarios": 1,
        "categories": 2,
    }


def test_counterfactual_wording_change_is_rejected() -> None:
    scenarios = synthetic_scenarios()
    scenarios[0]["versoes"][1]["frase"] += " Texto não contrafactual."
    assert "strict_counterfactual_mismatch" in issue_codes(scenarios)


def test_missing_racial_group_is_reported() -> None:
    scenarios = synthetic_scenarios()
    scenarios[0]["versoes"][4]["grupo_racial_alvo"] = "branco"
    assert "racial_group_set" in issue_codes(scenarios)


def test_wrong_variant_count_is_reported() -> None:
    scenarios = synthetic_scenarios()
    scenarios[0]["versoes"].pop()
    assert "variant_count" in issue_codes(scenarios)


def test_duplicate_scenario_id_is_reported() -> None:
    scenarios = synthetic_scenarios()
    duplicate = deepcopy(scenarios[0])
    scenarios.append(duplicate)
    assert "duplicate_scenario_id" in issue_codes(scenarios)

