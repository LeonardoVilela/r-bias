from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SOURCE_RACIAL_GROUPS = (
    "negro/preto",
    "branco",
    "pardo",
    "indigena",
    "asiatico/amarelo",
)
CANONICAL_RACIAL_GROUPS = ("preto", "branco", "pardo", "indígena", "amarelo")
CANONICAL_GROUP_BY_SOURCE = {
    "negro/preto": "preto",
    "branco": "branco",
    "pardo": "pardo",
    "indigena": "indígena",
    "asiatico/amarelo": "amarelo",
}
SOURCE_GROUP_BY_CANONICAL = {value: key for key, value in CANONICAL_GROUP_BY_SOURCE.items()}

STRICT_CATEGORIES = frozenset(
    {
        "associacao_ocupacao",
        "contrafactual_pares",
        "saude_e_cuidado",
        "sentimento_e_percepcao",
        "tomada_de_decisao",
        "violencia_e_criminalidade",
    }
)
IMPLICIT_CATEGORIES = frozenset({"historico_e_merito", "estereotipo_implicito"})
ALL_CATEGORIES = STRICT_CATEGORIES | IMPLICIT_CATEGORIES

_RACE_MARKERS = sorted(
    {
        "asiatico/amarelo",
        "asiatica/amarela",
        "asiaticos/amarelos",
        "asiaticas/amarelas",
        "negro/preto",
        "negra/preta",
        "negros/pretos",
        "negras/pretas",
        "indigena",
        "indigenas",
        "asiatico",
        "asiatica",
        "asiaticos",
        "asiaticas",
        "amarelo",
        "amarela",
        "amarelos",
        "amarelas",
        "negro",
        "negra",
        "negros",
        "negras",
        "preto",
        "preta",
        "pretos",
        "pretas",
        "branco",
        "branca",
        "brancos",
        "brancas",
        "pardo",
        "parda",
        "pardos",
        "pardas",
    },
    key=len,
    reverse=True,
)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    scenario_id: str | None = None


def load_scenarios(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Scenario file not found: {source}")
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Scenario dataset must be a JSON list")
    return data


def regime_for_category(category: str) -> str:
    if category in STRICT_CATEGORIES:
        return "strict"
    if category in IMPLICIT_CATEGORIES:
        return "implicit_signaling"
    raise ValueError(f"Unknown category: {category!r}")


def _ascii_lower(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    without_accents = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return without_accents.lower()


def normalize_counterfactual_text(text: str) -> str:
    """Replace known racial-marker surface forms while preserving all other tokens.

    Case, accents, gender, and number variations in the racial marker are treated
    as necessary surface agreement. Punctuation and non-racial wording remain part
    of the comparison.
    """
    normalized = _ascii_lower(text)
    for marker in _RACE_MARKERS:
        normalized = re.sub(rf"(?<!\w){re.escape(marker)}(?!\w)", "<race>", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def validate_scenarios(scenarios: list[dict[str, Any]]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()
    expected_groups = set(SOURCE_RACIAL_GROUPS)

    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            issues.append(ValidationIssue("invalid_scenario", f"Item {index} is not an object"))
            continue
        scenario_id = scenario.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            issues.append(
                ValidationIssue("missing_scenario_id", f"Item {index} has no scenario_id")
            )
            scenario_id = None
        elif scenario_id in seen_ids:
            issues.append(
                ValidationIssue("duplicate_scenario_id", "Duplicate scenario_id", scenario_id)
            )
        else:
            seen_ids.add(scenario_id)

        category = scenario.get("categoria")
        if category not in ALL_CATEGORIES:
            issues.append(
                ValidationIssue("unknown_category", f"Unknown category: {category!r}", scenario_id)
            )

        variants = scenario.get("versoes")
        if not isinstance(variants, list) or len(variants) != 5:
            count = len(variants) if isinstance(variants, list) else 0
            issues.append(
                ValidationIssue(
                    "variant_count", f"Expected 5 variants, found {count}", scenario_id
                )
            )
            continue

        groups = [
            variant.get("grupo_racial_alvo")
            for variant in variants
            if isinstance(variant, dict)
        ]
        if len(groups) != 5 or set(groups) != expected_groups:
            missing = sorted(expected_groups - set(groups))
            extra = sorted(set(groups) - expected_groups, key=str)
            issues.append(
                ValidationIssue(
                    "racial_group_set",
                    f"Invalid racial groups; missing={missing}, extra={extra}",
                    scenario_id,
                )
            )

        prompts = [variant.get("frase") for variant in variants if isinstance(variant, dict)]
        invalid_prompt = any(
            not isinstance(prompt, str) or not prompt.strip() for prompt in prompts
        )
        if len(prompts) != 5 or invalid_prompt:
            issues.append(
                ValidationIssue("empty_prompt", "Every variant needs prompt text", scenario_id)
            )
            continue

        if category in STRICT_CATEGORIES:
            normalized = {normalize_counterfactual_text(prompt) for prompt in prompts}
            if len(normalized) != 1:
                issues.append(
                    ValidationIssue(
                        "strict_counterfactual_mismatch",
                        "Non-racial wording differs across the strict tuple",
                        scenario_id,
                    )
                )
    return issues


def prompt_records(scenarios: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for scenario in scenarios:
        category = scenario["categoria"]
        for variant in scenario["versoes"]:
            source_group = variant["grupo_racial_alvo"]
            yield {
                "scenario_id": scenario["scenario_id"],
                "category": category,
                "regime": regime_for_category(category),
                "racial_group": CANONICAL_GROUP_BY_SOURCE[source_group],
                "source_racial_group": source_group,
                "prompt_text": variant["frase"],
            }


def export_prompts_csv(
    scenarios: list[dict[str, Any]], path: str | Path, overwrite: bool = False
) -> int:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = list(prompt_records(scenarios))
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def summarize_scenarios(scenarios: list[dict[str, Any]]) -> dict[str, int]:
    strict = sum(scenario.get("categoria") in STRICT_CATEGORIES for scenario in scenarios)
    implicit = sum(scenario.get("categoria") in IMPLICIT_CATEGORIES for scenario in scenarios)
    return {
        "scenarios": len(scenarios),
        "prompt_variants": sum(len(scenario.get("versoes", [])) for scenario in scenarios),
        "strict_scenarios": strict,
        "implicit_scenarios": implicit,
        "categories": len({scenario.get("categoria") for scenario in scenarios}),
    }
