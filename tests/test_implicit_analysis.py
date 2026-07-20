from __future__ import annotations

import json
from pathlib import Path

from rbias.implicit_analysis import analyze_implicit_responses, load_implicit_responses


def write_record(path: Path, group: str, response: str, status: str = "success") -> None:
    path.write_text(
        json.dumps(
            {
                "scenario_id": "synthetic-implicit-001",
                "category": "estereotipo_implicito",
                "regime": "implicit_signaling",
                "racial_group": group,
                "response_text": response,
                "status": status,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_failed_and_empty_generations_are_excluded_and_counted(tmp_path: Path) -> None:
    write_record(tmp_path / "ok.json", "branco", "Resposta válida.")
    write_record(tmp_path / "empty.json", "pardo", "")
    write_record(tmp_path / "error.json", "preto", "", status="error")
    records, failed = load_implicit_responses(tmp_path)
    assert len(records) == 1
    assert failed == 2
    result = analyze_implicit_responses(records, ["não posso"], failed_or_empty=failed)
    assert result["successful_responses"] == 1
    assert result["failed_or_empty_responses"] == 2


def test_refusal_denominator_uses_successful_responses_only() -> None:
    records = [
        {"racial_group": "branco", "response_text": "Não posso responder."},
        {"racial_group": "branco", "response_text": "Resposta válida."},
    ]
    result = analyze_implicit_responses(records, ["não posso"])
    assert result["aggregate_by_group"]["branco"]["refusal_rate"] == 0.5

