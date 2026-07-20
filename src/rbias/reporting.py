from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def load_json_results(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Result file not found: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError(f"Expected a JSON object: {path}")
        results.append(result)
    return results


def build_markdown_report(results: list[dict[str, Any]]) -> str:
    lines = [
        "# R-BIAS evaluation results",
        "",
        (
            "R-BIAS reports race-conditioned semantic disparity. Statistical significance "
            "or a larger score does not by itself establish harmful discrimination."
        ),
        "",
        "| Model | Encoder | Complete scenarios | Score_strict | p-value |",
        "|---|---|---:|---:|---:|",
    ]
    for result in results:
        model = result.get("model_name") or result.get("model") or "MISSING"
        encoder = result.get("encoder") or "MISSING"
        complete = result.get("n_scenarios") or result.get("n_tuples_complete") or "MISSING"
        score = result.get("score_strict", result.get("observed_score"))
        p_value = result.get("p_value")
        score_text = "MISSING" if score is None else f"{float(score):.6f}"
        p_text = "MISSING" if p_value is None else f"{float(p_value):.6f}"
        lines.append(f"| {model} | {encoder} | {complete} | {score_text} | {p_text} |")
    lines.extend(
        [
            "",
            "Interpretation requires inspection of pairwise distances, decoding settings, "
            "model versions, auxiliary encoders, missing responses, and the intended use context.",
            "",
        ]
    )
    return "\n".join(lines)
