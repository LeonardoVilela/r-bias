#!/usr/bin/env python3
"""Compare analytic centroid-null floors with empirical permutation means."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

from _common import OUTPUTS_DIR


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_table() -> str:
    rows = []
    for strict_path in sorted((OUTPUTS_DIR / "scores").glob("*_strict.json")):
        strict = _read(strict_path)
        analytic = strict.get("null_floor_analytic")
        if analytic is None:
            continue
        model, encoder = strict["model"], strict["encoder"]
        safe = model.replace("/", "_").replace(":", "_")
        perm_path = OUTPUTS_DIR / "scores" / f"{safe}_{encoder}_permutation.json"
        if not perm_path.exists():
            rows.append((model, encoder, analytic, None, strict["score_strict"], strict["signal_to_noise_ratio"], "MISSING PERM"))
            continue
        permutation = _read(perm_path)
        empirical = permutation.get("null_distribution_mean")
        if empirical is None:
            empirical = fmean(permutation["null_distribution"])
        relative_gap = abs(analytic - empirical) / empirical if empirical else float("inf")
        status = "OK" if relative_gap <= 0.30 else f"REVIEW ({relative_gap:.1%} gap)"
        rows.append((model, encoder, analytic, empirical, strict["score_strict"], strict["signal_to_noise_ratio"], status))

    lines = [
        "# Analytic vs empirical null floor",
        "",
        "| Model | Encoder | Analytic floor | Permutation null mean | Observed score | Ratio | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for model, encoder, analytic, empirical, observed, ratio, status in rows:
        empirical_text = "—" if empirical is None else f"{empirical:.6f}"
        lines.append(
            f"| {model} | {encoder} | {analytic:.6f} | {empirical_text} | "
            f"{observed:.6f} | {ratio:.3f} | {status} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    table = build_table()
    output = OUTPUTS_DIR / "reports" / "null_floor_sanity.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table, encoding="utf-8")
    print(table)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
