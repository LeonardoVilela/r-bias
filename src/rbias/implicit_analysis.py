from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .dataset import CANONICAL_GROUP_BY_SOURCE, CANONICAL_RACIAL_GROUPS


def is_refusal(text: str, lexicon: tuple[str, ...] | list[str]) -> bool:
    lowered = text.lower()
    return any(phrase.lower() in lowered for phrase in lexicon)


def _canonical_group(record: dict[str, Any]) -> str:
    group = record.get("racial_group")
    if group in CANONICAL_RACIAL_GROUPS:
        return str(group)
    source = record.get("source_racial_group") or record.get("grupo_racial_alvo")
    if source in CANONICAL_GROUP_BY_SOURCE:
        return CANONICAL_GROUP_BY_SOURCE[str(source)]
    raise ValueError(f"Unknown racial group: {group or source!r}")


def load_implicit_responses(response_dir: str | Path) -> tuple[list[dict[str, Any]], int]:
    directory = Path(response_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Response directory not found: {directory}")
    records: list[dict[str, Any]] = []
    failed = 0
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid response JSON: {path}") from exc
        category = record.get("category") or record.get("categoria")
        regime = record.get("regime")
        if regime != "implicit_signaling" and category not in {
            "historico_e_merito",
            "estereotipo_implicito",
        }:
            continue
        text = record.get("response_text")
        status = record.get("status", "success" if text else "error")
        if status not in {"success", "ok"} or not isinstance(text, str) or not text.strip():
            failed += 1
            continue
        records.append(
            {
                "scenario_id": record.get("scenario_id"),
                "category": category,
                "racial_group": _canonical_group(record),
                "response_text": text,
            }
        )
    return records, failed


def classify_sentiment(
    texts: list[str], primary_model: str, fallback_model: str
) -> tuple[list[str | None], str | None]:
    import torch
    from transformers import pipeline

    classifier = None
    loaded_model = None
    for model_id in (primary_model, fallback_model):
        try:
            classifier = pipeline(
                "sentiment-analysis",
                model=model_id,
                truncation=True,
                max_length=512,
                device=0 if torch.cuda.is_available() else -1,
            )
            loaded_model = model_id
            break
        except Exception:
            continue
    if classifier is None:
        return [None] * len(texts), None
    raw = classifier(texts, truncation=True, max_length=512, batch_size=16)
    labels: list[str | None] = []
    for result in raw:
        label = str(result.get("label", "")).lower()
        if "pos" in label:
            labels.append("positive")
        elif "neg" in label:
            labels.append("negative")
        else:
            labels.append("neutral")
    return labels, loaded_model


def analyze_implicit_responses(
    records: list[dict[str, Any]],
    refusal_lexicon: tuple[str, ...] | list[str],
    sentiment_labels: list[str | None] | None = None,
    failed_or_empty: int = 0,
) -> dict[str, Any]:
    """Aggregate refusal, sentiment, and word-count distributions by group.

    Failed and empty generations are excluded from metric denominators and
    reported separately. Sentiment labels must align one-to-one with successful
    response records; when omitted, sentiment is recorded as null.
    """
    if sentiment_labels is None:
        sentiment_labels = [None] * len(records)
    if len(sentiment_labels) != len(records):
        raise ValueError("sentiment_labels must align with records")

    grouped: dict[str, list[tuple[dict[str, Any], str | None]]] = defaultdict(list)
    for record, sentiment in zip(records, sentiment_labels):
        grouped[str(record["racial_group"])].append((record, sentiment))

    result: dict[str, Any] = {}
    for group in CANONICAL_RACIAL_GROUPS:
        rows = grouped.get(group, [])
        texts = [str(record["response_text"]) for record, _ in rows]
        refusal_count = sum(is_refusal(text, refusal_lexicon) for text in texts)
        sentiments = Counter(sentiment for _, sentiment in rows)
        n = len(rows)
        word_counts = [len(text.split()) for text in texts]
        result[group] = {
            "n_responses": n,
            "refusals": refusal_count,
            "refusal_rate": refusal_count / n if n else None,
            "sentiment": {
                "negative": sentiments["negative"],
                "neutral": sentiments["neutral"],
                "positive": sentiments["positive"],
                "null": sentiments[None],
            },
            "sentiment_distribution": {
                label: sentiments[label if label != "null" else None] / n if n else None
                for label in ("negative", "neutral", "positive", "null")
            },
            "word_count": {
                "mean": sum(word_counts) / n if n else None,
                "min": min(word_counts) if n else None,
                "max": max(word_counts) if n else None,
            },
        }
    return {
        "aggregate_by_group": result,
        "successful_responses": len(records),
        "failed_or_empty_responses": failed_or_empty,
    }
