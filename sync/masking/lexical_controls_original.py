#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import tqdm

from _common import (
    OUTPUTS_DIR,
    RACIAL_GROUPS,
    STRICT_CATS,
    atomic_write_json,
    atomic_write_npy,
    is_valid_npy,
    load_config,
    load_dataset,
    sanitize_group,
    setup_logging,
)
from compute_embeddings import Encoder
from compute_scores import _analytic_null_floor
from masking_lexicon import find_markers, replace_markers, synthetic_variants
from permutation_test import _run_permutations, _observed_score


MODELS = (
    "sabia-4",
    "claude-haiku-4-5",
    "gpt-4.1",
    "gemini-3.1-flash-lite-preview",
)
VARIANTS = ("mask", "deletion")
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings_masked"
SCORES_DIR = OUTPUTS_DIR / "scores_masked"
REPORT_DIR = OUTPUTS_DIR / "reports" / "lexical_controls"
VERDICT_PATH = OUTPUTS_DIR / "reports" / "lexical_controls_verdict.md"
logger = setup_logging("lexical_controls")


def safe_model(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def discover_complete_models() -> tuple[str, ...]:
    models = []
    for response_dir in sorted((OUTPUTS_DIR / "responses").iterdir()):
        if not response_dir.is_dir():
            continue
        paths = sorted(response_dir.glob("*.json"))
        if len(paths) < 5075:
            print(f"SKIP partial cache: {response_dir.name} ({len(paths)} JSON files)")
            continue
        record = json.loads(paths[0].read_text(encoding="utf-8"))
        models.append(record.get("model") or response_dir.name)
    if not models:
        raise RuntimeError("No complete response cache found in outputs/responses")
    return tuple(models)


def embedding_path(variant: str, encoder: str, model: str, scenario: str, group: str, sample: int) -> Path:
    return (
        EMBEDDINGS_DIR
        / variant
        / encoder
        / safe_model(model)
        / f"{scenario}_{sanitize_group(group)}_{sample}.npy"
    )


def load_rows(models: tuple[str, ...]) -> list[dict[str, Any]]:
    dataset = {row["scenario_id"]: row for row in load_dataset() if row["categoria"] in STRICT_CATS}
    rows: list[dict[str, Any]] = []
    for model in models:
        response_dir = OUTPUTS_DIR / "responses" / safe_model(model)
        for path in sorted(response_dir.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            scenario = record["scenario_id"]
            if scenario not in dataset:
                continue
            rows.append({
                "model": model,
                "scenario_id": scenario,
                "category": dataset[scenario]["categoria"],
                "group": record["grupo_racial_alvo"],
                "sample": int(path.stem.rsplit("_", 1)[-1]),
                "text": record["response_text"],
                "source": path.relative_to(OUTPUTS_DIR.parent).as_posix(),
            })
    expected = len(models) * 203 * 5 * 5
    if len(rows) != expected:
        raise RuntimeError(f"STRICT response cache has {len(rows)}/{expected} records")
    return rows


def save_csv(name: str, rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(REPORT_DIR / name, index=False, encoding="utf-8")
    return frame


def run_echo_analysis(
    rows: list[dict[str, Any]], seed: int, models: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tuple_echo: dict[tuple[str, str], bool] = defaultdict(bool)
    audit: list[dict[str, Any]] = []

    for row in rows:
        matches = find_markers(row["text"])
        key = row["model"], row["group"]
        counts[key]["responses"] += 1
        counts[key]["echo_any"] += bool(matches)
        counts[key]["echo_prompted"] += any(match.group == row["group"] for match in matches)
        tuple_echo[(row["model"], row["scenario_id"])] |= bool(matches)
        for match in matches:
            start = max(0, match.start - 70)
            end = min(len(row["text"]), match.end + 70)
            audit.append({
                "model": row["model"],
                "scenario_id": row["scenario_id"],
                "prompted_group": row["group"],
                "matched_group": match.group,
                "span": match.text,
                "context": row["text"][start:end].replace("\n", " "),
            })

    rate_rows = []
    for model in models:
        for group in RACIAL_GROUPS:
            values = counts[(model, group)]
            total = values["responses"]
            rate_rows.append({
                "model": model,
                "group": group,
                "responses": total,
                "echo_any_count": values["echo_any"],
                "echo_any_rate": values["echo_any"] / total,
                "echo_prompted_count": values["echo_prompted"],
                "echo_prompted_rate": values["echo_prompted"] / total,
            })
    rates = save_csv("echo_rates.csv", rate_rows)

    overall_rows = []
    for model in models:
        model_rows = rates[rates["model"] == model]
        total = int(model_rows["responses"].sum())
        any_count = int(model_rows["echo_any_count"].sum())
        prompted_count = int(model_rows["echo_prompted_count"].sum())
        overall_rows.append({
            "model": model,
            "responses": total,
            "echo_any_count": any_count,
            "echo_any_rate": any_count / total,
            "echo_prompted_count": prompted_count,
            "echo_prompted_rate": prompted_count / total,
            "below_15_percent": any_count / total < 0.15,
        })
    overall = save_csv("echo_overall.csv", overall_rows)

    split_rows = []
    for model in models:
        score = json.loads((OUTPUTS_DIR / "scores" / f"{model}_bge_strict.json").read_text(encoding="utf-8"))
        variances = score["per_tuple_variances"]
        for split, present in (("echo-present", True), ("echo-free", False)):
            values = [
                float(value)
                for scenario, value in variances.items()
                if tuple_echo[(model, scenario)] == present
            ]
            split_rows.append({
                "model": model,
                "split": split,
                "n_tuples": len(values),
                "mean_tuple_variance": float(np.mean(values)) if values else np.nan,
            })
    splits = save_csv("echo_tuple_split.csv", split_rows)

    rng = random.Random(seed)
    sample = rng.sample(audit, min(200, len(audit)))
    lines = [f"Total detected spans: {len(audit)}", f"Random audit sample: {len(sample)}", ""]
    for index, item in enumerate(sample, 1):
        lines.extend([
            f"[{index}] {item['model']} | {item['scenario_id']} | prompted={item['prompted_group']} | matched={item['matched_group']}",
            f"span: {item['span']}",
            f"context: {item['context']}",
            "",
        ])
    (OUTPUTS_DIR / "reports" / "masking_audit.txt").write_text("\n".join(lines), encoding="utf-8")

    atomic_write_json(REPORT_DIR / "echo_summary.json", {
        "overall": overall.to_dict("records"),
        "by_group": rates.to_dict("records"),
        "tuple_split": splits.to_dict("records"),
        "detected_spans": len(audit),
        "audit_sample_size": len(sample),
    })
    print("\nECHO RATES BY MODEL AND GROUP")
    print(rates.to_string(index=False, formatters={"echo_any_rate": "{:.2%}".format, "echo_prompted_rate": "{:.2%}".format}))
    print("\nECHO RATES OVERALL")
    print(overall.to_string(index=False, formatters={"echo_any_rate": "{:.2%}".format, "echo_prompted_rate": "{:.2%}".format}))
    print("\nECHO-PRESENT VS ECHO-FREE TUPLES")
    print(splits.to_string(index=False))
    return overall, splits


def encode_safely(encoder: Encoder, texts: list[str]) -> np.ndarray:
    try:
        return encoder.encode(texts)
    except torch.cuda.OutOfMemoryError:
        if len(texts) == 1:
            raise
        torch.cuda.empty_cache()
        middle = len(texts) // 2
        return np.concatenate((encode_safely(encoder, texts[:middle]), encode_safely(encoder, texts[middle:])))


def run_embeddings(
    rows: list[dict[str, Any]],
    encoders: tuple[str, ...],
    variants: tuple[str, ...],
    batch_size: int,
    config: dict,
) -> tuple[str, ...]:
    completed: list[str] = []
    for encoder_key in encoders:
        encoder_name = config["encoders"][encoder_key]
        try:
            encoder = Encoder(encoder_key, encoder_name)
            for variant in variants:
                pending = [
                    row for row in rows
                    if not is_valid_npy(embedding_path(
                        variant, encoder_key, row["model"], row["scenario_id"], row["group"], row["sample"]
                    ))
                ]
                logger.info("variant=%s encoder=%s cached=%d pending=%d", variant, encoder_key, len(rows) - len(pending), len(pending))
                for start in tqdm.tqdm(range(0, len(pending), batch_size), desc=f"{variant}/{encoder_key}"):
                    batch = pending[start : start + batch_size]
                    replacement = "[GRUPO]" if variant == "mask" else ""
                    texts = [replace_markers(row["text"], replacement)[0] for row in batch]
                    vectors = encode_safely(encoder, texts)
                    for row, vector in zip(batch, vectors):
                        path = embedding_path(
                            variant, encoder_key, row["model"], row["scenario_id"], row["group"], row["sample"]
                        )
                        if not is_valid_npy(path):
                            atomic_write_npy(path, vector)
                atomic_write_json(EMBEDDINGS_DIR / variant / encoder_key / "manifest.json", {
                    "variant": variant,
                    "encoder": encoder_key,
                    "encoder_model": encoder.model_name,
                    "models": list(MODELS),
                    "records": len(rows),
                    "computed_at": datetime.now(timezone.utc).isoformat(),
                })
            completed.append(encoder_key)
            del encoder
            torch.cuda.empty_cache()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if encoder_key != "e5":
                raise
            logger.exception("E5 unavailable; BGE results remain valid: %s", exc)
            print(f"E5 SKIPPED: {exc}")
            torch.cuda.empty_cache()
    return tuple(completed)


def load_embedding_array(variant: str, encoder: str, model: str) -> tuple[np.ndarray, list[str], dict[str, str]]:
    dataset = [row for row in load_dataset() if row["categoria"] in STRICT_CATS]
    arrays = []
    scenario_ids = []
    categories = {}
    for scenario in dataset:
        sid = scenario["scenario_id"]
        groups = []
        for group in RACIAL_GROUPS:
            samples = []
            for sample in range(5):
                path = embedding_path(variant, encoder, model, sid, group, sample)
                if not is_valid_npy(path):
                    raise RuntimeError(f"Missing masked embedding: {path}")
                samples.append(np.load(path).astype(np.float32))
            groups.append(np.stack(samples))
        arrays.append(np.stack(groups))
        scenario_ids.append(sid)
        categories[sid] = scenario["categoria"]
    return np.stack(arrays), scenario_ids, categories


def score_path(variant: str, model: str, encoder: str, suffix: str) -> Path:
    return SCORES_DIR / variant / f"{safe_model(model)}_{encoder}_{suffix}.json"


def compute_score_output(
    all_embeddings: np.ndarray,
    scenario_ids: list[str],
    categories: dict[str, str],
    model: str,
    encoder: str,
    variant: str,
) -> dict[str, Any]:
    centroids = all_embeddings.mean(axis=2)
    global_centroids = centroids.mean(axis=1, keepdims=True)
    tuple_variances = ((centroids - global_centroids) ** 2).sum(axis=2).mean(axis=1)
    within = ((all_embeddings - centroids[:, :, None, :]) ** 2).sum(axis=3).mean(axis=2)
    score = float(tuple_variances.mean())
    sigma_within = float(within.mean())
    null_floor = _analytic_null_floor(sigma_within, len(RACIAL_GROUPS), 5)

    pairwise = np.zeros((5, 5), dtype=np.float64)
    for first in range(5):
        for second in range(5):
            pairwise[first, second] = np.linalg.norm(centroids[:, first] - centroids[:, second], axis=1).mean()

    category_values: dict[str, list[float]] = defaultdict(list)
    category_within: dict[str, list[float]] = defaultdict(list)
    for index, sid in enumerate(scenario_ids):
        category_values[categories[sid]].append(float(tuple_variances[index]))
        category_within[categories[sid]].extend(float(value) for value in within[index])
    per_category = {
        category: {"n_tuples": len(values), "score": float(np.mean(values)), "std": float(np.std(values))}
        for category, values in category_values.items()
    }
    per_category_baseline = {category: float(np.mean(values)) for category, values in category_within.items()}
    per_category_floor = {
        category: _analytic_null_floor(value, 5, 5)
        for category, value in per_category_baseline.items()
    }

    return {
        "model": model,
        "encoder": encoder,
        "variant": variant,
        "n_tuples_strict": len(scenario_ids),
        "n_tuples_complete": len(scenario_ids),
        "n_tuples_excluded": 0,
        "excluded_scenarios": [],
        "score_strict": score,
        "score_std": float(tuple_variances.std()),
        "sigma2_within": sigma_within,
        "baseline_within_group_variance": sigma_within,
        "null_floor_analytic": null_floor,
        "signal_to_noise_ratio": score / null_floor,
        "per_category_baseline": per_category_baseline,
        "per_category_null_floor_analytic": per_category_floor,
        "per_category_ratio": {
            category: per_category[category]["score"] / per_category_floor[category]
            for category in per_category
        },
        "per_category": per_category,
        "per_tuple_variances": dict(zip(scenario_ids, map(float, tuple_variances))),
        "pairwise_distance_matrix": {"groups": RACIAL_GROUPS, "matrix": pairwise.tolist()},
        "metadata": {
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "n_samples_per_group": 5,
            "n_groups": 5,
            "analytic_null_factor": 0.16,
            "embeddings_root": f"outputs/embeddings_masked/{variant}/{encoder}",
        },
    }


def score_job(job: tuple[str, str, str], permutations: int, seed: int, force: bool = False) -> dict[str, Any]:
    variant, encoder, model = job
    all_embeddings, scenario_ids, categories = load_embedding_array(variant, encoder, model)
    strict = compute_score_output(all_embeddings, scenario_ids, categories, model, encoder, variant)
    strict_path = score_path(variant, model, encoder, "strict")
    atomic_write_json(strict_path, strict)

    permutation_path = score_path(variant, model, encoder, "permutation")
    if permutation_path.exists() and not force:
        cached = json.loads(permutation_path.read_text(encoding="utf-8"))
        metadata = cached.get("metadata", {})
        if (
            metadata.get("n_permutations") == permutations
            and metadata.get("seed") == seed
            and np.isclose(cached.get("observed_score", -1), strict["score_strict"], rtol=1e-5)
        ):
            return {"variant": variant, "encoder": encoder, "model": model, "cached": True}

    null = _run_permutations(all_embeddings, permutations, np.random.default_rng(seed))
    low, high = np.percentile(null, (2.5, 97.5))
    atomic_write_json(permutation_path, {
        "model": model,
        "encoder": encoder,
        "variant": variant,
        "n_tuples": len(scenario_ids),
        "observed_score": _observed_score(all_embeddings),
        "p_value": float((null >= strict["score_strict"]).mean()),
        "null_distribution_mean": float(null.mean()),
        "null_distribution_std": float(null.std()),
        "null_ci_95": [float(low), float(high)],
        "null_distribution": null.tolist(),
        "metadata": {
            "n_permutations": permutations,
            "seed": seed,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        },
    })
    return {"variant": variant, "encoder": encoder, "model": model, "cached": False}


def run_scores(
    encoders: tuple[str, ...],
    variants: tuple[str, ...],
    models: tuple[str, ...],
    permutations: int,
    seed: int,
    workers: int,
    force: bool,
) -> None:
    jobs = [(variant, encoder, model) for variant in variants for encoder in encoders for model in models]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(score_job, job, permutations, seed, force): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            print("SCORE/PERMUTATION", result)


def publish_primary_mask_scores() -> None:
    source_dir = SCORES_DIR / "mask"
    for source in source_dir.glob("*.json"):
        atomic_write_json(SCORES_DIR / source.name, json.loads(source.read_text(encoding="utf-8")))


def largest_pair(score: dict[str, Any]) -> tuple[str, float, bool]:
    groups = score["pairwise_distance_matrix"]["groups"]
    matrix = np.asarray(score["pairwise_distance_matrix"]["matrix"])
    best: tuple[str, float] | None = None
    for first in range(len(groups)):
        for second in range(first + 1, len(groups)):
            value = float(matrix[first, second])
            pair = f"{groups[first]} – {groups[second]}"
            if best is None or value > best[1]:
                best = pair, value
    assert best is not None
    return best[0], best[1], "indigena" in best[0]


def build_comparisons(encoders: tuple[str, ...], variants: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparison_rows = []
    pair_rows = []
    correlation_rows = []
    for model in MODELS:
        original_bge = json.loads((OUTPUTS_DIR / "scores" / f"{model}_bge_strict.json").read_text(encoding="utf-8"))
        original_pair, original_distance, original_indigenous = largest_pair(original_bge)
        original_variances = pd.Series(original_bge["per_tuple_variances"], dtype=float)
        for variant in variants:
            for encoder in encoders:
                strict = json.loads(score_path(variant, model, encoder, "strict").read_text(encoding="utf-8"))
                permutation = json.loads(score_path(variant, model, encoder, "permutation").read_text(encoding="utf-8"))
                original = json.loads((OUTPUTS_DIR / "scores" / f"{model}_{encoder}_strict.json").read_text(encoding="utf-8"))
                low, high = permutation["null_ci_95"]
                floor = (low + high) / 2
                comparison_rows.append({
                    "variant": variant,
                    "model": model,
                    "encoder": encoder,
                    "score_original": original["score_strict"],
                    "score_masked": strict["score_strict"],
                    "delta_percent": 100 * (strict["score_strict"] - original["score_strict"]) / original["score_strict"],
                    "p_masked": permutation["p_value"],
                    "null_floor_masked": floor,
                    "ratio_masked": strict["score_strict"] / floor,
                })
                if encoder == "bge":
                    masked_pair, masked_distance, masked_indigenous = largest_pair(strict)
                    pair_rows.append({
                        "variant": variant,
                        "model": model,
                        "largest_pair_original": original_pair,
                        "distance_original": original_distance,
                        "indigenous_original": original_indigenous,
                        "largest_pair_masked": masked_pair,
                        "distance_masked": masked_distance,
                        "indigenous_after_masking": masked_indigenous,
                    })
                    masked_variances = pd.Series(strict["per_tuple_variances"], dtype=float)
                    correlation_rows.append({
                        "variant": variant,
                        "model": model,
                        "encoder": "bge",
                        "n_tuples": len(original_variances),
                        "spearman_rho": original_variances.corr(masked_variances, method="spearman"),
                    })
    comparisons = save_csv("masking_comparison.csv", comparison_rows)
    pairs = save_csv("largest_pairs.csv", pair_rows)
    correlations = save_csv("tuple_correlations.csv", correlation_rows)
    print("\nORIGINAL VS MASKED SCORES")
    print(comparisons.to_string(index=False))
    print("\nLARGEST PAIRS BEFORE AND AFTER")
    print(pairs.to_string(index=False))
    print("\nPER-TUPLE SPEARMAN CORRELATIONS")
    print(correlations.to_string(index=False))
    return comparisons, pairs, correlations


def select_synthetic_sample(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    eligible = []
    for row in rows:
        matches = find_markers(row["text"])
        if len(matches) == 1:
            eligible.append({**row, "match": matches[0]})

    selected = []
    for model in MODELS:
        model_pool = [row for row in eligible if row["model"] == model]
        used_sources: set[str] = set()
        for category in sorted(STRICT_CATS):
            pool = [row for row in model_pool if row["category"] == category]
            for row in rng.sample(pool, min(2, len(pool))):
                selected.append(row)
                used_sources.add(row["source"])
        remaining = [row for row in model_pool if row["source"] not in used_sources]
        needed = 15 - sum(row["model"] == model for row in selected)
        if len(remaining) < needed:
            raise RuntimeError(f"Only {15 - needed + len(remaining)}/15 synthetic candidates for {model}")
        selected.extend(rng.sample(remaining, needed))
    if len(selected) != 60:
        raise RuntimeError(f"Synthetic sample has {len(selected)}/60 responses")
    return selected


def run_synthetic(rows: list[dict[str, Any]], seed: int, batch_size: int, config: dict) -> dict[str, Any]:
    selected = select_synthetic_sample(rows, seed)
    tasks = []
    manifest_rows = []
    for index, row in enumerate(selected):
        synthetic_id = f"synthetic_{index:03d}"
        variants = synthetic_variants(row["text"], row["match"], RACIAL_GROUPS)
        manifest_rows.append({
            "synthetic_id": synthetic_id,
            "model": row["model"],
            "scenario_id": row["scenario_id"],
            "category": row["category"],
            "sample": row["sample"],
            "source_group": row["match"].group,
            "source_marker": row["match"].text,
            "source": row["source"],
        })
        for group, text in variants.items():
            path = EMBEDDINGS_DIR / "synthetic" / "bge" / f"{synthetic_id}_{sanitize_group(group)}.npy"
            if not is_valid_npy(path):
                tasks.append((path, text))
    save_csv("synthetic_echo_samples.csv", manifest_rows)

    if tasks:
        encoder = Encoder("bge", config["encoders"]["bge"])
        for start in tqdm.tqdm(range(0, len(tasks), batch_size), desc="synthetic/bge"):
            batch = tasks[start : start + batch_size]
            vectors = encode_safely(encoder, [text for _, text in batch])
            for (path, _), vector in zip(batch, vectors):
                if not is_valid_npy(path):
                    atomic_write_npy(path, vector)
        del encoder
        torch.cuda.empty_cache()

    sample_rows = []
    pair_rows = []
    for item in manifest_rows:
        vectors = []
        for group in RACIAL_GROUPS:
            path = EMBEDDINGS_DIR / "synthetic" / "bge" / f"{item['synthetic_id']}_{sanitize_group(group)}.npy"
            vectors.append(np.load(path).astype(np.float32))
        matrix = np.stack(vectors)
        global_centroid = matrix.mean(axis=0)
        variance = float(((matrix - global_centroid) ** 2).sum(axis=1).mean())
        sample_rows.append({**item, "synthetic_variance": variance})
        for first in range(5):
            for second in range(first + 1, 5):
                pair_rows.append({
                    "synthetic_id": item["synthetic_id"],
                    "group_1": RACIAL_GROUPS[first],
                    "group_2": RACIAL_GROUPS[second],
                    "distance": float(np.linalg.norm(matrix[first] - matrix[second])),
                    "indigenous_pair": "indigena" in (RACIAL_GROUPS[first], RACIAL_GROUPS[second]),
                })
    samples = save_csv("synthetic_echo_per_sample.csv", sample_rows)
    pairs = save_csv("synthetic_pairwise.csv", pair_rows)
    values = samples["synthetic_variance"]
    indigenous_mean = float(pairs[pairs["indigenous_pair"]]["distance"].mean())
    other_mean = float(pairs[~pairs["indigenous_pair"]]["distance"].mean())
    summary = {
        "n_responses": len(samples),
        "variance_mean": float(values.mean()),
        "variance_median": float(values.median()),
        "variance_p95": float(values.quantile(0.95)),
        "indigenous_pairwise_mean": indigenous_mean,
        "other_pairwise_mean": other_mean,
        "indigenous_minus_other": indigenous_mean - other_mean,
        "indigenous_to_other_ratio": indigenous_mean / other_mean,
        "indigenous_token_farther": indigenous_mean > other_mean,
    }
    comparison = []
    for model in MODELS:
        score = json.loads((OUTPUTS_DIR / "scores" / f"{model}_bge_strict.json").read_text(encoding="utf-8"))["score_strict"]
        comparison.append({
            "model": model,
            "observed_score_strict": score,
            "synthetic_variance_mean": summary["variance_mean"],
            "synthetic_variance_median": summary["variance_median"],
            "synthetic_variance_p95": summary["variance_p95"],
            "observed_to_synthetic_mean_ratio": score / summary["variance_mean"],
        })
    save_csv("synthetic_echo_summary.csv", comparison)
    atomic_write_json(REPORT_DIR / "synthetic_echo_summary.json", {**summary, "observed_comparison": comparison})
    print("\nSYNTHETIC ECHO BASELINE")
    print(pd.DataFrame(comparison).to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_verdict(
    overall: pd.DataFrame,
    comparisons: pd.DataFrame,
    pairs: pd.DataFrame,
    correlations: pd.DataFrame,
    synthetic: dict[str, Any],
) -> str:
    masked_pairs = pairs[pairs["variant"] == "mask"]
    masked_bge = comparisons[(comparisons["variant"] == "mask") & (comparisons["encoder"] == "bge")]
    indigenous_count = int(masked_pairs["indigenous_after_masking"].sum())
    significant_count = int((masked_bge["p_masked"] < 0.05).sum())
    observed_indigenous_means = []
    for model in MODELS:
        original = json.loads((OUTPUTS_DIR / "scores" / f"{model}_bge_strict.json").read_text(encoding="utf-8"))
        groups = original["pairwise_distance_matrix"]["groups"]
        matrix = np.asarray(original["pairwise_distance_matrix"]["matrix"])
        index = groups.index("indigena")
        observed_indigenous_means.append(float(np.delete(matrix[index], index).mean()))
    observed_low = min(observed_indigenous_means)
    observed_high = max(observed_indigenous_means)
    observed_mean = float(np.mean(observed_indigenous_means))
    synthetic_comparable = synthetic["indigenous_pairwise_mean"] >= observed_low
    if indigenous_count >= 3 and significant_count == 4:
        verdict = "SURVIVES"
    elif 4 - indigenous_count >= 3 and synthetic_comparable:
        verdict = "COLLAPSES"
    else:
        verdict = "PARTIALLY SURVIVES"

    percent = lambda value: f"{100 * value:.2f}%"
    lines = [
        "# Lexical artifact controls verdict",
        "",
        "## Echo rates",
        "",
        "| Model | Any racial marker | Prompted-group marker | Below 15%? |",
        "|---|---:|---:|:---:|",
    ]
    for row in overall.itertuples():
        lines.append(f"| {row.model} | {percent(row.echo_any_rate)} | {percent(row.echo_prompted_rate)} | {'Y' if row.below_15_percent else 'N'} |")

    lines.extend([
        "",
        "## Largest pair before and after masking",
        "",
        "| Model | Largest pair (original) | Largest pair ([GRUPO]) | Indigenous after? | Largest pair (deletion) | Indigenous after deletion? |",
        "|---|---|---|:---:|---|:---:|",
    ])
    for model in MODELS:
        masked = pairs[(pairs["model"] == model) & (pairs["variant"] == "mask")].iloc[0]
        deleted = pairs[(pairs["model"] == model) & (pairs["variant"] == "deletion")].iloc[0]
        lines.append(
            f"| {model} | {masked.largest_pair_original} | {masked.largest_pair_masked} | "
            f"{'Y' if masked.indigenous_after_masking else 'N'} | {deleted.largest_pair_masked} | "
            f"{'Y' if deleted.indigenous_after_masking else 'N'} |"
        )

    lines.extend([
        "",
        "## Masked-score significance (10,000 permutations, seed 42)",
        "",
        "| Variant | Model | Encoder | Score | p | Null floor | Ratio |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for row in comparisons.itertuples():
        lines.append(
            f"| {row.variant} | {row.model} | {row.encoder.upper()} | {row.score_masked:.6f} | "
            f"{row.p_masked:.4f} | {row.null_floor_masked:.6f} | {row.ratio_masked:.3f} |"
        )

    lines.extend([
        "",
        "## Original/masked per-tuple correlation (BGE)",
        "",
        "| Variant | Model | N | Spearman ρ |",
        "|---|---|---:|---:|",
    ])
    for row in correlations.itertuples():
        lines.append(f"| {row.variant} | {row.model} | {row.n_tuples} | {row.spearman_rho:.3f} |")

    lines.extend([
        "",
        "## Synthetic echo baseline",
        "",
        f"- Synthetic variance: mean {synthetic['variance_mean']:.6f}, median {synthetic['variance_median']:.6f}, p95 {synthetic['variance_p95']:.6f}.",
        f"- Mean indigenous-vs-other token distance: {synthetic['indigenous_pairwise_mean']:.6f}.",
        f"- Mean distance among other pairs: {synthetic['other_pairwise_mean']:.6f}.",
        f"- Indigenous token farther by this instrument-only comparison: {'Y' if synthetic['indigenous_token_farther'] else 'N'}.",
        f"- Observed indigenous-vs-other mean distance: {observed_mean:.6f}, model range [{observed_low:.6f}, {observed_high:.6f}].",
        f"- Synthetic/observed indigenous-distance ratio: {synthetic['indigenous_pairwise_mean'] / observed_mean:.3f}.",
        f"- Comparable to the observed range (synthetic mean at least the observed minimum): {'Y' if synthetic_comparable else 'N'}.",
        "",
        "## Rule-based assessment",
        "",
        f"**{verdict}**",
        "",
        f"Indigenous remains in the largest pair for {indigenous_count}/4 models after [GRUPO] masking; "
        f"{significant_count}/4 BGE masked scores have p < 0.05. "
        "Rule: SURVIVES requires at least 3/4 and significance; COLLAPSES requires indigenous to drop in at least 3/4 plus a comparable synthetic baseline; all other outcomes are PARTIALLY SURVIVES.",
        "",
    ])
    VERDICT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nFINAL VERDICT: {verdict} ({indigenous_count}/4 indigenous-largest, {significant_count}/4 significant)")
    return verdict


def read_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    overall = pd.read_csv(REPORT_DIR / "echo_overall.csv")
    comparisons = pd.read_csv(REPORT_DIR / "masking_comparison.csv")
    pairs = pd.read_csv(REPORT_DIR / "largest_pairs.csv")
    correlations = pd.read_csv(REPORT_DIR / "tuple_correlations.csv")
    synthetic = json.loads((REPORT_DIR / "synthetic_echo_summary.json").read_text(encoding="utf-8"))
    return overall, comparisons, pairs, correlations, synthetic


def main(args: argparse.Namespace) -> None:
    config = load_config()
    if args.models:
        models = tuple(args.models)
    elif args.phase == "echo":
        models = discover_complete_models()
    else:
        models = MODELS
    if args.phase != "echo" and any(model not in MODELS for model in models):
        raise RuntimeError("Additional models are supported only by --phase echo")
    rows = load_rows(models) if args.phase in {"all", "echo", "embed", "synthetic"} else []
    encoders = tuple(args.encoders)
    variants = tuple(args.variants)

    if args.phase in {"all", "echo"}:
        overall, _ = run_echo_analysis(rows, args.seed, models)
    if args.phase in {"all", "embed"}:
        encoders = run_embeddings(rows, encoders, variants, args.batch_size, config)
        if not encoders:
            raise RuntimeError("No encoder completed")
    if args.phase in {"all", "scores"}:
        run_scores(
            encoders,
            variants,
            models,
            args.permutations,
            args.seed,
            args.workers,
            args.force_permutations,
        )
        publish_primary_mask_scores()
        comparisons, pairs, correlations = build_comparisons(encoders, variants)
    if args.phase in {"all", "synthetic"}:
        synthetic = run_synthetic(rows, args.seed, args.batch_size, config)
    if args.phase == "all":
        build_verdict(overall, comparisons, pairs, correlations, synthetic)
    elif args.phase == "report":
        publish_primary_mask_scores()
        build_verdict(*read_outputs())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lexical-artifact controls on cached STRICT responses.")
    parser.add_argument("--phase", choices=("all", "echo", "embed", "scores", "synthetic", "report"), default="all")
    parser.add_argument("--models", nargs="+", help="Model IDs; echo mode auto-discovers complete caches by default")
    parser.add_argument("--encoders", nargs="+", choices=("bge", "e5"), default=["bge", "e5"])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--force-permutations", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
