from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .config import format_resolved_config, load_experiment_config, load_registry
from .dataset import (
    CANONICAL_RACIAL_GROUPS,
    export_prompts_csv,
    load_scenarios,
    summarize_scenarios,
    validate_scenarios,
)
from .embeddings import (
    embed_response_cache,
    load_complete_strict_embeddings,
    model_embedding_directory,
)
from .generation import generate_responses
from .implicit_analysis import (
    analyze_implicit_responses,
    classify_sentiment,
    load_implicit_responses,
)
from .metrics import strict_rbias_metrics
from .reporting import build_markdown_report, load_json_results
from .statistics import permutation_test
from .storage import atomic_write_json, require_output_path, safe_name


def _print_config(config: Any) -> None:
    print("Resolved experiment configuration:")
    print(format_resolved_config(config))


def _verified_scenarios(config: Any) -> list[dict[str, Any]]:
    scenarios = load_scenarios(config.scenarios_path)
    issues = validate_scenarios(scenarios)
    if issues:
        preview = "\n".join(
            f"- {issue.code} [{issue.scenario_id or 'dataset'}]: {issue.message}"
            for issue in issues[:20]
        )
        raise ValueError(f"Scenario validation failed ({len(issues)} issues):\n{preview}")
    return scenarios


def _check_expected_counts(config: Any, summary: dict[str, int]) -> list[str]:
    expected = {
        "scenarios": config.expected_scenarios,
        "prompt_variants": config.expected_prompt_variants,
        "strict_scenarios": config.expected_strict_scenarios,
        "implicit_scenarios": config.expected_implicit_scenarios,
    }
    return [
        f"{name}: expected {expected[name]}, found {summary[name]}"
        for name in expected
        if summary[name] != expected[name]
    ]


def command_validate(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    _print_config(config)
    scenarios = load_scenarios(config.scenarios_path)
    issues = validate_scenarios(scenarios)
    summary = summarize_scenarios(scenarios)
    count_errors = _check_expected_counts(config, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if issues or count_errors:
        for issue in issues:
            print(f"ERROR {issue.code} [{issue.scenario_id or 'dataset'}]: {issue.message}")
        for message in count_errors:
            print(f"ERROR count_mismatch: {message}")
        return 1
    if args.export_prompts:
        count = export_prompts_csv(scenarios, args.export_prompts, overwrite=args.overwrite)
        print(f"Exported {count} prompt variants to {args.export_prompts}")
    print("Validation passed.")
    return 0


def command_generate(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    _print_config(config)
    scenarios = _verified_scenarios(config)
    models = load_registry(config.models_path, "models")
    if args.model not in models:
        raise ValueError(f"Unknown model {args.model!r}; available: {', '.join(models)}")
    summary = generate_responses(
        scenarios,
        args.model,
        models[args.model],
        config.generation,
        args.output_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["errors"] == 0 else 2


def command_embed(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    _print_config(config)
    encoders = load_registry(config.encoders_path, "auxiliary_encoders")
    if args.encoder not in encoders:
        raise ValueError(
            f"Unknown auxiliary encoder {args.encoder!r}; available: {', '.join(encoders)}"
        )
    embedding_dir = model_embedding_directory(args.output_dir, args.encoder, args.model)
    summary = embed_response_cache(
        args.input_dir,
        embedding_dir,
        encoders[args.encoder],
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(json.dumps({**summary, "embedding_dir": str(embedding_dir)}, indent=2))
    return 0


def _score_output(args: argparse.Namespace, config: Any) -> tuple[Path, dict[str, Any]]:
    scenarios = _verified_scenarios(config)
    batch = load_complete_strict_embeddings(
        scenarios, args.input_dir, int(config.generation["n_samples"])
    )
    metrics = strict_rbias_metrics(batch.values)
    per_category: dict[str, list[float]] = defaultdict(list)
    for category, value in zip(batch.categories, metrics["per_scenario"]):
        per_category[category].append(float(value))
    category_summary = {
        category: {
            "n_scenarios": len(values),
            "score": sum(values) / len(values),
        }
        for category, values in sorted(per_category.items())
    }
    output = {
        "model": args.model,
        "encoder": args.encoder,
        **metrics,
        "racial_groups": list(CANONICAL_RACIAL_GROUPS),
        "scenario_ids": list(batch.scenario_ids),
        "excluded_scenarios": list(batch.excluded),
        "per_category": category_summary,
        "metadata": {
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "experiment": config.name,
            "missing_data_policy": "exclude any tuple missing a group or requested generation",
        },
    }
    filename = f"{safe_name(args.model)}_{safe_name(args.encoder)}_strict.json"
    destination = Path(args.output_dir) / "scores" / filename
    return destination, output


def command_evaluate(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    _print_config(config)
    destination, output = _score_output(args, config)
    if args.dry_run:
        print(f"Would write: {destination}")
        print(f"Complete strict scenarios: {output['n_scenarios']}")
        print(f"Excluded scenarios: {len(output['excluded_scenarios'])}")
        return 0
    atomic_write_json(destination, output, overwrite=args.overwrite)
    print(f"Wrote: {destination}")
    print(f"Score_strict: {output['score_strict']:.6f}")
    return 0


def command_permutation(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    _print_config(config)
    scenarios = _verified_scenarios(config)
    batch = load_complete_strict_embeddings(
        scenarios, args.input_dir, int(config.generation["n_samples"])
    )
    n_permutations = args.n_permutations or int(config.permutation["n_permutations"])
    seed = args.seed if args.seed is not None else int(config.permutation["seed"])
    destination = (
        Path(args.output_dir)
        / "scores"
        / f"{safe_name(args.model)}_{safe_name(args.encoder)}_permutation.json"
    )
    require_output_path(destination, args.overwrite)
    if args.dry_run:
        print(f"Would run {n_permutations} permutations on {len(batch.scenario_ids)} scenarios")
        print(f"Would write: {destination}")
        return 0
    result = permutation_test(batch.values, n_permutations=n_permutations, seed=seed)
    output = {
        "model": args.model,
        "encoder": args.encoder,
        "n_scenarios": len(batch.scenario_ids),
        "observed_score": result.observed_score,
        "p_value": result.p_value,
        "null_distribution_mean": result.null_mean,
        "null_ci_95": list(result.null_ci_95),
        "null_distribution": result.null_distribution.tolist(),
        "excluded_scenarios": list(batch.excluded),
        "metadata": {
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "n_permutations": n_permutations,
            "seed": seed,
            "experiment": config.name,
        },
    }
    atomic_write_json(destination, output, overwrite=args.overwrite)
    print(f"Wrote: {destination}")
    print(f"Observed score: {result.observed_score:.6f}; p-value: {result.p_value:.6f}")
    return 0


def command_analyze_implicit(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    _print_config(config)
    records, failed = load_implicit_responses(args.input_dir)
    destination = Path(args.output_dir) / "scores" / f"{safe_name(args.model)}_implicit.json"
    require_output_path(destination, args.overwrite)
    if args.dry_run:
        print(
            f"Would analyze {len(records)} successful implicit responses "
            f"({failed} failed or empty)"
        )
        print(f"Would write: {destination}")
        return 0
    sentiment_model = None
    labels: list[str | None] | None = None
    if not args.without_sentiment:
        labels, sentiment_model = classify_sentiment(
            [record["response_text"] for record in records],
            str(config.sentiment["primary"]),
            str(config.sentiment["fallback"]),
        )
    output = analyze_implicit_responses(
        records, config.refusal_lexicon, labels, failed_or_empty=failed
    )
    output.update(
        {
            "model": args.model,
            "sentiment_model": sentiment_model,
            "metadata": {
                "computed_at": datetime.now(timezone.utc).isoformat(),
                "experiment": config.name,
            },
        }
    )
    atomic_write_json(destination, output, overwrite=args.overwrite)
    print(json.dumps(output["aggregate_by_group"], ensure_ascii=False, indent=2))
    print(f"Wrote: {destination}")
    return 0


def command_report(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    _print_config(config)
    results = load_json_results(args.scores)
    destination = Path(args.output_dir) / "reports" / "summary.md"
    require_output_path(destination, args.overwrite)
    if args.dry_run:
        print(f"Would assemble {len(results)} result files into {destination}")
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_markdown_report(results), encoding="utf-8")
    print(f"Wrote: {destination}")
    return 0


def _config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path, help="Experiment YAML file")


def _overwrite_arguments(parser: argparse.ArgumentParser, include_dry_run: bool = True) -> None:
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow replacement of existing outputs"
    )
    if include_dry_run:
        parser.add_argument(
            "--dry-run", action="store_true", help="Resolve and print work without writing"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rbias", description="R-BIAS local evaluation workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="Validate scenarios and counterfactual tuples"
    )
    _config_argument(validate)
    validate.add_argument("--export-prompts", type=Path)
    validate.add_argument("--overwrite", action="store_true")
    validate.set_defaults(function=command_validate)

    generate = subparsers.add_parser("generate", help="Generate or reuse model responses")
    _config_argument(generate)
    generate.add_argument("--model", required=True)
    generate.add_argument("--output-dir", required=True, type=Path)
    _overwrite_arguments(generate)
    generate.set_defaults(function=command_generate)

    embed = subparsers.add_parser("embed", help="Extract auxiliary-encoder response embeddings")
    _config_argument(embed)
    embed.add_argument("--model", required=True)
    embed.add_argument("--encoder", required=True)
    embed.add_argument("--input-dir", required=True, type=Path)
    embed.add_argument("--output-dir", required=True, type=Path)
    embed.add_argument("--batch-size", type=int, default=128)
    _overwrite_arguments(embed)
    embed.set_defaults(function=command_embed)

    evaluate = subparsers.add_parser("evaluate", help="Calculate strict R-BIAS metrics")
    _config_argument(evaluate)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--encoder", required=True)
    evaluate.add_argument("--input-dir", required=True, type=Path)
    evaluate.add_argument("--output-dir", required=True, type=Path)
    _overwrite_arguments(evaluate)
    evaluate.set_defaults(function=command_evaluate)

    permutation = subparsers.add_parser("permutation-test", help="Run the label-permutation null")
    _config_argument(permutation)
    permutation.add_argument("--model", required=True)
    permutation.add_argument("--encoder", required=True)
    permutation.add_argument("--input-dir", required=True, type=Path)
    permutation.add_argument("--output-dir", required=True, type=Path)
    permutation.add_argument("--n-permutations", type=int)
    permutation.add_argument("--seed", type=int)
    _overwrite_arguments(permutation)
    permutation.set_defaults(function=command_permutation)

    implicit = subparsers.add_parser(
        "analyze-implicit", help="Analyze implicit-signaling responses"
    )
    _config_argument(implicit)
    implicit.add_argument("--model", required=True)
    implicit.add_argument("--input-dir", required=True, type=Path)
    implicit.add_argument("--output-dir", required=True, type=Path)
    implicit.add_argument("--without-sentiment", action="store_true")
    _overwrite_arguments(implicit)
    implicit.set_defaults(function=command_analyze_implicit)

    report = subparsers.add_parser("report", help="Build a conservative Markdown summary")
    _config_argument(report)
    report.add_argument("--scores", required=True, nargs="+", type=Path)
    report.add_argument("--output-dir", required=True, type=Path)
    _overwrite_arguments(report)
    report.set_defaults(function=command_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))
