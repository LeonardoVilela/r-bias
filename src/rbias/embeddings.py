from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .dataset import SOURCE_GROUP_BY_CANONICAL, SOURCE_RACIAL_GROUPS, STRICT_CATEGORIES
from .storage import atomic_write_npy, safe_name


def _group_file_name(scenario_id: str, source_group: str, sample_index: int) -> str:
    group = source_group.replace("/", "-").replace(" ", "_")
    return f"{scenario_id}_{group}_{sample_index}.npy"


def _sample_index(record: dict[str, Any], path: Path) -> int:
    value = record.get("sample_index")
    if isinstance(value, int) and value >= 0:
        return value
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"Missing sample_index in {path}") from exc


def _source_group(record: dict[str, Any]) -> str:
    source = record.get("source_racial_group") or record.get("grupo_racial_alvo")
    if source in SOURCE_RACIAL_GROUPS:
        return str(source)
    canonical = record.get("racial_group")
    if canonical in SOURCE_GROUP_BY_CANONICAL:
        return SOURCE_GROUP_BY_CANONICAL[str(canonical)]
    raise ValueError(f"Unknown racial group in response record: {source or canonical!r}")


class ResponseEncoder:
    def __init__(self, encoder_config: dict[str, Any]) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        model_id = encoder_config.get("model_id")
        if not isinstance(model_id, str):
            raise ValueError("Encoder configuration requires model_id")
        if encoder_config.get("pooling") != "mean":
            raise ValueError("The verified implementation currently supports mean pooling only")

        self.torch = torch
        self.prefix = str(encoder_config.get("prefix") or "")
        self.normalize = bool(encoder_config.get("normalize", True))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(self.device)
        self.model.eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        inputs = self.tokenizer(
            [self.prefix + text for text in texts],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with torch.no_grad():
            output = self.model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).to(output.last_hidden_state.dtype)
        vectors = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        if self.normalize:
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
        return vectors.cpu().float().numpy()


def embed_response_cache(
    response_dir: str | Path,
    embedding_dir: str | Path,
    encoder_config: dict[str, Any],
    batch_size: int = 128,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    source = Path(response_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Response directory not found: {source}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    pending: list[tuple[Path, dict[str, Any], Path]] = []
    skipped_failed = 0
    skipped_cached = 0
    for path in sorted(source.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid response JSON: {path}") from exc
        text = record.get("response_text")
        status = record.get("status", "ok" if text else "error")
        if status not in {"ok", "success"} or not isinstance(text, str) or not text.strip():
            skipped_failed += 1
            continue
        scenario_id = record.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError(f"Missing scenario_id in {path}")
        output = Path(embedding_dir) / _group_file_name(
            scenario_id, _source_group(record), _sample_index(record, path)
        )
        if output.exists() and not overwrite:
            skipped_cached += 1
            continue
        pending.append((path, record, output))

    if dry_run or not pending:
        return {
            "response_files": len(list(source.glob("*.json"))),
            "pending": len(pending),
            "cached": skipped_cached,
            "failed_or_empty": skipped_failed,
            "written": 0,
        }

    encoder = ResponseEncoder(encoder_config)
    written = 0
    for start in tqdm(range(0, len(pending), batch_size), desc="Embedding", unit="batch"):
        batch = pending[start : start + batch_size]
        vectors = encoder.encode([str(record["response_text"]) for _, record, _ in batch])
        for vector, (_, _, output) in zip(vectors, batch):
            atomic_write_npy(output, vector, overwrite=overwrite)
            written += 1
    return {
        "response_files": len(list(source.glob("*.json"))),
        "pending": len(pending),
        "cached": skipped_cached,
        "failed_or_empty": skipped_failed,
        "written": written,
    }


@dataclass(frozen=True)
class EmbeddingBatch:
    values: np.ndarray
    scenario_ids: tuple[str, ...]
    categories: tuple[str, ...]
    excluded: tuple[dict[str, str], ...]


def load_complete_strict_embeddings(
    scenarios: list[dict[str, Any]],
    embedding_dir: str | Path,
    n_samples: int,
) -> EmbeddingBatch:
    """Load only strict tuples with all five groups and all requested samples.

    Every vector must be one-dimensional, finite, and have a common dimension.
    Incomplete tuples are excluded and reported; they are never padded or silently
    evaluated with a smaller sample count.
    """
    directory = Path(embedding_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Embedding directory not found: {directory}")
    rows: list[np.ndarray] = []
    scenario_ids: list[str] = []
    categories: list[str] = []
    excluded: list[dict[str, str]] = []
    expected_dimensions: int | None = None

    for scenario in scenarios:
        if scenario.get("categoria") not in STRICT_CATEGORIES:
            continue
        scenario_id = str(scenario["scenario_id"])
        groups: list[np.ndarray] = []
        reason: str | None = None
        for source_group in SOURCE_RACIAL_GROUPS:
            samples: list[np.ndarray] = []
            for sample_index in range(n_samples):
                path = directory / _group_file_name(scenario_id, source_group, sample_index)
                if not path.is_file():
                    reason = f"missing {source_group} sample {sample_index}"
                    break
                try:
                    vector = np.load(path).astype(np.float64)
                except Exception:
                    reason = f"invalid embedding file {path.name}"
                    break
                if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
                    reason = f"invalid embedding vector {path.name}"
                    break
                if expected_dimensions is None:
                    expected_dimensions = int(vector.size)
                elif vector.size != expected_dimensions:
                    reason = f"dimension mismatch in {path.name}"
                    break
                samples.append(vector)
            if reason:
                break
            groups.append(np.stack(samples))
        if reason:
            excluded.append({"scenario_id": scenario_id, "reason": reason})
            continue
        rows.append(np.stack(groups))
        scenario_ids.append(scenario_id)
        categories.append(str(scenario["categoria"]))

    if not rows:
        raise ValueError("No complete strict tuples were found")
    return EmbeddingBatch(
        values=np.stack(rows),
        scenario_ids=tuple(scenario_ids),
        categories=tuple(categories),
        excluded=tuple(excluded),
    )


def model_embedding_directory(output_dir: str | Path, encoder: str, model: str) -> Path:
    return Path(output_dir) / "embeddings" / safe_name(encoder) / safe_name(model)
