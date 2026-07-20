from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from tqdm import tqdm

from .dataset import prompt_records
from .storage import atomic_write_json, safe_name


class GenerationBackend(Protocol):
    provider_name: str

    def generate(
        self, prompt: str, *, temperature: float, top_p: float, max_tokens: int
    ) -> tuple[str, str]: ...

    def close(self) -> None: ...


class AnthropicBackend:
    provider_name = "anthropic"

    def __init__(self, model: str, api_key: str) -> None:
        import anthropic

        self.model = model
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(
        self, prompt: str, *, temperature: float, top_p: float, max_tokens: int
    ) -> tuple[str, str]:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text, message.model

    def close(self) -> None:
        return None


class OpenAIBackend:
    def __init__(self, model: str, api_key: str, provider: str, base_url: str | None) -> None:
        import openai

        self.model = model
        self.provider_name = provider
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def generate(
        self, prompt: str, *, temperature: float, top_p: float, max_tokens: int
    ) -> tuple[str, str]:
        arguments: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.model.startswith(("o1", "o3", "o4")):
            arguments["max_completion_tokens"] = max_tokens
        elif self.model.startswith(("gpt-4.1", "gpt-5")):
            arguments["max_completion_tokens"] = max_tokens
            arguments["temperature"] = temperature
        else:
            arguments.update(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        response = self.client.chat.completions.create(**arguments)
        return response.choices[0].message.content or "", response.model

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


class GoogleBackend:
    provider_name = "google"

    def __init__(self, model: str, api_key: str) -> None:
        from google import genai

        self.model = model
        self.client = genai.Client(api_key=api_key)

    def generate(
        self, prompt: str, *, temperature: float, top_p: float, max_tokens: int
    ) -> tuple[str, str]:
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text or "", self.model

    def close(self) -> None:
        return None


class LocalVLLMBackend:
    provider_name = "local"

    def __init__(self, alias: str, model_config: dict[str, Any]) -> None:
        from vllm import LLM

        self.alias = alias
        self.lock = threading.Lock()
        arguments: dict[str, Any] = {
            "model": model_config.get("model_id", alias),
            "tensor_parallel_size": int(model_config.get("tensor_parallel_size", 1)),
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.90,
        }
        if model_config.get("quantization"):
            arguments["quantization"] = model_config["quantization"]
            arguments["load_format"] = model_config["quantization"]
        self.model = LLM(**arguments)

    def generate(
        self, prompt: str, *, temperature: float, top_p: float, max_tokens: int
    ) -> tuple[str, str]:
        from vllm import SamplingParams

        parameters = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        with self.lock:
            outputs = self.model.chat(
                [[{"role": "user", "content": prompt}]], sampling_params=parameters
            )
        return outputs[0].outputs[0].text, self.alias

    def close(self) -> None:
        self.model = None


def build_backend(model: str, model_config: dict[str, Any]) -> GenerationBackend:
    provider = model_config.get("provider")
    if provider == "local":
        return LocalVLLMBackend(model, model_config)
    environment_variable = model_config.get("environment_variable")
    if not isinstance(environment_variable, str):
        raise ValueError(f"Model {model!r} has no credential environment variable")
    api_key = os.environ.get(environment_variable)
    if not api_key:
        raise RuntimeError(f"Required environment variable is not set: {environment_variable}")
    if provider == "anthropic":
        return AnthropicBackend(model, api_key)
    if provider in {"openai", "maritaca"}:
        return OpenAIBackend(model, api_key, str(provider), model_config.get("base_url"))
    if provider == "google":
        return GoogleBackend(model, api_key)
    raise ValueError(f"Unsupported provider for {model!r}: {provider!r}")


def _response_path(
    output_dir: Path,
    model: str,
    scenario_id: str,
    source_group: str,
    sample: int,
) -> Path:
    group = source_group.replace("/", "-").replace(" ", "_")
    return output_dir / "responses" / safe_name(model) / f"{scenario_id}_{group}_{sample}.json"


def _cached_success(path: Path) -> bool:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        status = record.get("status", "ok" if record.get("response_text") else "error")
        return status in {"ok", "success"} and bool(str(record.get("response_text") or "").strip())
    except Exception:
        return False


def _transient_error(error: Exception) -> bool:
    text = str(error).lower()
    transient_tokens = ("429", "rate limit", "quota", "500", "502", "503", "504")
    return any(token in text for token in transient_tokens)


def _generate_record(
    backend: GenerationBackend,
    model: str,
    prompt: dict[str, Any],
    sample_index: int,
    generation: dict[str, Any],
) -> dict[str, Any]:
    error: Exception | None = None
    for attempt in range(int(generation["max_retries"])):
        try:
            response_text, model_version = backend.generate(
                str(prompt["prompt_text"]),
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                max_tokens=int(generation["max_tokens"]),
            )
            if not response_text.strip():
                raise ValueError("Provider returned an empty response")
            return {
                "scenario_id": prompt["scenario_id"],
                "prompt_id": f"{prompt['scenario_id']}:{prompt['racial_group']}",
                "category": prompt["category"],
                "regime": prompt["regime"],
                "racial_group": prompt["racial_group"],
                "source_racial_group": prompt["source_racial_group"],
                "prompt_text": prompt["prompt_text"],
                "response_text": response_text,
                "sample_index": sample_index,
                "model_name": model,
                "model_version": model_version,
                "provider": backend.provider_name,
                "temperature": float(generation["temperature"]),
                "top_p": float(generation["top_p"]),
                "max_tokens": int(generation["max_tokens"]),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "success",
                "error_message": None,
            }
        except Exception as exc:
            error = exc
            if not _transient_error(exc) or attempt + 1 == int(generation["max_retries"]):
                break
            time.sleep(2**attempt)
    return {
        "scenario_id": prompt["scenario_id"],
        "prompt_id": f"{prompt['scenario_id']}:{prompt['racial_group']}",
        "category": prompt["category"],
        "regime": prompt["regime"],
        "racial_group": prompt["racial_group"],
        "source_racial_group": prompt["source_racial_group"],
        "prompt_text": prompt["prompt_text"],
        "response_text": "",
        "sample_index": sample_index,
        "model_name": model,
        "model_version": None,
        "provider": backend.provider_name,
        "temperature": float(generation["temperature"]),
        "top_p": float(generation["top_p"]),
        "max_tokens": int(generation["max_tokens"]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "error",
        "error_message": f"{type(error).__name__}: {error}" if error else "Unknown error",
    }


def generate_responses(
    scenarios: list[dict[str, Any]],
    model: str,
    model_config: dict[str, Any],
    generation: dict[str, Any],
    output_dir: str | Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    """Generate or reuse one response file per prompt and sample.

    Existing successful responses are always reused. Corrupt or failed records
    cause a preflight error unless ``overwrite`` is explicit, preventing an
    interrupted rerun from issuing accidental duplicate API calls.
    """
    destination = Path(output_dir)
    tasks: list[tuple[dict[str, Any], int, Path]] = []
    cached = 0
    blocked: list[Path] = []
    for prompt in prompt_records(scenarios):
        for sample_index in range(int(generation["n_samples"])):
            path = _response_path(
                destination,
                model,
                str(prompt["scenario_id"]),
                str(prompt["source_racial_group"]),
                sample_index,
            )
            if path.exists() and _cached_success(path):
                cached += 1
                continue
            if path.exists() and not overwrite:
                blocked.append(path)
                continue
            tasks.append((prompt, sample_index, path))
    if blocked:
        raise FileExistsError(
            f"{len(blocked)} corrupt or failed response files already exist; "
            "inspect them and pass --overwrite to replace them"
        )
    if dry_run or not tasks:
        return {
            "total": cached + len(tasks),
            "cached": cached,
            "pending": len(tasks),
            "written": 0,
            "errors": 0,
        }

    backend = build_backend(model, model_config)
    errors = 0
    try:
        with ThreadPoolExecutor(max_workers=int(generation["concurrency"])) as executor:
            future_to_task = {
                executor.submit(
                    _generate_record, backend, model, prompt, sample_index, generation
                ): (path, prompt, sample_index)
                for prompt, sample_index, path in tasks
            }
            completed = tqdm(
                as_completed(future_to_task),
                total=len(tasks),
                desc="Generation",
                unit="response",
            )
            for future in completed:
                path, _, _ = future_to_task[future]
                record = future.result()
                atomic_write_json(path, record, overwrite=overwrite)
                errors += record["status"] != "success"
    finally:
        backend.close()
    return {
        "total": cached + len(tasks),
        "cached": cached,
        "pending": len(tasks),
        "written": len(tasks),
        "errors": int(errors),
    }
