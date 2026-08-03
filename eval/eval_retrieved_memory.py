"""Run retrieval-only memory quality and budget metrics."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from statistics import mean
from typing import Any

from soca.knowledge.factory import RetrievalConfig, build_retrieval_source
from soca.llm.providers.model_catalog import fetch_catalog
from soca.llm.providers.provider_registry import get_provider
from soca.llm.providers.remote_openai_llm import RemoteOpenAILLM
from soca.memory import CoreMemoryStore, RetrievedMemory, RetrievedMemoryConfig


def _cases(path: Path, limit: int | None) -> tuple[dict[str, Any], ...]:
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("memory eval rows must be objects")
                values.append(value)
                if limit is not None and len(values) >= limit:
                    break
    if not values:
        raise ValueError("memory eval dataset is empty")
    return tuple(values)


def evaluate(vault: Path, cases_path: Path, *, limit: int | None = None) -> dict[str, Any]:
    cases = _cases(cases_path, limit)
    core = CoreMemoryStore(vault, max_chars=2_200)
    source = build_retrieval_source(
        vault,
        include_globs=("memory/**/*.md",),
        config=RetrievalConfig(mode="chunk_sparse"),
    )
    retrieved = RetrievedMemory(source, core, config=RetrievedMemoryConfig(top_k=3, max_chars=2_200))
    samples: list[dict[str, float | bool]] = []
    try:
        for case in cases:
            started = time.perf_counter()
            result = retrieved.retrieve_archive(str(case["query"]))
            latency = (time.perf_counter() - started) * 1000
            text = result.text.casefold()
            expected = tuple(str(item).casefold() for item in case.get("expected_contains", ()))
            forbidden = tuple(str(item).casefold() for item in case.get("forbidden_contains", ()))
            samples.append(
                {
                    "hit": bool(expected) and all(item in text for item in expected),
                    "forbidden_leak": any(item in text for item in forbidden),
                    "chars": len(text),
                    "latency_ms": latency,
                }
            )
    finally:
        retrieved.close()
    return {
        "status": "ok",
        "case_count": len(samples),
        "recall": mean(float(sample["hit"]) for sample in samples),
        "forbidden_leakage_rate": mean(float(sample["forbidden_leak"]) for sample in samples),
        "context_chars_mean": mean(float(sample["chars"]) for sample in samples),
        "latency_mean_ms": mean(float(sample["latency_ms"]) for sample in samples),
    }


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def evaluate_answers(
    vault: Path,
    cases_path: Path,
    *,
    provider_key: str,
    model_id: str,
    variants: tuple[str, ...],
    repetitions: int = 1,
    limit: int | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    cases = _cases(cases_path, limit)
    provider = get_provider(provider_key)
    key = api_key or os.environ.get(provider.api_key_env)
    if not key:
        raise ValueError(f"{provider.api_key_env} is required for answer mode")
    llm = RemoteOpenAILLM(provider, model_id, key, timeout=60.0, max_retries=0)
    catalog = fetch_catalog(provider, key)
    model_info = next((item for item in catalog if item.id == model_id), None)
    source_cache: dict[str, RetrievedMemory] = {}
    core = CoreMemoryStore(vault, max_chars=2_200)
    results: dict[str, dict[str, float | int]] = {}
    try:
        for variant in variants:
            if variant == "core_only":
                retrieved = None
            elif variant in {"retrieved_chunk_sparse", "retrieved_hybrid"}:
                mode = "chunk_sparse" if variant.endswith("chunk_sparse") else "hybrid"
                source = build_retrieval_source(
                    vault,
                    include_globs=("memory/**/*.md",),
                    config=RetrievalConfig(mode=mode),
                )
                retrieved = RetrievedMemory(source, core, config=RetrievedMemoryConfig(top_k=3, max_chars=2_200))
                source_cache[variant] = retrieved
            else:
                raise ValueError(f"unknown answer variant: {variant}")

            samples: list[dict[str, float | int]] = []
            for _ in range(repetitions):
                for case in cases:
                    started = time.perf_counter()
                    context = core.read_core() if retrieved is None else retrieved.retrieve_archive(str(case["query"])).text
                    prompt = (
                        "Answer the question using only the retrieved memory. "
                        "If the memory does not contain the answer, say unknown.\n"
                        f"Question: {case['query']}\nMemory:\n{context}"
                    )
                    answer = llm.generate(prompt, max_tokens=64, temperature=0.0, top_p=1.0, inject_persona=False)
                    text = answer.text.casefold()
                    expected = tuple(str(item).casefold() for item in case.get("expected_contains", ()))
                    forbidden = tuple(str(item).casefold() for item in case.get("forbidden_contains", ()))
                    samples.append(
                        {
                            "correct": int(bool(expected) and all(item in text for item in expected)),
                            "forbidden": int(any(item in text for item in forbidden)),
                            "prompt_tokens": answer.n_prompt_tokens,
                            "completion_tokens": answer.n_completion_tokens,
                            "latency_ms": (time.perf_counter() - started) * 1000,
                        }
                    )
            prompt_cost = model_info.price_prompt_per_1m if model_info else None
            completion_cost = model_info.price_completion_per_1m if model_info else None
            results[variant] = {
                "case_count": len(samples),
                "accuracy": mean(float(sample["correct"]) for sample in samples),
                "forbidden_leakage_rate": mean(float(sample["forbidden"]) for sample in samples),
                "prompt_tokens": int(sum(sample["prompt_tokens"] for sample in samples)),
                "completion_tokens": int(sum(sample["completion_tokens"] for sample in samples)),
                "latency_mean_ms": mean(float(sample["latency_ms"]) for sample in samples),
                "estimated_cost_usd": (
                    (sum(sample["prompt_tokens"] for sample in samples) * (prompt_cost or 0.0)
                     + sum(sample["completion_tokens"] for sample in samples) * (completion_cost or 0.0))
                    / 1_000_000
                    if prompt_cost is not None and completion_cost is not None
                    else -1.0
                ),
            }
    finally:
        close_errors: list[BaseException] = []
        for retrieved in source_cache.values():
            try:
                retrieved.close()
            except BaseException as exc:  # noqa: BLE001 - preserve eval cleanup failure
                close_errors.append(exc)
        if close_errors:
            raise RuntimeError("retrieved-memory evaluation cleanup failed") from close_errors[0]
    return {"status": "ok", "provider": provider_key, "model": model_id, "variants": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("retrieval-only", "answer"), default="retrieval-only")
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--model")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "answer":
        if not args.model:
            raise SystemExit("--model is required in answer mode")
        env = _read_env_file(args.env_file) if args.env_file else {}
        result = evaluate_answers(
            args.vault,
            args.cases,
            provider_key=args.provider,
            model_id=args.model,
            variants=tuple(args.variant or ["core_only", "retrieved_chunk_sparse"]),
            repetitions=args.repetitions,
            limit=args.limit,
            api_key=env.get(get_provider(args.provider).api_key_env),
        )
        if args.max_cost_usd is not None:
            costs = [float(item["estimated_cost_usd"]) for item in result["variants"].values()]
            if all(cost >= 0 for cost in costs) and sum(costs) > args.max_cost_usd:
                raise SystemExit("estimated cost exceeds --max-cost-usd")
    else:
        result = evaluate(args.vault, args.cases, limit=args.limit)
    output = args.output or ((args.output_dir / "latest.json") if args.output_dir else None)
    if output is None:
        raise SystemExit("--output or --output-dir is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
