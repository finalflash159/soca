"""Capture and score the complete deterministic -> semantic router cascade.

P0 deliberately records raw semantic scores and does not fit a production
threshold. The evaluator can run with provisioned local embeddings or score a
captured prediction file without loading any model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval.eval_turn_policy import evaluate as evaluate_turn_policy
from eval.eval_turn_policy import require_exact_prediction_ids
from soca.core.router_setup import build_runtime_tool_router
from soca.core.runtime import DefaultRuntimeToolRouter
from soca.core.tool_routing import SemanticRouterConfig, ToolRouterConfig
from soca.knowledge.retrievers.dense import FastEmbedModel
from soca.tools import LocalTimeTool, ToolRuntime

_DISPOSITIONS = {"direct_tool", "retrieval_request", "smalltalk", "out_of_scope", "unresolved"}
_SOURCES = {"knowledge", "memory"}
_SPLITS = {"train", "validation", "test"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    families: dict[str, str] = {}
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            row_id = row.get("id")
            family = row.get("family")
            split = row.get("split")
            query = row.get("query") or row.get("transcript")
            disposition = row.get("disposition")
            sources = row.get("sources", [])
            if (
                not isinstance(row_id, str)
                or not row_id.strip()
                or row_id in ids
                or not isinstance(family, str)
                or not family.strip()
                or split not in _SPLITS
                or not isinstance(query, str)
                or not query.strip()
                or disposition not in _DISPOSITIONS
                or not isinstance(sources, list)
                or not set(sources) <= _SOURCES
            ):
                raise ValueError(f"{path}:{line_number}: invalid routing row")
            previous_split = families.get(family)
            if previous_split is not None and previous_split != split:
                raise ValueError(f"{path}:{line_number}: family crosses splits: {family}")
            if disposition == "direct_tool" and not isinstance(row.get("tool"), str):
                raise ValueError(f"{path}:{line_number}: direct tool row needs tool")
            if disposition != "direct_tool" and row.get("tool") is not None:
                raise ValueError(f"{path}:{line_number}: non-tool row has tool")
            if disposition != "retrieval_request" and sources:
                raise ValueError(f"{path}:{line_number}: non-retrieval row has sources")
            ids.add(row_id)
            families[family] = str(split)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: dataset is empty")
    if set(row["split"] for row in rows) != _SPLITS:
        raise ValueError(f"{path}: dataset needs train/validation/test splits")
    return tuple(rows)


def _catalog() -> ToolRuntime:
    return ToolRuntime([LocalTimeTool()])


def _build_router(examples: Path, *, threshold: float, margin: float):
    model = FastEmbedModel(allow_download=False)
    return build_runtime_tool_router(
        llm=None,
        tool_runtime=_catalog(),
        deterministic=DefaultRuntimeToolRouter(enable_memory_search=True),
        config=ToolRouterConfig(
            mode="cascade",
            semantic=SemanticRouterConfig(
                enabled=True,
                threshold=threshold,
                margin=margin,
                examples_path=examples,
            ),
        ),
        embedding_model=model,
        voice=False,
    )


def run_local(
    dataset: Path,
    examples: Path,
    predictions: Path,
    *,
    threshold: float = 0.58,
    margin: float = 0.04,
) -> dict[str, Any]:
    rows = load_dataset(dataset)
    router = _build_router(examples, threshold=threshold, margin=margin)
    captured: list[dict[str, Any]] = []
    for row in rows:
        query = str(row.get("query") or row.get("transcript"))
        started = time.perf_counter()
        call = router.select(query, knowledge_limit=3)
        elapsed = (time.perf_counter() - started) * 1000
        decision = router.last_decision
        disposition = decision.disposition
        if call is not None:
            disposition = "direct_tool"
        record = {
            "id": row["id"],
            "tool": call.name if call is not None else "none",
            "disposition": disposition,
            "sources": list(decision.sources),
            "source_scores": decision.source_scores,
            "tier": getattr(router, "last_tier", "none"),
            "reason": decision.reason,
            "scores": decision.scores,
            "runner_up": decision.runner_up,
            "margin": decision.margin,
            "latency_ms": elapsed,
        }
        if isinstance(row.get("parity_id"), str):
            record["parity_id"] = row["parity_id"]
        if isinstance(row.get("input_mode"), str):
            record["input_mode"] = row["input_mode"]
        captured.append(record)
    predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in captured),
        encoding="utf-8",
    )
    return evaluate(dataset, predictions, examples=examples)


def _load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError(f"{path}:{line_number}: invalid prediction row")
            if row["id"] in result:
                raise ValueError(f"{path}:{line_number}: duplicate prediction id")
            if row.get("disposition") not in _DISPOSITIONS:
                raise ValueError(f"{path}:{line_number}: invalid predicted disposition")
            sources = row.get("sources", [])
            if not isinstance(sources, list) or not set(sources) <= _SOURCES:
                raise ValueError(f"{path}:{line_number}: invalid predicted sources")
            result[row["id"]] = row
    return result


def _distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.50) - 1)],
        "p95": ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)],
        "min": ordered[0],
        "max": ordered[-1],
    }


def _wilson(successes: int, total: int, *, z: float = 1.959963984540054) -> dict[str, float | int]:
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    if total == 0:
        return {"successes": successes, "total": total, "rate": 0.0, "lower": 0.0, "upper": 0.0}
    rate = successes / total
    denominator = 1 + z * z / total
    centre = rate + z * z / (2 * total)
    spread = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
    return {
        "successes": successes,
        "total": total,
        "rate": rate,
        "lower": max(0.0, (centre - spread) / denominator),
        "upper": min(1.0, (centre + spread) / denominator),
    }


def _score_tools(rows: tuple[dict[str, Any], ...], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pairs = [(row, predictions[row["id"]]) for row in rows if row["id"] in predictions]
    direct = [(row, prediction) for row, prediction in pairs if row["disposition"] == "direct_tool"]
    exact = sum(prediction.get("tool") == row.get("tool") for row, prediction in direct)
    unsupported_rows = [(row, prediction) for row, prediction in pairs if row["disposition"] == "out_of_scope"]
    unsupported = sum(prediction.get("tool") not in {None, "", "none"} for _, prediction in unsupported_rows)
    return {
        "direct_tool_exact": _wilson(exact, len(direct)),
        "unsupported_to_real_tool": _wilson(unsupported, len(unsupported_rows)),
    }


def _score_families(rows: tuple[dict[str, Any], ...], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        if row["id"] in predictions:
            grouped[str(row["family"])].append((row, predictions[row["id"]]))
    result: dict[str, Any] = {}
    for family, pairs in sorted(grouped.items()):
        disposition_matches = sum(row["disposition"] == prediction.get("disposition") for row, prediction in pairs)
        source_matches = sum(set(row.get("sources", [])) == set(prediction.get("sources", [])) for row, prediction in pairs)
        result[family] = {
            "count": len(pairs),
            "disposition_accuracy": disposition_matches / len(pairs),
            "source_exact_accuracy": source_matches / len(pairs),
        }
    return result


def _score_splits(rows: tuple[dict[str, Any], ...], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        subset = tuple(row for row in rows if row["split"] == split)
        pairs = [(row, predictions[row["id"]]) for row in subset if row["id"] in predictions]
        disposition_matches = sum(row["disposition"] == prediction.get("disposition") for row, prediction in pairs)
        source_matches = sum(set(row.get("sources", [])) == set(prediction.get("sources", [])) for row, prediction in pairs)
        out_of_scope = [(row, prediction) for row, prediction in pairs if row["disposition"] == "out_of_scope"]
        unsupported = sum(prediction.get("tool") not in {None, "", "none"} for _, prediction in out_of_scope)
        result[split] = {
            "case_count": len(subset),
            "scored_count": len(pairs),
            "disposition_accuracy": disposition_matches / len(pairs) if pairs else 0.0,
            "source_exact_accuracy": source_matches / len(pairs) if pairs else 0.0,
            "unsupported_to_real_tool": _wilson(unsupported, len(out_of_scope)),
        }
    return result


def evaluate(dataset: Path, predictions: Path, *, examples: Path | None = None) -> dict[str, Any]:
    rows = load_dataset(dataset)
    captured = _load_predictions(predictions)
    require_exact_prediction_ids(
        (row["id"] for row in rows),
        captured,
        dataset=dataset,
        predictions=predictions,
    )
    result = evaluate_turn_policy(dataset, predictions)
    result.update(
        {
            "dataset_sha256": _sha256(dataset),
            "examples_sha256": _sha256(examples) if examples is not None else None,
            "tool_metrics": _score_tools(rows, captured),
            "by_split": _score_splits(rows, captured),
            "by_family": _score_families(rows, captured),
            "raw_score_distributions": {
                disposition: _distribution(
                    [
                        float(prediction["scores"][disposition])
                        for prediction in captured.values()
                        if isinstance(prediction.get("scores"), dict)
                        and disposition in prediction["scores"]
                        and isinstance(prediction["scores"][disposition], (int, float))
                    ]
                )
                for disposition in sorted(_DISPOSITIONS)
            },
            "raw_source_score_distributions": {
                source: _distribution(
                    [
                        float(prediction["source_scores"][source])
                        for prediction in captured.values()
                        if isinstance(prediction.get("source_scores"), dict)
                        and source in prediction["source_scores"]
                    ]
                )
                for source in ("knowledge", "memory")
            },
            "latency_ms": _distribution(
                [float(prediction["latency_ms"]) for prediction in captured.values() if "latency_ms" in prediction]
            ),
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture/score the full SoCa routing cascade.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--run-local", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.58)
    parser.add_argument("--margin", type=float, default=0.04)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.run_local and args.examples is None:
        parser.error("--examples is required with --run-local")
    result = (
        run_local(
            args.dataset,
            args.examples,
            args.predictions,
            threshold=args.threshold,
            margin=args.margin,
        )
        if args.run_local
        else evaluate(args.dataset, args.predictions, examples=args.examples)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
