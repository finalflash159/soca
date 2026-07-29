"""Evaluate multi-label knowledge/memory source selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from eval.eval_full_cascade import _build_router
from eval.eval_turn_policy import require_exact_prediction_ids

_PROFILES = {"knowledge", "memory", "both", "neither"}
_SOURCES = {"knowledge", "memory"}
_SPLITS = {"train", "validation", "test"}


def _profile(sources: list[str] | tuple[str, ...]) -> str:
    selected = set(sources)
    if selected == {"knowledge"}:
        return "knowledge"
    if selected == {"memory"}:
        return "memory"
    if selected == _SOURCES:
        return "both"
    return "neither"


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
            row_id, family, split = row.get("id"), row.get("family"), row.get("split")
            sources = row.get("sources", [])
            expected = row.get("expected_source_profile")
            if (
                not isinstance(row_id, str)
                or not row_id.strip()
                or row_id in ids
                or not isinstance(family, str)
                or split not in _SPLITS
                or not isinstance(row.get("query"), str)
                or not row["query"].strip()
                or row.get("disposition") != "retrieval_request"
                or not isinstance(sources, list)
                or not set(sources) <= _SOURCES
                or expected not in _PROFILES
                or _profile(sources) != expected
            ):
                raise ValueError(f"{path}:{line_number}: invalid source-policy row")
            if family in families and families[family] != split:
                raise ValueError(f"{path}:{line_number}: family crosses splits: {family}")
            ids.add(row_id)
            families[family] = str(split)
            rows.append(row)
    if not rows or {row["split"] for row in rows} != _SPLITS:
        raise ValueError(f"{path}: source dataset needs all three splits")
    return tuple(rows)


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
            sources = row.get("sources", [])
            if not isinstance(sources, list) or not set(sources) <= _SOURCES:
                raise ValueError(f"{path}:{line_number}: invalid sources")
            result[row["id"]] = row
    return result


def _wilson(successes: int, total: int) -> dict[str, float | int]:
    if total == 0:
        return {"successes": successes, "total": total, "rate": 0.0, "lower": 0.0, "upper": 0.0}
    z = 1.959963984540054
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


def evaluate(dataset: Path, predictions: Path) -> dict[str, Any]:
    rows = load_dataset(dataset)
    captured = _load_predictions(predictions)
    require_exact_prediction_ids(
        (row["id"] for row in rows),
        captured,
        dataset=dataset,
        predictions=predictions,
    )
    pairs = [(row, captured[row["id"]]) for row in rows if row["id"] in captured]
    exact = sum(set(row["sources"]) == set(prediction.get("sources", [])) for row, prediction in pairs)
    by_profile: dict[str, dict[str, float | int]] = {}
    for profile in sorted(_PROFILES):
        profile_pairs = [(row, prediction) for row, prediction in pairs if row["expected_source_profile"] == profile]
        true_positive = sum(
            set(row["sources"]) == set(prediction.get("sources", []))
            and _profile(prediction.get("sources", [])) == profile
            for row, prediction in profile_pairs
        )
        predicted_count = sum(_profile(prediction.get("sources", [])) == profile for _, prediction in pairs)
        expected_count = sum(row["expected_source_profile"] == profile for row, _ in pairs)
        by_profile[profile] = {
            "count": len(profile_pairs),
            "correct": true_positive,
            "precision": true_positive / predicted_count if predicted_count else 0.0,
            "recall": true_positive / expected_count if expected_count else 0.0,
        }
    return {
        "dataset": str(dataset),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "prediction_source": str(predictions),
        "case_count": len(rows),
        "scored_count": len(pairs),
        "coverage": len(pairs) / len(rows),
        "source_exact_accuracy": _wilson(exact, len(pairs)),
        "by_profile": by_profile,
        "confusion": {
            f"{expected}->{actual}": count
            for (expected, actual), count in Counter(
                (row["expected_source_profile"], _profile(prediction.get("sources", [])))
                for row, prediction in pairs
            ).items()
        },
    }


def run_local(dataset: Path, examples: Path, predictions: Path, *, threshold: float, margin: float) -> dict[str, Any]:
    rows = load_dataset(dataset)
    router = _build_router(examples, threshold=threshold, margin=margin)
    captured: list[dict[str, Any]] = []
    for row in rows:
        started = time.perf_counter()
        router.select(row["query"], knowledge_limit=3)
        decision = router.last_decision
        captured.append(
            {
                "id": row["id"],
                "disposition": decision.disposition,
                "sources": list(decision.sources),
                "source_scores": decision.source_scores,
                "scores": decision.scores,
                "tier": getattr(router, "last_tier", "none"),
                "reason": decision.reason,
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
        )
    predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in captured), encoding="utf-8")
    return evaluate(dataset, predictions)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate multi-label retrieval source selection.")
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
        run_local(args.dataset, args.examples, args.predictions, threshold=args.threshold, margin=args.margin)
        if args.run_local
        else evaluate(args.dataset, args.predictions)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
