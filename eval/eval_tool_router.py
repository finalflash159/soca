"""Evaluate deterministic/captured tool-router decisions on the v2 JSONL set."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from soca.core.runtime import DefaultRuntimeToolRouter


def _load(path: Path, *, require_labels: bool = True) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"router dataset row {line_number} must be an object")
            if not value.get("id") or (require_labels and not value.get("expected_tool")):
                raise ValueError(f"router dataset row {line_number} is missing labels")
            if not require_labels and not value.get("tool"):
                raise ValueError(f"router prediction row {line_number} is missing tool")
            if require_labels and not (value.get("query") or value.get("text")):
                raise ValueError(f"router dataset row {line_number} is missing query/text")
            rows.append(value)
    if not rows:
        raise ValueError("router dataset is empty")
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("router dataset IDs must be unique")
    return tuple(rows)


def _query(row: dict[str, Any]) -> str:
    return str(row.get("query") or row.get("text") or "")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((percentile / 100.0) * len(ordered))))
    return ordered[index]


def _classification_metrics(
    expected: list[str],
    actual: list[str],
) -> dict[str, float | int]:
    tool_labels = sorted({label for label in (*expected, *actual) if label != "none"})
    true_positive = sum(left == right and right != "none" for left, right in zip(expected, actual, strict=True))
    false_positive = sum(right != "none" and left != right for left, right in zip(expected, actual, strict=True))
    false_negative = sum(left != "none" and left != right for left, right in zip(expected, actual, strict=True))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    per_tool: dict[str, dict[str, float | int]] = {}
    for label in tool_labels:
        tp = sum(left == label and right == label for left, right in zip(expected, actual, strict=True))
        fp = sum(left != label and right == label for left, right in zip(expected, actual, strict=True))
        fn = sum(left == label and right != label for left, right in zip(expected, actual, strict=True))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per_tool[label] = {
            "precision": p,
            "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "support": sum(item == label for item in expected),
        }
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": sum(left == right for left, right in zip(expected, actual, strict=True)) / len(expected),
        "per_tool": per_tool,
    }


def _score_rows(rows: tuple[dict[str, Any], ...], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scored: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        prediction = predictions.get(str(row["id"]))
        if prediction is not None:
            scored.append((row, prediction))
    expected = [str(row["expected_tool"]) for row, _ in scored]
    actual = [str(prediction.get("tool", "none")) for _, prediction in scored]
    metrics = _classification_metrics(expected, actual) if scored else _classification_metrics(["none"], ["none"])
    latencies = [float(prediction["latency_ms"]) for _, prediction in scored if "latency_ms" in prediction]
    none_count = sum(label == "none" for label in expected)
    false_triggers = sum(
        left == "none" and right != "none"
        for left, right in zip(expected, actual, strict=True)
    )

    slices: dict[str, dict[str, Any]] = {}
    for slice_name in sorted({str(row.get("slice", "unlabelled")) for row, _ in scored}):
        slice_pairs = [(row, prediction) for row, prediction in scored if str(row.get("slice", "unlabelled")) == slice_name]
        slice_expected = [str(row["expected_tool"]) for row, _ in slice_pairs]
        slice_actual = [str(prediction.get("tool", "none")) for _, prediction in slice_pairs]
        slice_metrics = _classification_metrics(slice_expected, slice_actual)
        slice_none = sum(label == "none" for label in slice_expected)
        slice_false = sum(
            left == "none" and right != "none"
            for left, right in zip(slice_expected, slice_actual, strict=True)
        )
        slices[slice_name] = {
            "case_count": len(slice_pairs),
            **slice_metrics,
            "false_trigger_rate": slice_false / slice_none if slice_none else 0.0,
        }

    return {
        "case_count": len(rows),
        "scored_count": len(scored),
        "coverage": len(scored) / len(rows),
        **metrics,
        "false_trigger_rate": false_triggers / none_count if none_count else 0.0,
        "false_trigger_count": false_triggers,
        "latency_ms": {
            "count": len(latencies),
            "mean": mean(latencies) if latencies else 0.0,
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
        },
        "slices": slices,
        "confusion": {
            f"{left}->{right}": count
            for (left, right), count in sorted(Counter(zip(expected, actual, strict=True)).items())
        },
    }


def evaluate(
    dataset: Path,
    predictions: Path | None = None,
    *,
    tier: str = "deterministic",
) -> dict[str, Any]:
    rows = _load(dataset)
    captured: dict[str, dict[str, Any]] = {}
    if predictions is None:
        if tier != "deterministic":
            return _score_rows(rows, captured)
        router = DefaultRuntimeToolRouter()
        for row in rows:
            started = time.perf_counter()
            call = router.select(_query(row), knowledge_limit=3)
            elapsed = (time.perf_counter() - started) * 1000
            captured[str(row["id"])] = {
                "tool": call.name if call is not None else "none",
                "tier": router.last_tier,
                "reason": getattr(router.last_decision, "reason", "no_match"),
                "latency_ms": elapsed,
            }
    else:
        for row in _load(predictions, require_labels=False):
            captured[str(row["id"])] = row
    result = _score_rows(rows, captured)
    result["tier"] = tier
    result["prediction_source"] = "deterministic_live" if predictions is None else str(predictions)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--tier", choices=["deterministic", "semantic", "llm", "cascade"], default="deterministic")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evaluate(args.dataset, args.predictions, tier=args.tier), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
