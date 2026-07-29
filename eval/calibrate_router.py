"""Fit a release-candidate route/source calibration artifact on train+validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from eval.eval_full_cascade import _load_predictions, load_dataset
from eval.eval_source_policy import load_dataset as load_source_dataset
from eval.eval_turn_policy import require_exact_prediction_ids
from soca.core.calibration import CalibrationArtifact
from soca.core.route_catalog import source_profile

ROUTES = ("direct_tool", "retrieval_request", "smalltalk", "out_of_scope", "unresolved")
SOURCES = ("knowledge", "memory")
FIT_SPLITS = {"train", "validation"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_sha256(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(f"{row['id']}\t{row['split']}" for row in rows).encode()
    return hashlib.sha256(payload).hexdigest()


def _candidate_values(rows: list[dict[str, Any]], key: str, names: tuple[str, ...]) -> tuple[float, ...]:
    values = {0.0, 1.0}
    for row in rows:
        scores = row.get(key, {})
        if isinstance(scores, dict):
            values.update(float(scores[name]) for name in names if isinstance(scores.get(name), (int, float)))
    return tuple(sorted(values))


def _f05(precision: float, recall: float) -> float:
    beta2 = 0.25
    denominator = beta2 * precision + recall
    return (1 + beta2) * precision * recall / denominator if denominator else 0.0


def _fit_binary_threshold(
    rows: list[dict[str, Any]],
    *,
    score_key: str,
    name: str,
    positive: Callable[[dict[str, Any]], bool],
) -> float:
    positives = sum(positive(row) for row in rows)
    candidates = _candidate_values(rows, score_key, (name,))
    best_score: tuple[float, float, float, float] = (-1.0, -1.0, -1.0, -1.0)
    best_threshold = 1.0
    for threshold in candidates:
        predicted = [
            isinstance(row.get(score_key), dict)
            and float(row[score_key].get(name, float("-inf"))) >= threshold
            for row in rows
        ]
        true_positive = sum(flag and positive(row) for flag, row in zip(predicted, rows, strict=True))
        predicted_positive = sum(predicted)
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / positives if positives else 0.0
        rank = (1.0 if precision >= 0.90 else 0.0, _f05(precision, recall), precision, recall)
        if rank > best_score or (rank == best_score and threshold < best_threshold):
            best_score = rank
            best_threshold = threshold
    return best_threshold


def _route_prediction(
    row: dict[str, Any], thresholds: dict[str, float], margin: float
) -> str:
    scores = row["scores"]
    ranked = sorted(((route, float(scores[route])) for route in ROUTES), key=lambda item: (-item[1], item[0]))
    top, top_score = ranked[0]
    score_margin = top_score - ranked[1][1]
    return top if top_score >= thresholds[top] and score_margin >= margin else "unresolved"


def _source_prediction(
    row: dict[str, Any], thresholds: dict[str, float], margin: float
) -> str:
    scores = row["source_scores"]
    best_score = max(float(scores[source]) for source in SOURCES)
    selected = tuple(
        source
        for source in SOURCES
        if float(scores[source]) >= thresholds[source]
        and best_score - float(scores[source]) <= margin
    )
    return source_profile(selected)


def _fit_margin(
    rows: list[dict[str, Any]],
    *,
    thresholds: dict[str, float],
    source: bool,
) -> tuple[float, dict[str, float | int]]:
    key = "source_scores" if source else "scores"
    names = SOURCES if source else ROUTES
    candidates = {0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20}
    for row in rows:
        scores = row[key]
        ordered = sorted((float(scores[name]) for name in names), reverse=True)
        candidates.add(ordered[0] - ordered[1])
    best_margin = 0.0
    best_rank: tuple[int, float, float] = (-1, -1.0, 0.0)
    for margin in sorted(value for value in candidates if 0.0 <= value <= 1.0):
        if source:
            correct = sum(
                _source_prediction(row, thresholds, margin) == source_profile(row["sources"])
                for row in rows
            )
            score = correct / len(rows) if rows else 0.0
            rank = (correct, score, -margin)
        else:
            predictions = [_route_prediction(row, thresholds, margin) for row in rows]
            unsupported = sum(
                prediction == "direct_tool" and row["disposition"] == "out_of_scope"
                for row, prediction in zip(rows, predictions, strict=True)
            )
            correct = sum(prediction == row["disposition"] for row, prediction in zip(rows, predictions, strict=True))
            score = correct / len(rows) if rows else 0.0
            rank = (int(unsupported == 0), score, -margin)
        if rank > best_rank:
            best_rank = rank
            best_margin = margin
    if source:
        return best_margin, {"correct": best_rank[0], "total": len(rows), "accuracy": best_rank[1]}
    return best_margin, {"unsupported": 0 if best_rank[0] else 1, "total": len(rows), "accuracy": best_rank[1]}


def fit(
    *,
    route_dataset: Path,
    route_predictions: Path,
    source_dataset: Path,
    source_predictions: Path,
    output: Path,
    examples: Path,
    encoder_id: str,
    aggregation: str,
    git_sha: str,
) -> dict[str, Any]:
    route_rows = list(load_dataset(route_dataset))
    route_capture = _load_predictions(route_predictions)
    require_exact_prediction_ids((row["id"] for row in route_rows), route_capture, dataset=route_dataset, predictions=route_predictions)
    source_rows = list(load_source_dataset(source_dataset))
    source_capture = _load_predictions(source_predictions)
    require_exact_prediction_ids((row["id"] for row in source_rows), source_capture, dataset=source_dataset, predictions=source_predictions)
    route_fit = [
        {**row, "scores": route_capture[row["id"]]["scores"]}
        for row in route_rows
        if row["split"] in FIT_SPLITS
    ]
    source_fit = [
        {**row, "source_scores": source_capture[row["id"]]["source_scores"]}
        for row in source_rows
        if row["split"] in FIT_SPLITS
    ]
    route_thresholds = {
        route: _fit_binary_threshold(
            route_fit,
            score_key="scores",
            name=route,
            positive=lambda row, route=route: row["disposition"] == route,
        )
        for route in ROUTES
    }
    route_margin, route_metrics = _fit_margin(route_fit, thresholds=route_thresholds, source=False)
    source_thresholds = {
        source: _fit_binary_threshold(
            source_fit,
            score_key="source_scores",
            name=source,
            positive=lambda row, source=source: source in row["sources"],
        )
        for source in SOURCES
    }
    source_margin, source_metrics = _fit_margin(source_fit, thresholds=source_thresholds, source=True)
    artifact = CalibrationArtifact(
        version=1,
        encoder_id=encoder_id,
        aggregation=aggregation,
        route_thresholds=route_thresholds,
        route_margin=route_margin,
        source_thresholds=source_thresholds,
        source_margin=source_margin,
        metadata={
            "fit_splits": sorted(FIT_SPLITS),
            "final_test_sealed": True,
            "route_dataset_sha256": _sha256(route_dataset),
            "route_split_sha256": _split_sha256(route_rows),
            "route_examples_sha256": _sha256(examples),
            "route_predictions_sha256": _sha256(route_predictions),
            "source_dataset_sha256": _sha256(source_dataset),
            "source_split_sha256": _split_sha256(source_rows),
            "source_predictions_sha256": _sha256(source_predictions),
            "catalog_sha256": hashlib.sha256(
                json.dumps(
                    {"routes": ROUTES, "sources": SOURCES},
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "git_sha": git_sha,
            "route_fit_count": len(route_fit),
            "source_fit_count": len(source_fit),
            "route_fit_metrics": route_metrics,
            "source_fit_metrics": source_metrics,
            "catalog": list(ROUTES),
            "sources": list(SOURCES),
        },
    )
    result = artifact.to_dict()
    result["decision"] = "release_candidate_not_enabled"
    result["gate"] = {
        "unsupported_to_real_tool": "must be zero on critical hard-negative set",
        "source_ambiguous_cases": "must preserve both profile",
        "final_test": "sealed until P5/P6",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit P1 route/source calibration on train+validation only.")
    parser.add_argument("--route-dataset", type=Path, required=True)
    parser.add_argument("--route-predictions", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-predictions", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--encoder-id", default="intfloat/multilingual-e5-small")
    parser.add_argument("--aggregation", default="max")
    parser.add_argument("--git-sha", default="unknown")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fit(
        route_dataset=args.route_dataset,
        route_predictions=args.route_predictions,
        source_dataset=args.source_dataset,
        source_predictions=args.source_predictions,
        output=args.output,
        examples=args.examples,
        encoder_id=args.encoder_id,
        aggregation=args.aggregation,
        git_sha=args.git_sha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
