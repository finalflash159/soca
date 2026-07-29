"""Evaluate route encoders and aggregation policies on held-out families."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from eval.eval_full_cascade import (
    _score_families,
    _score_splits,
    _score_tools,
    load_dataset,
)
from soca.knowledge.retrievers.dense import FastEmbedModel, Model2VecModel


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if array.ndim != 2 or not np.isfinite(array).all() or np.any(norms <= 1e-12):
        raise ValueError("embedding matrix is invalid")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


def _normalise_vector(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if array.ndim != 1 or not np.isfinite(array).all() or not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("query embedding is invalid")
    return np.ascontiguousarray(array / norm, dtype=np.float32)


class _SymmetricE5:
    def __init__(self) -> None:
        delegate = FastEmbedModel(allow_download=False)
        self._model = delegate._model  # evaluation-only adapter; production stays asymmetric
        self.model_id = delegate.model_id + ":symmetric"

    def embed_documents(self, texts: tuple[str, ...]) -> np.ndarray:
        return _normalise_rows(np.asarray(list(self._model.embed(texts)), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise_vector(np.asarray(list(self._model.embed(text))[0], dtype=np.float32))


def _model(name: str):
    if name == "e5_asymmetric":
        return FastEmbedModel(allow_download=False)
    if name == "e5_symmetric":
        return _SymmetricE5()
    if name == "model2vec_symmetric":
        return Model2VecModel(allow_download=False)
    raise ValueError(f"unknown route bake-off encoder: {name}")


def _aggregate(values: np.ndarray, *, mode: str, top_k: int = 2) -> float:
    if mode == "max":
        return float(np.max(values))
    if mode == "top_k_mean":
        count = min(top_k, values.size)
        return float(np.mean(np.sort(values)[-count:]))
    if mode == "centroid":
        raise AssertionError("centroid is computed from vectors, not scores")
    raise ValueError(f"unknown aggregation: {mode}")


def _predict(
    query_vector: np.ndarray,
    example_vectors: np.ndarray,
    examples: list[dict[str, Any]],
    *,
    aggregation: str,
    threshold: float,
    margin: float,
) -> dict[str, Any]:
    raw_scores = example_vectors @ query_vector
    groups: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        groups.setdefault(str(example["disposition"]), []).append(index)

    route_scores: dict[str, float] = {}
    for disposition, indexes in groups.items():
        values = raw_scores[indexes]
        if aggregation == "centroid":
            centroid = _normalise_vector(np.mean(example_vectors[indexes], axis=0))
            route_scores[disposition] = float(centroid @ query_vector)
        else:
            route_scores[disposition] = _aggregate(values, mode=aggregation)
    ranked = sorted(route_scores.items(), key=lambda item: (-item[1], item[0]))
    top_name, top_score = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else None
    score_margin = top_score - ranked[1][1] if len(ranked) > 1 else None
    base = {
        "disposition": "unresolved",
        "tool": "none",
        "sources": [],
        "scores": {name: round(score, 6) for name, score in ranked},
        "runner_up": runner_up,
        "margin": score_margin,
    }
    if top_score < threshold or (score_margin is not None and score_margin < margin):
        base["reason"] = "below_threshold" if top_score < threshold else "ambiguous_margin"
        return base
    base["disposition"] = top_name
    base["reason"] = f"bakeoff_{top_name}"
    if top_name == "direct_tool":
        best = max(groups[top_name], key=lambda index: float(raw_scores[index]))
        base["tool"] = examples[best].get("tool", "none")
    elif top_name == "retrieval_request":
        source_scores: dict[str, float] = {}
        for index in groups[top_name]:
            for source in examples[index].get("sources", []):
                source_scores[source] = max(source_scores.get(source, -1.0), float(raw_scores[index]))
        if source_scores:
            best_source_score = max(source_scores.values())
            base["sources"] = sorted(
                source
                for source, score in source_scores.items()
                if score >= threshold and best_source_score - score <= margin
            )
    return base


def _load_examples(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError(f"{path}:{line_number}: invalid route example")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no route examples")
    return rows


def run(
    *,
    dataset: Path,
    examples_path: Path,
    output: Path,
    threshold: float,
    margin: float,
) -> dict[str, Any]:
    rows = load_dataset(dataset)
    examples = _load_examples(examples_path)
    example_texts = tuple(str(row["text"]) for row in examples)
    query_texts = tuple(str(row.get("query") or row.get("transcript")) for row in rows)
    reports: dict[str, Any] = {}
    for encoder_name in ("e5_asymmetric", "e5_symmetric", "model2vec_symmetric"):
        model_started = time.perf_counter()
        model = _model(encoder_name)
        example_vectors = _normalise_rows(model.embed_documents(example_texts))
        query_vectors = tuple(_normalise_vector(model.embed_query(text)) for text in query_texts)
        encoder_ms = (time.perf_counter() - model_started) * 1000
        for aggregation in ("max", "top_k_mean", "centroid"):
            predictions: list[dict[str, Any]] = []
            started = time.perf_counter()
            for row, query_vector in zip(rows, query_vectors, strict=True):
                prediction = _predict(
                    query_vector,
                    example_vectors,
                    examples,
                    aggregation=aggregation,
                    threshold=threshold,
                    margin=margin,
                )
                prediction["id"] = row["id"]
                predictions.append(prediction)
            capture_path = output.parent / f"{output.stem}_{encoder_name}_{aggregation}.jsonl"
            capture_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
                encoding="utf-8",
            )
            captured = {row["id"]: row for row in predictions}
            reports[f"{encoder_name}/{aggregation}"] = {
                "encoder": encoder_name,
                "aggregation": aggregation,
                "encoder_and_query_ms": encoder_ms,
                "routing_mean_ms": (time.perf_counter() - started) * 1000 / len(rows),
                "disposition_accuracy": sum(
                    row["disposition"] == captured[row["id"]]["disposition"] for row in rows
                ) / len(rows),
                "source_exact_accuracy": sum(
                    set(row.get("sources", [])) == set(captured[row["id"]].get("sources", [])) for row in rows
                ) / len(rows),
                "tool_metrics": _score_tools(rows, captured),
                "by_split": _score_splits(rows, captured),
                "by_family": _score_families(rows, captured),
                "prediction_file": str(capture_path),
            }
    result = {
        "dataset": str(dataset),
        "examples": str(examples_path),
        "threshold": threshold,
        "margin": margin,
        "reports": reports,
        "decision": "diagnostic_only_no_threshold_fit",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Bake off route encoder and aggregation policies.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.58)
    parser.add_argument("--margin", type=float, default=0.04)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        dataset=args.dataset,
        examples_path=args.examples,
        output=args.output,
        threshold=args.threshold,
        margin=args.margin,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
