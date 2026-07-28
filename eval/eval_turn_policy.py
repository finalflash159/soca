"""Evaluate the disposition/source contract used by chat and voice.

Unlike the historical tool-router evaluator, this evaluator keeps the three
decisions separate: disposition, selected retrieval sources, and executable
tool.  It deliberately accepts captured predictions so it can score the exact
same runtime path for text and ASR transcripts without loading a model in CI.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

_DISPOSITIONS = {"direct_tool", "retrieval_request", "smalltalk", "out_of_scope", "unresolved"}
_SOURCES = {"knowledge", "memory"}


def _load(path: Path, *, predictions: bool = False) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError(f"{path}:{line_number}: row needs a string id")
            if predictions:
                if row.get("disposition") not in _DISPOSITIONS:
                    raise ValueError(f"{path}:{line_number}: invalid predicted disposition")
            else:
                if not isinstance(row.get("query"), str) or row.get("disposition") not in _DISPOSITIONS:
                    raise ValueError(f"{path}:{line_number}: invalid labelled turn")
                if not isinstance(row.get("family"), str) or not row["family"]:
                    raise ValueError(f"{path}:{line_number}: semantic family is required")
            sources = row.get("sources", [])
            if not isinstance(sources, list) or not set(sources) <= _SOURCES:
                raise ValueError(f"{path}:{line_number}: invalid sources")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty dataset")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"{path}: duplicate ids")
    return tuple(rows)


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(dataset: Path, predictions: Path) -> dict[str, Any]:
    """Score one captured policy run against a sealed labelled dataset."""
    expected = _load(dataset)
    actual = {row["id"]: row for row in _load(predictions, predictions=True)}
    pairs = [(row, actual[row["id"]]) for row in expected if row["id"] in actual]
    expected_dispositions = [row["disposition"] for row, _ in pairs]
    actual_dispositions = [row["disposition"] for _, row in pairs]
    exact_sources = sum(
        set(row.get("sources", [])) == set(prediction.get("sources", []))
        for row, prediction in pairs
    )
    unsupported = sum(
        row["disposition"] == "out_of_scope"
        and prediction.get("tool") not in {None, "", "none"}
        for row, prediction in pairs
    )
    out_of_scope = sum(row["disposition"] == "out_of_scope" for row, _ in pairs)
    parity_groups: dict[str, list[dict[str, Any]]] = {}
    for _, prediction in pairs:
        parity_id = prediction.get("parity_id")
        if isinstance(parity_id, str) and parity_id:
            parity_groups.setdefault(parity_id, []).append(prediction)
    parity_mismatches = sum(
        len({(item["disposition"], tuple(sorted(item.get("sources", []))), item.get("tool")) for item in group}) > 1
        for group in parity_groups.values()
        if len(group) > 1
    )
    return {
        "dataset": str(dataset),
        "prediction_source": str(predictions),
        "case_count": len(expected),
        "scored_count": len(pairs),
        "coverage": _rate(len(pairs), len(expected)),
        "disposition_accuracy": _rate(
            sum(left == right for left, right in zip(expected_dispositions, actual_dispositions, strict=True)),
            len(pairs),
        ),
        "source_exact_accuracy": _rate(exact_sources, len(pairs)),
        "unsupported_to_real_tool": {
            "count": unsupported,
            "denominator": out_of_scope,
            "rate": _rate(unsupported, out_of_scope),
        },
        "chat_voice_parity": {
            "paired_group_count": sum(len(group) > 1 for group in parity_groups.values()),
            "mismatch_count": parity_mismatches,
        },
        "confusion": {
            f"{left}->{right}": count
            for (left, right), count in sorted(
                Counter(zip(expected_dispositions, actual_dispositions, strict=True)).items()
            )
        },
    }


__all__ = ["evaluate"]
