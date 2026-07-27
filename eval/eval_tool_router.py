"""Evaluate router decisions against a small, auditable JSONL qrel set."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from soca.core.runtime import DefaultRuntimeToolRouter


def _load(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("router dataset rows must be objects")
            rows.append(value)
    if not rows:
        raise ValueError("router dataset is empty")
    return tuple(rows)


def evaluate(dataset: Path, predictions: Path | None = None) -> dict[str, Any]:
    rows = _load(dataset)
    expected = {str(row["id"]): str(row["expected_tool"]) for row in rows}
    if predictions is None:
        router = DefaultRuntimeToolRouter()
        actual: dict[str, str] = {}
        for row in rows:
            call = router.select(str(row["query"]), knowledge_limit=3)
            actual[str(row["id"])] = call.name if call is not None else "none"
    else:
        actual = {str(row["id"]): str(row["tool"]) for row in _load(predictions)}
    matches = sum(expected.get(case_id) == tool for case_id, tool in actual.items())
    confusion = Counter((expected.get(case_id, "missing"), tool) for case_id, tool in actual.items())
    return {
        "status": "ok",
        "case_count": len(rows),
        "coverage": len(actual) / len(rows),
        "accuracy": matches / len(rows),
        "confusion": {f"{left}->{right}": count for (left, right), count in sorted(confusion.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evaluate(args.dataset, args.predictions), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
