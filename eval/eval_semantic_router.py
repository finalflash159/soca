"""Score semantic-router predictions without requiring a provider call."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _rows(path: Path) -> tuple[dict[str, Any], ...]:
    with path.open(encoding="utf-8") as handle:
        values = tuple(json.loads(line) for line in handle if line.strip())
    if not values or not all(isinstance(value, dict) for value in values):
        raise ValueError("semantic router JSONL must contain objects")
    return values


def evaluate(dataset: Path, predictions: Path | None = None) -> dict[str, float | int]:
    rows = _rows(dataset)
    if predictions is None:
        return {"case_count": len(rows), "coverage": 0.0, "accuracy": 0.0}
    predicted = {str(row["id"]): str(row["tool"]) for row in _rows(predictions)}
    correct = sum(str(row.get("expected_tool")) == predicted.get(str(row["id"])) for row in rows)
    return {
        "case_count": len(rows),
        "coverage": len(predicted) / len(rows),
        "accuracy": correct / len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evaluate(args.dataset, args.predictions), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
