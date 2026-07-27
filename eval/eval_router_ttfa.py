"""Compare voice/text latency reports and enforce a regression budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _latency(report: dict[str, Any]) -> float:
    for key in ("ttfa_p50_ms", "first_tts_latency_ms", "latency_p50_ms"):
        value = report.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    metrics = report.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("latency_p50_ms"), (int, float)):
        return float(metrics["latency_p50_ms"])
    raise ValueError("report has no supported latency field")


def compare(baseline_path: Path, candidate_path: Path, budget_pct: float) -> dict[str, float | bool]:
    baseline = _latency(json.loads(baseline_path.read_text(encoding="utf-8")))
    candidate = _latency(json.loads(candidate_path.read_text(encoding="utf-8")))
    delta_pct = ((candidate - baseline) / baseline * 100.0) if baseline else 0.0
    return {
        "baseline_ms": baseline,
        "candidate_ms": candidate,
        "delta_pct": delta_pct,
        "within_budget": delta_pct <= budget_pct,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--budget-pct", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline_report, args.candidate_report, args.budget_pct)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["within_budget"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
