from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import Any


def _profile_result(report: dict[str, Any], profile: str) -> dict[str, Any]:
    for row in report.get("results", report.get("profiles", [])):
        if row.get("profile") == profile:
            return row
    raise ValueError(f"profile not found: {profile}")


def _ttfa_by_id(report_path: Path | dict[str, Any], profile: str) -> dict[str, float]:
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if isinstance(report_path, Path)
        else report_path
    )
    profile_result = _profile_result(report, profile)
    values: dict[str, float] = {}
    for row in profile_result.get("rows", profile_result.get("samples", [])):
        if row.get("status", "ok") not in {"ok", "partial"}:
            continue
        value = row.get("ttfa_ms")
        sample_id = row.get("id", row.get("sample_id"))
        if isinstance(sample_id, str) and isinstance(value, (int, float)) and value > 0:
            if sample_id in values:
                raise ValueError(f"duplicate voice sample id: {sample_id}")
            values[sample_id] = float(value)
    return values


def _p95(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("at least one TTFA value is required")
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return ordered[index]


def compare_reports(
    baseline: dict[str, Any] | Path,
    candidate: dict[str, Any] | Path,
    *,
    profile: str,
    max_regression: float = 0.10,
) -> dict[str, Any]:
    if isinstance(baseline, Path):
        baseline = json.loads(baseline.read_text(encoding="utf-8"))
    if isinstance(candidate, Path):
        candidate = json.loads(candidate.read_text(encoding="utf-8"))
    base = _ttfa_by_id(baseline, profile)
    current = _ttfa_by_id(candidate, profile)
    shared = sorted(set(base) & set(current))
    if len(shared) < 3:
        raise ValueError("paired TTFA comparison requires at least three shared samples")
    base_values = [base[key] for key in shared]
    candidate_values = [current[key] for key in shared]
    base_p50 = float(median(base_values))
    candidate_p50 = float(median(candidate_values))
    base_p95 = _p95(base_values)
    candidate_p95 = _p95(candidate_values)
    p50_regression = candidate_p50 / base_p50 - 1 if base_p50 else 0.0
    p95_regression = candidate_p95 / base_p95 - 1 if base_p95 else 0.0
    passed = p50_regression <= max_regression and p95_regression <= max_regression
    return {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "paired_count": len(shared),
        "profile": profile,
        "sample_count": len(shared),
        "baseline_p50_ms": base_p50,
        "candidate_p50_ms": candidate_p50,
        "intent_p50_ms": candidate_p50,
        "baseline_p95_ms": base_p95,
        "candidate_p95_ms": candidate_p95,
        "intent_p95_ms": candidate_p95,
        "p50_regression": p50_regression,
        "p95_regression": p95_regression,
        "max_allowed_regression": max_regression,
        "within_budget": passed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare paired voice TTFA reports.")
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--intent-report", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = compare_reports(args.baseline_report, args.intent_report, profile=args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result["within_budget"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
