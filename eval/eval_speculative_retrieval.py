"""Gate paired visible-latency receipts for speculative knowledge retrieval."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from eval.result_io import make_eval_artifact_metadata, write_json_artifact

CacheStatus = Literal["hit", "miss"]
_MIN_CASES = 20
_MIN_HIT_RATE = 0.80


@dataclass(frozen=True)
class SpeculativeLatencyReceipt:
    turn_id: str
    query_sha256: str
    source_identity: str
    baseline_result_sha256: str
    speculative_result_sha256: str
    baseline_visible_ms: float
    speculative_visible_ms: float
    prefetch_lead_ms: float
    cache_status: CacheStatus
    controller_terminal: str
    verification_passed: bool


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def evaluate_receipts(
    receipts: tuple[SpeculativeLatencyReceipt, ...],
) -> dict[str, Any]:
    reasons: list[str] = []
    if len(receipts) < _MIN_CASES:
        reasons.append("insufficient_case_count")
    if len({receipt.turn_id for receipt in receipts}) != len(receipts):
        reasons.append("duplicate_turn_id")
    invalid = any(
        not receipt.turn_id.strip()
        or len(receipt.query_sha256) != 64
        or not receipt.source_identity.strip()
        or len(receipt.baseline_result_sha256) != 64
        or len(receipt.speculative_result_sha256) != 64
        or not math.isfinite(receipt.baseline_visible_ms)
        or not math.isfinite(receipt.speculative_visible_ms)
        or not math.isfinite(receipt.prefetch_lead_ms)
        or receipt.baseline_visible_ms < 0.0
        or receipt.speculative_visible_ms < 0.0
        or receipt.prefetch_lead_ms < 0.0
        or receipt.cache_status not in {"hit", "miss"}
        for receipt in receipts
    )
    if invalid:
        reasons.append("invalid_receipt")
    if any(
        receipt.baseline_result_sha256 != receipt.speculative_result_sha256
        for receipt in receipts
    ):
        reasons.append("result_mismatch")
    if any(
        receipt.controller_terminal != "achieved" or not receipt.verification_passed
        for receipt in receipts
    ):
        reasons.append("controller_or_verification_failed")

    baseline = [receipt.baseline_visible_ms for receipt in receipts]
    speculative = [receipt.speculative_visible_ms for receipt in receipts]
    saved = [before - after for before, after in zip(baseline, speculative, strict=True)]
    hit_rate = (
        sum(receipt.cache_status == "hit" for receipt in receipts) / len(receipts)
        if receipts
        else 0.0
    )
    if hit_rate < _MIN_HIT_RATE:
        reasons.append("cache_hit_rate_below_threshold")
    median_saved = statistics.median(saved) if saved else None
    if median_saved is None or median_saved <= 0.0:
        reasons.append("median_visible_latency_not_improved")
    baseline_p95 = _p95(baseline) if baseline else None
    speculative_p95 = _p95(speculative) if speculative else None
    if (
        baseline_p95 is None
        or speculative_p95 is None
        or speculative_p95 > baseline_p95
    ):
        reasons.append("visible_latency_p95_regressed")

    return {
        "schema_version": "soca-speculative-retrieval-gate-v1",
        "gate_status": "pass" if not reasons else "fail",
        "reasons": reasons,
        "case_count": len(receipts),
        "cache_hit_rate": hit_rate,
        "baseline_visible_p50_ms": statistics.median(baseline) if baseline else None,
        "baseline_visible_p95_ms": baseline_p95,
        "speculative_visible_p50_ms": statistics.median(speculative) if speculative else None,
        "speculative_visible_p95_ms": speculative_p95,
        "median_visible_latency_saved_ms": median_saved,
        "thresholds": {
            "minimum_case_count": _MIN_CASES,
            "minimum_cache_hit_rate": _MIN_HIT_RATE,
            "median_visible_latency_saved_ms": ">0",
            "speculative_p95_must_not_exceed_baseline": True,
        },
    }


def load_receipts(path: Path) -> tuple[SpeculativeLatencyReceipt, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "soca-speculative-retrieval-receipts-v1":
        raise ValueError("unsupported speculative retrieval receipt schema")
    rows = payload.get("receipts")
    if not isinstance(rows, list):
        raise ValueError("speculative retrieval receipts must be a list")
    return tuple(SpeculativeLatencyReceipt(**row) for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipts = load_receipts(args.receipts)
    report = evaluate_receipts(receipts)
    report["artifact"] = make_eval_artifact_metadata(
        suite="speculative_retrieval_latency",
        run_type="benchmark",
        data_files=(args.receipts,),
        config=report["thresholds"],
        ignored_untracked_paths=(args.output,),
    ).to_dict()
    report["receipt_inventory"] = [asdict(receipt) for receipt in receipts]
    write_json_artifact(args.output, report)
    return 0 if report["gate_status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
