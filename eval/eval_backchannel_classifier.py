"""Gate a backchannel classifier on reviewed Vietnamese audio receipts."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from eval.result_io import make_eval_artifact_metadata, write_json_artifact

Intent = Literal["backchannel", "interruption"]
_MIN_PER_CLASS = 10
_MIN_RECALL = 0.90
_MAX_P95_MS = 300.0


@dataclass(frozen=True)
class ClassificationReceipt:
    case_id: str
    expected: Intent
    predicted: Intent
    confidence: float
    latency_ms: float
    model: str
    revision: str
    audio_sha256: str


def _nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def evaluate_receipts(receipts: tuple[ClassificationReceipt, ...]) -> dict[str, Any]:
    failures: list[str] = []
    if not receipts:
        failures.append("empty_receipts")
        return {
            "schema_version": "soca-backchannel-classifier-gate-v1",
            "gate_status": "fail",
            "failures": failures,
        }
    if len({receipt.case_id for receipt in receipts}) != len(receipts):
        failures.append("duplicate_case_id")
    identities = {(receipt.model, receipt.revision) for receipt in receipts}
    if len(identities) != 1 or any(not all(identity) for identity in identities):
        failures.append("model_identity_drift")
    if len({receipt.audio_sha256 for receipt in receipts}) != len(receipts):
        failures.append("duplicate_audio_identity")
    if any(
        receipt.expected not in {"backchannel", "interruption"}
        or receipt.predicted not in {"backchannel", "interruption"}
        or not 0.0 <= receipt.confidence <= 1.0
        or receipt.latency_ms < 0.0
        or len(receipt.audio_sha256) != 64
        for receipt in receipts
    ):
        failures.append("invalid_receipt")

    counts = {
        intent: sum(receipt.expected == intent for receipt in receipts)
        for intent in ("backchannel", "interruption")
    }
    recalls = {
        intent: (
            sum(
                receipt.expected == intent and receipt.predicted == intent
                for receipt in receipts
            )
            / counts[intent]
            if counts[intent]
            else 0.0
        )
        for intent in ("backchannel", "interruption")
    }
    if any(count < _MIN_PER_CLASS for count in counts.values()):
        failures.append("insufficient_class_coverage")
    if recalls["backchannel"] < _MIN_RECALL:
        failures.append("backchannel_recall_below_threshold")
    if recalls["interruption"] < _MIN_RECALL:
        failures.append("interruption_recall_below_threshold")
    latency_p95 = _nearest_rank([receipt.latency_ms for receipt in receipts], 0.95)
    if latency_p95 > _MAX_P95_MS:
        failures.append("latency_above_threshold")
    identity = next(iter(identities)) if len(identities) == 1 else ("", "")
    return {
        "schema_version": "soca-backchannel-classifier-gate-v1",
        "gate_status": "pass" if not failures else "fail",
        "failures": failures,
        "model": identity[0],
        "revision": identity[1],
        "case_count": len(receipts),
        "case_count_by_class": counts,
        "backchannel_recall": recalls["backchannel"],
        "interruption_recall": recalls["interruption"],
        "latency_p95_ms": latency_p95,
        "thresholds": {
            "min_cases_per_class": _MIN_PER_CLASS,
            "min_recall": _MIN_RECALL,
            "max_latency_p95_ms": _MAX_P95_MS,
        },
    }


def load_receipts(path: Path) -> tuple[ClassificationReceipt, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "soca-backchannel-receipts-v1":
        raise ValueError("unsupported backchannel receipt schema")
    rows = payload.get("receipts")
    if not isinstance(rows, list):
        raise ValueError("backchannel receipts must be a list")
    return tuple(ClassificationReceipt(**row) for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipts = load_receipts(args.receipts)
    report = evaluate_receipts(receipts)
    report["artifact"] = make_eval_artifact_metadata(
        suite="backchannel_classifier_vi",
        run_type="benchmark",
        data_files=(args.receipts,),
        config=report.get("thresholds", {}),
        ignored_untracked_paths=(args.output,),
    ).to_dict()
    report["receipt_inventory"] = [asdict(receipt) for receipt in receipts]
    write_json_artifact(args.output, report)
    return 0 if report["gate_status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
