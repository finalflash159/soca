"""Validate a timed SoCa edge-daemon receipt from a real Linux aarch64 SBC."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from eval.result_io import make_eval_artifact_metadata, write_json_artifact

_MIN_CAPTURE_SECONDS = 300.0
_MIN_COMPLETED_TURNS = 5
_MAX_PROCESSING_P95_MS = 32.0
_MAX_PEAK_RSS_KIB = 512 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema_version": "soca-edge-release-gate-v1",
            "gate_status": "blocked",
            "reason": "real_linux_aarch64_sbc_receipt_required",
            "checks": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("edge device receipt must be an object")
    if payload.get("schema_version") != "soca-edge-device-receipt-v1":
        raise ValueError("unsupported edge device receipt schema")

    real_arm_sbc = (
        payload.get("gate_target") == "linux_aarch64_sbc"
        and payload.get("os") == "linux"
        and payload.get("architecture") == "aarch64"
    )
    checks = {
        "real_arm_sbc": real_arm_sbc,
        "minimum_capture_duration": float(payload.get("capture_seconds", 0.0))
        >= _MIN_CAPTURE_SECONDS,
        "minimum_completed_turns": int(payload.get("completed_turns", 0))
        >= _MIN_COMPLETED_TURNS,
        "zero_dropped_capture_samples": payload.get("dropped_capture_samples") == 0,
        "stream_remained_healthy": payload.get("stream_error") is False,
        "processing_within_frame_budget": float(
            payload.get("processing_latency_p95_ms", float("inf"))
        )
        <= _MAX_PROCESSING_P95_MS,
        "memory_within_budget": isinstance(payload.get("peak_rss_kib"), int)
        and int(payload["peak_rss_kib"]) <= _MAX_PEAK_RSS_KIB,
        "target_sample_rate": payload.get("target_sample_rate") == 16_000,
        "typed_protocol": payload.get("protocol") == "soca-edge-ndjson-v1",
        "model_identities": all(
            isinstance(payload.get(field), str) and len(str(payload[field])) == 64
            for field in ("silero_sha256", "smart_turn_sha256")
        ),
    }
    if not real_arm_sbc:
        status = "blocked"
        reason = "real_linux_aarch64_sbc_receipt_required"
    elif all(checks.values()):
        status = "pass"
        reason = "all_device_gates_passed"
    else:
        status = "fail"
        reason = "arm_device_measurement_failed"
    return {
        "schema_version": "soca-edge-release-gate-v1",
        "gate_status": status,
        "reason": reason,
        "receipt_sha256": _sha256(path),
        "thresholds": {
            "min_capture_seconds": _MIN_CAPTURE_SECONDS,
            "min_completed_turns": _MIN_COMPLETED_TURNS,
            "max_processing_latency_p95_ms": _MAX_PROCESSING_P95_MS,
            "max_peak_rss_kib": _MAX_PEAK_RSS_KIB,
        },
        "checks": checks,
        "receipt": payload,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_receipt(args.receipt)
    data_files = (args.receipt,) if args.receipt.is_file() else ()
    report["artifact"] = make_eval_artifact_metadata(
        suite="edge_daemon_arm",
        run_type="benchmark",
        data_files=data_files,
        config=report["thresholds"] if "thresholds" in report else {},
        ignored_untracked_paths=(args.output,),
    ).to_dict()
    write_json_artifact(args.output, report)
    return 0 if report["gate_status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
