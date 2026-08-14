from __future__ import annotations

import json
from pathlib import Path

from eval.eval_edge_daemon import evaluate_receipt


def _receipt(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "soca-edge-device-receipt-v1",
        "gate_target": "linux_aarch64_sbc",
        "os": "linux",
        "architecture": "aarch64",
        "device": "USB microphone",
        "source_sample_rate": 48_000,
        "target_sample_rate": 16_000,
        "capture_seconds": 300.5,
        "completed_turns": 5,
        "dropped_capture_samples": 0,
        "stream_error": False,
        "processing_latency_p95_ms": 8.5,
        "peak_rss_kib": 120_000,
        "silero_sha256": "a" * 64,
        "smart_turn_sha256": "b" * 64,
        "protocol": "soca-edge-ndjson-v1",
    }
    payload.update(overrides)
    return payload


def test_linux_aarch64_device_receipt_closes_gate(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(_receipt()), encoding="utf-8")

    report = evaluate_receipt(path)

    assert report["gate_status"] == "pass"
    assert report["checks"]["real_arm_sbc"] is True
    assert report["receipt_sha256"]


def test_non_arm_run_is_blocked_not_relabelled_as_device_evidence(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(
        json.dumps(_receipt(os="macos", architecture="aarch64")),
        encoding="utf-8",
    )

    report = evaluate_receipt(path)

    assert report["gate_status"] == "blocked"
    assert report["reason"] == "real_linux_aarch64_sbc_receipt_required"


def test_arm_measurement_fails_when_audio_drops_or_latency_exceeds_frame(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(
        json.dumps(
            _receipt(dropped_capture_samples=2, processing_latency_p95_ms=40.0)
        ),
        encoding="utf-8",
    )

    report = evaluate_receipt(path)

    assert report["gate_status"] == "fail"
    assert report["checks"]["zero_dropped_capture_samples"] is False
    assert report["checks"]["processing_within_frame_budget"] is False
