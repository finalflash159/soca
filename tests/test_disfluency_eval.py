from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from eval.eval_disfluency import (
    EndpointReceipt,
    ToolReceipt,
    evaluate_disfluency,
    load_materialized_cases,
    load_scenario_specs,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _spec(case_id: str, kind: str) -> dict[str, object]:
    return {
        "id": case_id,
        "language": "vi",
        "disfluency_type": kind,
        "transcript": "Ừm, tìm trong knowledge về định lý Bayes.",
        "expected_tool": "knowledge.search",
        "expected_terminal": "achieved",
    }


def test_loader_requires_all_five_types_and_verifies_audio_hash(tmp_path: Path) -> None:
    kinds = ["filler", "pause", "hesitation", "false_start", "self_correction"]
    specs_path = tmp_path / "specs.jsonl"
    _write_jsonl(specs_path, [_spec(f"case-{i}", kind) for i, kind in enumerate(kinds)])
    specs = load_scenario_specs(specs_path)

    audio_path = tmp_path / "case.wav"
    sf.write(audio_path, np.full(16_000, 0.2, dtype=np.float32), 16_000)
    digest = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    materialized = tmp_path / "materialized.json"
    materialized.write_text(
        json.dumps(
            {
                "schema_version": "soca-disfluency-audio-v1",
                "source": "reviewed_private_vietnamese_speech",
                "cases": [
                    {
                        "id": spec.case_id,
                        "wav_path": "case.wav",
                        "wav_sha256": digest,
                        "true_end_ms": 900.0,
                        "hold_spans_ms": [[300.0, 700.0]],
                    }
                    for spec in specs
                ],
            }
        ),
        encoding="utf-8",
    )

    cases = load_materialized_cases(specs, materialized)

    assert len(cases) == 5
    assert {case.spec.disfluency_type for case in cases} == set(kinds)
    assert all(case.audio_sha256 == digest for case in cases)


def test_evaluator_scores_endpoint_hold_and_actual_tool_terminal(tmp_path: Path) -> None:
    kinds = ["filler", "pause", "hesitation", "false_start", "self_correction"]
    specs_path = tmp_path / "specs.jsonl"
    _write_jsonl(specs_path, [_spec(f"case-{i}", kind) for i, kind in enumerate(kinds)])
    specs = load_scenario_specs(specs_path)
    audio_path = tmp_path / "case.wav"
    sf.write(audio_path, np.full(16_000, 0.2, dtype=np.float32), 16_000)
    digest = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    materialized = tmp_path / "materialized.json"
    materialized.write_text(
        json.dumps(
            {
                "schema_version": "soca-disfluency-audio-v1",
                "source": "reviewed_private_vietnamese_speech",
                "cases": [
                    {
                        "id": spec.case_id,
                        "wav_path": "case.wav",
                        "wav_sha256": digest,
                        "true_end_ms": 900.0,
                        "hold_spans_ms": [[300.0, 700.0]],
                    }
                    for spec in specs
                ],
            }
        ),
        encoding="utf-8",
    )
    cases = load_materialized_cases(specs, materialized)

    report = evaluate_disfluency(
        cases,
        endpoint_receipts={
            case.spec.case_id: EndpointReceipt(stopped=True, stop_ms=1_100.0)
            for case in cases
        },
        tool_receipts={
            case.spec.case_id: ToolReceipt(
                tool_name="knowledge.search",
                terminal="achieved",
                provider="openrouter",
                model="test-model",
            )
            for case in cases
        },
    )

    assert report["gate_status"] == "pass"
    assert report["scenario_count"] == 5
    assert report["endpoint_hold_accuracy"] == 1.0
    assert report["tool_terminal_accuracy"] == 1.0


def test_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    kinds = ["filler", "pause", "hesitation", "false_start", "self_correction"]
    specs_path = tmp_path / "specs.jsonl"
    _write_jsonl(specs_path, [_spec(f"case-{i}", kind) for i, kind in enumerate(kinds)])
    audio_path = tmp_path / "case.wav"
    sf.write(audio_path, np.zeros(16_000, dtype=np.float32), 16_000)
    materialized = tmp_path / "materialized.json"
    materialized.write_text(
        json.dumps(
            {
                "schema_version": "soca-disfluency-audio-v1",
                "source": "reviewed_private_vietnamese_speech",
                "cases": [
                    {
                        "id": f"case-{i}",
                        "wav_path": "case.wav",
                        "wav_sha256": "0" * 64,
                        "true_end_ms": 900.0,
                        "hold_spans_ms": [[300.0, 700.0]],
                    }
                    for i in range(5)
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="audio hash mismatch"):
        load_materialized_cases(load_scenario_specs(specs_path), materialized)
