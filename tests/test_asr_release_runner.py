from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from eval.asr_release_runner import (
    BenchmarkItem,
    GuardThresholds,
    ReleaseBenchmarkError,
    collect_predictions,
    context_echo_screening,
    derive_thresholds,
    evaluate_context_echo_labels,
    evaluate_release_quality_gates,
    guard_rejection,
    summarize_predictions,
    write_calibration_artifact,
)
from soca.asr.context import ASRContextBuilder, ASRContextSourceRecord
from soca.asr.result import ASRResult


class FakeDetector:
    def detect(self, audio: np.ndarray):
        return SimpleNamespace(
            has_speech=bool(np.any(audio)),
            speech_audio=audio,
            speech_duration_ms=len(audio) / 16,
            vad_latency_ms=2.0,
        )


class FakeBackend:
    model_key = "fake"
    supports_avg_logprob = True

    def transcribe(self, audio, max_new_tokens=128, *, context=None):
        del max_new_tokens
        text = "xin chào thế giới" if context else "xin chào"
        return ASRResult(
            text=text,
            latency_ms=10.0,
            audio_duration_ms=len(audio) / 16,
            rtf=0.1,
            avg_logprob=-0.2,
        )

    def runtime_metadata(self, max_new_tokens=128):
        return {"max_new_tokens": max_new_tokens}

    def close(self):
        return None


def test_collect_predictions_runs_vad_once_and_context_variants(tmp_path) -> None:
    import soundfile as sf

    audio_path = tmp_path / "speech.wav"
    sf.write(audio_path, np.ones(1_600, dtype=np.float32) * 0.1, 16_000)
    item = BenchmarkItem(
        item_id="speech-1",
        dataset="fleurs_vi",
        path=audio_path,
        reference="xin chào thế giới",
        cohort="calibration",
    )

    builder = ASRContextBuilder()
    snapshots = {
        "empty": builder.build(()),
        "production_catalog": builder.build([ASRContextSourceRecord("thế giới", "test catalog")]),
    }
    rows = collect_predictions(
        items=[item],
        backend=FakeBackend(),
        detector=FakeDetector(),
        context_variants=("empty", "production_catalog"),
        context_factory=snapshots.__getitem__,
        max_new_tokens=128,
    )

    assert [row.context_variant for row in rows] == ["empty", "production_catalog"]
    assert [row.text for row in rows] == ["xin chào", "xin chào thế giới"]
    assert rows[0].audio_duration_ms == 100.0


def test_collect_predictions_resumes_without_repeating_checkpointed_inference(
    tmp_path,
) -> None:
    import soundfile as sf

    audio_path = tmp_path / "speech.wav"
    sf.write(audio_path, np.ones(1_600, dtype=np.float32) * 0.1, 16_000)
    item = BenchmarkItem(
        item_id="speech-1",
        dataset="fleurs_vi",
        path=audio_path,
        reference="xin chào thế giới",
        cohort="calibration",
    )
    builder = ASRContextBuilder()
    snapshots = {
        "empty": builder.build(()),
        "production_catalog": builder.build([ASRContextSourceRecord("thế giới", "test catalog")]),
    }
    checkpoint = collect_predictions(
        items=[item],
        backend=FakeBackend(),
        detector=FakeDetector(),
        context_variants=("empty",),
        context_factory=snapshots.__getitem__,
        max_new_tokens=128,
    )[0]
    written = []

    rows = collect_predictions(
        items=[item],
        backend=FakeBackend(),
        detector=FakeDetector(),
        context_variants=("empty", "production_catalog"),
        context_factory=snapshots.__getitem__,
        max_new_tokens=128,
        existing_rows=[checkpoint],
        row_sink=written.append,
    )

    assert [row.context_variant for row in rows] == ["empty", "production_catalog"]
    assert written == [rows[1]]


def test_collect_predictions_rejects_duplicate_checkpoint_rows(tmp_path) -> None:
    import soundfile as sf

    audio_path = tmp_path / "speech.wav"
    sf.write(audio_path, np.ones(160, dtype=np.float32) * 0.1, 16_000)
    item = BenchmarkItem(
        item_id="speech-1",
        dataset="fleurs_vi",
        path=audio_path,
        reference="xin chào",
        cohort="calibration",
    )
    checkpoint = _row("speech-1", "fleurs_vi", "calibration", -0.2, 1.0)

    with pytest.raises(ReleaseBenchmarkError, match="duplicate row"):
        collect_predictions(
            items=[item],
            backend=FakeBackend(),
            detector=FakeDetector(),
            context_variants=("empty",),
            context_factory=lambda _variant: ASRContextBuilder().build(()),
            max_new_tokens=128,
            existing_rows=[checkpoint, checkpoint],
        )


def test_thresholds_are_selected_only_from_calibration_speech() -> None:
    rows = [
        _row("speech-a", "fleurs_vi", "calibration", -0.7, 1.2),
        _row("speech-b", "fleurs_vi", "calibration", -0.2, 1.4),
        _row("noise", "non_speech", "calibration", -0.1, 9.0),
        _row("holdout", "fleurs_vi", "holdout", -9.0, 20.0),
    ]

    thresholds = derive_thresholds(
        rows,
        max_speech_false_reject_rate=0.0,
        compression_floor=2.4,
        context_echo_min_contiguous_tokens=4,
    )

    assert thresholds.min_avg_logprob == -0.7
    assert thresholds.max_compression_ratio == 2.4


def test_summary_keeps_hard_negative_and_quality_separate() -> None:
    thresholds = GuardThresholds(-0.5, 2.4, 4)
    rows = [
        _row("speech", "fleurs_vi", "holdout", -0.2, 1.0, reference="xin chào"),
        _row("noise", "non_speech", "holdout", -0.2, 1.0, reference=""),
    ]

    summary = summarize_predictions(rows, thresholds)

    assert summary["fleurs_vi"]["empty"]["quality"]["wer"] == 0.0
    assert summary["non_speech"]["empty"]["hard_negative"]["false_accept_rate"] == 1.0
    assert guard_rejection(rows[0], thresholds) is None


def test_calibration_artifact_is_canonical_and_atomic(tmp_path) -> None:
    import json

    path = tmp_path / "calibration.json"
    write_calibration_artifact(path, {"b" * 64: {"value": 2}, "a" * 64: {"value": 1}})

    payload = json.loads(path.read_text())
    assert list(payload["calibrations"]) == ["a" * 64, "b" * 64]
    assert not list(tmp_path.glob("*.tmp"))


def _row(
    item_id: str,
    dataset: str,
    cohort: str,
    avg_logprob: float,
    ratio: float,
    *,
    reference: str = "xin chào",
):
    from eval.asr_release_runner import RawPrediction

    return RawPrediction(
        item_id=item_id,
        dataset=dataset,
        cohort=cohort,
        context_variant="empty",
        context_digest="digest",
        context_provenance=(),
        reference=reference,
        text="xin chào",
        vad_has_speech=True,
        speech_duration_ms=1000.0,
        vad_latency_ms=1.0,
        asr_latency_ms=10.0,
        audio_duration_ms=1000.0,
        rtf=0.01,
        avg_logprob=avg_logprob,
        avg_logprob_reliable=True,
        compression_ratio=ratio,
        context_unique_token_overlap=0.0,
        context_max_contiguous_tokens=0,
        exact_context_echo=False,
        context_echo_rejected=False,
        english_reference_indices=(),
    )


def test_context_echo_screening_does_not_claim_manual_ground_truth() -> None:
    row = _row("echo", "fleurs_vi", "holdout", -0.2, 1.0)
    row = replace(
        row,
        context_variant="production_catalog",
        context_unique_token_overlap=0.8,
        exact_context_echo=True,
    )

    report = context_echo_screening(rows=[row], thresholds=[0.6], minimum_unique_tokens=2)

    assert report["evidence_class"] == "automatic_screening_requires_manual_review"
    assert report["threshold_sweep"][0]["exact_echo_true_positive"] == 1


def test_manual_context_echo_review_scores_contiguous_policy() -> None:
    copied = replace(
        _row("echo", "non_speech", "holdout", -0.2, 1.0, reference=""),
        context_variant="production_catalog",
        context_max_contiguous_tokens=8,
    )
    legitimate = replace(
        _row("term", "private_codeswitch", "holdout", -0.2, 1.0),
        context_variant="production_catalog",
        context_max_contiguous_tokens=2,
    )

    report = evaluate_context_echo_labels(
        [copied, legitimate],
        [
            {"dataset": "non_speech", "item_id": "echo", "context_echo": True},
            {
                "dataset": "private_codeswitch",
                "item_id": "term",
                "context_echo": False,
            },
        ],
        minimum_contiguous_tokens=4,
    )

    assert report["true_positive"] == 1
    assert report["true_negative"] == 1
    assert report["context_echo_false_accept_rate"] == 0.0
    assert report["context_echo_false_reject_rate"] == 0.0


def test_release_quality_gate_reports_pending_non_quality_evidence() -> None:
    summaries = {
        "release": {
            "private_codeswitch": {
                "production_catalog": {
                    "quality": {"wer": 0.3},
                    "code_switch": {"cs_wer": 0.3},
                    "latency_ms": {"rtf_p95": 0.2},
                }
            },
            "fleurs_vi": {"empty": {"quality": {"wer": 0.2}, "rejected": 1, "items": 100}},
            "non_speech": {"production_catalog": {"hard_negative": {"false_accept_rate": 0.05}}},
        },
        "control": {
            "private_codeswitch": {
                "empty": {"quality": {"wer": 0.5}, "code_switch": {"cs_wer": 0.7}}
            },
            "fleurs_vi": {"empty": {"quality": {"wer": 0.19}}},
        },
    }
    gates = {
        "private_codeswitch_min_absolute_cs_wer_improvement": 0.15,
        "private_codeswitch_max_absolute_wer_regression": 0.05,
        "public_vi_max_absolute_wer_regression": 0.05,
        "speech_false_reject_rate_max": 0.02,
        "hard_negative_false_accept_rate_max": 0.1,
        "final_rtf_p95_max": 0.5,
    }

    decision = evaluate_release_quality_gates(
        summaries,
        release_key="release",
        control_key="control",
        gates=gates,
    )

    assert decision["status"] == "incomplete"
    assert not decision["failed"]
    assert "full_voice_remote_llm_tts_trajectory" in decision["pending"]
