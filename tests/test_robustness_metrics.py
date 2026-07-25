"""Tests for the RobustASR stage-decomposition metric layer (P1.1 Pha A).

Pure functions over fake diagnostics — no model, no audio, fast in CI.
"""

from __future__ import annotations

import pytest

from eval.robustness_metrics import (
    STAGE_ORDER,
    Diagnostic,
    canonical_stage,
    compute_robustness_report,
    noise_stage_breakdown,
)


def speech(final_text: str, ground_truth: str, reason: str = "") -> Diagnostic:
    return Diagnostic(
        kind="speech",
        rejection_reason=reason,
        final_text=final_text,
        ground_truth=ground_truth,
    )


def noise(final_text: str, reason: str, subtype: str = "pure") -> Diagnostic:
    return Diagnostic(
        kind="noise",
        rejection_reason=reason,
        final_text=final_text,
        subtype=subtype,
    )


def test_canonical_stage_strips_dynamic_suffix() -> None:
    assert canonical_stage("low_confidence:-1.23", "") == "low_confidence"
    assert canonical_stage("high_compression:2.40", "") == "high_compression"
    assert canonical_stage("heuristic:too_short", "") == "heuristic"
    assert canonical_stage("no_speech", "") == "no_speech"
    assert canonical_stage("empty_after_boh", "") == "empty_after_boh"


def test_canonical_stage_distinguishes_accepted_from_empty_output() -> None:
    # No reason + surviving text = hallucinated straight through (accepted bucket).
    assert canonical_stage("", "xin chào") == "accepted"
    # No reason + no text = a silent catch not attributed to any guard stage.
    assert canonical_stage("", "") == "empty_output"
    assert canonical_stage("  ", "   ") == "empty_output"


def test_every_canonical_stage_is_known() -> None:
    for reason in (
        "no_speech",
        "empty_asr",
        "low_confidence:-2.0",
        "high_compression:3.1",
        "empty_after_boh",
        "heuristic:repetition",
        "",
    ):
        assert canonical_stage(reason, "" if reason else "text") in STAGE_ORDER


def test_false_reject_rate_counts_empty_speech() -> None:
    report = compute_robustness_report(
        [
            speech("xin chào", "xin chào"),
            speech("cảm ơn", "cảm ơn"),
            speech("", "bị chặn nhầm", reason="low_confidence:-2.1"),
        ]
    )

    assert report.n_speech == 3
    assert report.false_reject_count == 1
    assert report.false_reject_rate == pytest.approx(1 / 3)


def test_hallucination_and_catch_rate_on_noise() -> None:
    report = compute_robustness_report(
        [
            noise("ừ ừ ừ", reason=""),  # hallucinated through
            noise("", reason="no_speech"),
            noise("", reason="heuristic:too_short"),
            noise("", reason="low_confidence:-3.0"),
        ]
    )

    assert report.n_noise == 4
    assert report.hallucination_count == 1
    assert report.hallucination_rate == pytest.approx(0.25)
    assert report.catch_rate == pytest.approx(0.75)


def test_noise_stage_breakdown_groups_by_stage_in_order() -> None:
    breakdown = noise_stage_breakdown(
        [
            noise("", reason="no_speech"),
            noise("", reason="no_speech"),
            noise("", reason="low_confidence:-2.5"),
            noise("", reason="heuristic:repetition"),
            noise("ảo giác", reason=""),  # accepted = leaked through
            speech("xin chào", "xin chào"),  # speech ignored by noise breakdown
        ]
    )

    assert breakdown == {"no_speech": 2, "low_confidence": 1, "heuristic": 1, "accepted": 1}
    # Keys follow pipeline order, not insertion/count order.
    assert list(breakdown) == [
        k for k in STAGE_ORDER if k in breakdown
    ]


def test_wer_cer_computed_on_accepted_speech_only() -> None:
    report = compute_robustness_report(
        [
            speech("xin chào các bạn", "xin chào các bạn"),  # accepted, perfect
            speech("", "câu bị chặn", reason="high_compression:2.9"),  # rejected
        ]
    )

    # WER/CER measured on the accepted item only → perfect, not polluted by the reject.
    assert report.wer == pytest.approx(0.0)
    assert report.cer == pytest.approx(0.0)
    assert report.false_reject_rate == pytest.approx(0.5)


def test_hallucination_rate_by_subtype_stratifies() -> None:
    report = compute_robustness_report(
        [
            noise("", reason="no_speech", subtype="pure"),
            noise("", reason="no_speech", subtype="pure"),
            noise("người ta nói", reason="", subtype="speech_like"),
            noise("", reason="heuristic:x", subtype="speech_like"),
        ]
    )

    assert report.hallucination_rate_by_subtype == {
        "pure": pytest.approx(0.0),
        "speech_like": pytest.approx(0.5),
    }


def test_empty_diagnostics_do_not_crash() -> None:
    report = compute_robustness_report([])

    assert report.n_speech == 0
    assert report.n_noise == 0
    assert report.false_reject_rate == pytest.approx(0.0)
    assert report.hallucination_rate == pytest.approx(0.0)
    assert report.catch_rate == pytest.approx(0.0)
    assert report.wer is None
    assert report.cer is None
    assert report.noise_stage_breakdown == {}
    assert report.hallucination_rate_by_subtype == {}


def test_all_rejected_speech_gives_full_false_reject_and_no_wer() -> None:
    report = compute_robustness_report(
        [
            speech("", "một", reason="no_speech"),
            speech("", "hai", reason="empty_asr"),
        ]
    )

    assert report.false_reject_rate == pytest.approx(1.0)
    # No speech survived → WER undefined rather than a misleading 0/large number.
    assert report.wer is None
    assert report.cer is None
