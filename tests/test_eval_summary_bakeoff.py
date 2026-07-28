from __future__ import annotations

from eval.eval_summary_bakeoff import _assess_artifact, _field_recall
from soca.memory.working import WorkingSummaryArtifact


def test_summary_bakeoff_field_recall_uses_structured_fields() -> None:
    artifact = WorkingSummaryArtifact(
        version=1,
        generation=1,
        source_through_sequence=1,
        summary="ok",
        decisions=("Dùng TTS local.",),
    )
    assert _field_recall({"decisions": ["Dùng TTS local."]}, artifact) == 1.0


def test_summary_bakeoff_reports_forbidden_and_unexpected_structured_values() -> None:
    artifact = WorkingSummaryArtifact(
        version=1,
        generation=1,
        source_through_sequence=1,
        summary="ok",
        decisions=("Dùng TTS A.", "Dùng TTS B."),
    )
    assessment = _assess_artifact(
        {"decisions": ["Dùng TTS B."]},
        artifact,
        forbidden_claims=["Dùng TTS A."],
    )
    assert assessment["field_recall_by_field"]["decisions"] == 1.0
    assert assessment["unexpected_items"] == ["Dùng TTS A."]
    assert assessment["forbidden_leaks"] == ["Dùng TTS A."]
