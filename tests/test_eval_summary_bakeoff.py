from __future__ import annotations

from eval.eval_summary_bakeoff import _assess_artifact, _concept_assessment, _field_recall
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
        required_facts=[{"field": "decisions", "anchors": ["TTS B"]}],
        forbidden_claims=["Dùng TTS A."],
    )
    assert assessment["exact_field_recall_by_field"]["decisions"] == 1.0
    assert assessment["required_fact_recall"] == 1.0
    assert assessment["unexpected_items"] == ["Dùng TTS A."]
    assert assessment["forbidden_surface_matches"] == ["Dùng TTS A."]
    assert assessment["negative_state_case"] is False


def test_summary_bakeoff_requires_negative_cases_to_leave_no_state() -> None:
    artifact = WorkingSummaryArtifact(
        version=1,
        generation=1,
        source_through_sequence=1,
        summary="Đã ghi nhận.",
    )
    assessment = _assess_artifact(
        {"summary": ""},
        artifact,
        required_facts=[],
        forbidden_claims=[],
    )
    assert assessment["negative_state_case"] is True
    assert assessment["negative_state_clean"] is False


def test_summary_bakeoff_concept_assessment_allows_paraphrase() -> None:
    artifact = WorkingSummaryArtifact(
        version=1,
        generation=1,
        source_through_sequence=1,
        summary="ok",
        open_items=("Cần benchmark cold process trên máy 16 GiB.",),
    )
    assessment = _concept_assessment(
        [
            {
                "field": "open_items",
                "anchors": ["benchmark cold-process", "máy 16 GiB"],
            }
        ],
        artifact,
    )
    assert assessment["required_fact_recall"] == 0.0

    assessment = _concept_assessment(
        [{"field": "open_items", "anchors": ["benchmark", "16 GiB"]}],
        artifact,
    )
    assert assessment["required_fact_recall"] == 1.0
