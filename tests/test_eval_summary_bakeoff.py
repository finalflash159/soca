from __future__ import annotations

from eval.eval_summary_bakeoff import _field_recall
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
