"""Post-generation provenance validation (shadow mode until calibrated)."""

from __future__ import annotations

from dataclasses import dataclass

from soca.knowledge import KnowledgeCitation


@dataclass(frozen=True)
class AnswerValidationDecision:
    status: str
    expected_labels: tuple[str, ...]
    found_labels: tuple[str, ...]
    reason: str


def validate_grounded_answer(
    text: str,
    citations: tuple[KnowledgeCitation, ...],
) -> AnswerValidationDecision:
    """Check citation provenance without attempting to judge factual truth.

    This is intentionally a shadow/report contract: evidence acceptance and
    factuality need held-out calibration before any answer is retried or blocked.
    """
    expected: list[str] = []
    counters = {"knowledge": 0, "memory": 0}
    for citation in citations:
        source = citation.source if citation.source in counters else "knowledge"
        counters[source] += 1
        expected.append(f"[{'K' if source == 'knowledge' else 'M'}{counters[source]}]")
    found = tuple(label for label in expected if label in text)
    if not expected:
        return AnswerValidationDecision("not_applicable", (), (), "no_evidence_context")
    if not found:
        return AnswerValidationDecision("missing", tuple(expected), (), "no_provenance_label")
    if len(found) != len(set(found)):
        return AnswerValidationDecision("invalid", tuple(expected), found, "duplicate_label")
    return AnswerValidationDecision("valid", tuple(expected), found, "labels_present")


__all__ = ["AnswerValidationDecision", "validate_grounded_answer"]
