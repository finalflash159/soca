"""Post-generation provenance validation (shadow mode until calibrated)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from soca.knowledge import KnowledgeCitation


@dataclass(frozen=True)
class AnswerValidationDecision:
    status: str
    expected_labels: tuple[str, ...]
    found_labels: tuple[str, ...]
    reason: str
    unknown_labels: tuple[str, ...] = ()
    groundedness: Literal["not_run", "shadow"] = "not_run"
    cited_claim_count: int = 0
    uncited_claim_count: int = 0


_CITATION_LIKE_RE = re.compile(r"\[(?P<token>[KMkm][A-Za-z0-9]*)\]")


def validate_grounded_answer(
    text: str,
    citations: tuple[KnowledgeCitation, ...],
    *,
    evidence: tuple[Any, ...] = (),
) -> AnswerValidationDecision:
    """Check citation provenance and emit a non-blocking groundedness signal.

    Citation labels are deterministic. Claim entailment stays shadow-only until
    a held-out human calibration set establishes an acceptable judge.
    """
    expected: list[str] = []
    counters = {"knowledge": 0, "memory": 0}
    for citation in citations:
        source = citation.source if citation.source in counters else "knowledge"
        counters[source] += 1
        expected.append(f"[{'K' if source == 'knowledge' else 'M'}{counters[source]}]")
    if not expected:
        return AnswerValidationDecision("not_applicable", (), (), "no_evidence_context")
    mentioned = tuple(
        f"[{match.group('token')}]" for match in _CITATION_LIKE_RE.finditer(text)
    )
    unknown = tuple(label for label in mentioned if label not in expected)
    if unknown:
        return AnswerValidationDecision(
            "invalid",
            tuple(expected),
            tuple(label for label in mentioned if label in expected),
            "unknown_provenance_label",
            unknown_labels=unknown,
            groundedness="shadow" if evidence else "not_run",
            cited_claim_count=len(mentioned),
        )
    found = tuple(label for label in expected if label in text)
    if not found:
        return AnswerValidationDecision(
            "missing",
            tuple(expected),
            (),
            "no_provenance_label",
            groundedness="shadow" if evidence else "not_run",
            uncited_claim_count=1,
        )
    if len(found) != len(expected):
        return AnswerValidationDecision(
            "partial",
            tuple(expected),
            found,
            "partial_provenance_labels",
            groundedness="shadow" if evidence else "not_run",
            cited_claim_count=len(mentioned),
            uncited_claim_count=max(0, 1 - len(mentioned)),
        )
    return AnswerValidationDecision(
        "valid",
        tuple(expected),
        found,
        "labels_present",
        groundedness="shadow" if evidence else "not_run",
        cited_claim_count=len(mentioned),
    )


__all__ = ["AnswerValidationDecision", "validate_grounded_answer"]
