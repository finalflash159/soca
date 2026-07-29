"""Post-generation provenance validation (shadow mode until calibrated)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from soca.knowledge import KnowledgeCitation
from soca.knowledge.markdown_vault import tokenize_terms


@dataclass(frozen=True)
class AnswerValidationDecision:
    status: str
    expected_labels: tuple[str, ...]
    found_labels: tuple[str, ...]
    reason: str
    unknown_labels: tuple[str, ...] = ()
    groundedness: Literal["not_run", "shadow", "supported", "mixed", "unsupported"] = "not_run"
    cited_claim_count: int = 0
    uncited_claim_count: int = 0
    grounded_claim_count: int = 0
    ungrounded_claim_count: int = 0
    groundedness_score: float | None = None


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
    groundedness, groundedness_score = (
        _shadow_groundedness(text, citations, evidence)
        if evidence
        else ("not_run", None)
    )
    unknown = tuple(label for label in mentioned if label not in expected)
    if unknown:
        return AnswerValidationDecision(
            "invalid",
            tuple(expected),
            tuple(label for label in mentioned if label in expected),
            "unknown_provenance_label",
            unknown_labels=unknown,
            groundedness=groundedness,
            cited_claim_count=len(mentioned),
            groundedness_score=groundedness_score,
        )
    found = tuple(label for label in expected if label in text)
    if not found:
        return AnswerValidationDecision(
            "missing",
            tuple(expected),
            (),
            "no_provenance_label",
            groundedness=groundedness,
            uncited_claim_count=1,
            groundedness_score=groundedness_score,
        )
    if len(found) != len(expected):
        return AnswerValidationDecision(
            "partial",
            tuple(expected),
            found,
            "partial_provenance_labels",
            groundedness=groundedness,
            cited_claim_count=len(mentioned),
            uncited_claim_count=max(0, 1 - len(mentioned)),
            groundedness_score=groundedness_score,
        )
    return AnswerValidationDecision(
        "valid",
        tuple(expected),
        found,
        "labels_present",
        groundedness=groundedness,
        cited_claim_count=len(mentioned),
        groundedness_score=groundedness_score,
    )


def _shadow_groundedness(
    text: str,
    citations: tuple[KnowledgeCitation, ...],
    evidence: tuple[Any, ...],
) -> tuple[Literal["shadow", "supported", "mixed", "unsupported"], float | None]:
    """Return a telemetry-only lexical claim/evidence signal.

    This is deliberately not an answer gate. It supplies a reproducible shadow
    metric while a held-out human/model calibration set is still being built.
    Citation provenance remains the deterministic contract.
    """
    if not evidence or not citations:
        return "shadow", None
    evidence_by_label = _evidence_by_label(citations, evidence)
    scores: list[float] = []
    for sentence in _sentences(text):
        labels = tuple(
            f"[{match.group('token')}]"
            for match in _CITATION_LIKE_RE.finditer(sentence)
            if f"[{match.group('token')}]" in evidence_by_label
        )
        if not labels:
            continue
        claim_terms = set(tokenize_terms(_CITATION_LIKE_RE.sub("", sentence)))
        if not claim_terms:
            continue
        source_terms = set()
        for label in labels:
            source_terms.update(tokenize_terms(evidence_by_label[label]))
        scores.append(len(claim_terms & source_terms) / len(claim_terms))
    if not scores:
        return "shadow", None
    score = sum(scores) / len(scores)
    if score >= 0.5:
        return "supported", score
    if score > 0.0:
        return "mixed", score
    return "unsupported", score


def _evidence_by_label(
    citations: tuple[KnowledgeCitation, ...],
    evidence: tuple[Any, ...],
) -> dict[str, str]:
    knowledge_count = sum(1 for citation in citations if citation.source != "memory")
    grouped: dict[str, list[str]] = {
        "K": [
            str(getattr(item, "snippet", "")).strip()
            for item in evidence[:knowledge_count]
            if str(getattr(item, "snippet", "")).strip()
        ],
        "M": [
            str(getattr(item, "snippet", "")).strip()
            for item in evidence[knowledge_count:]
            if str(getattr(item, "snippet", "")).strip()
        ],
    }
    labels: dict[str, str] = {}
    counters = {"K": 0, "M": 0}
    for citation in citations:
        source = "M" if citation.source == "memory" else "K"
        counters[source] += 1
        index = counters[source] - 1
        if index < len(grouped[source]):
            labels[f"[{source}{counters[source]}]"] = grouped[source][index]
    return labels


def _sentences(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in re.split(r"[.!?。！？\n]+", text) if part.strip())


__all__ = ["AnswerValidationDecision", "validate_grounded_answer"]
