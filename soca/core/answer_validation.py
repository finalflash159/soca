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


@dataclass(frozen=True)
class _ShadowGroundedness:
    status: Literal["not_run", "shadow", "supported", "mixed", "unsupported"]
    score: float | None
    grounded_claim_count: int = 0
    ungrounded_claim_count: int = 0


_CITATION_LIKE_RE = re.compile(r"\[(?P<token>[KMkm][A-Za-z0-9]*)\]")
# Label-shaped tags ([K1], [M2]) are never legitimate prose, so they always go.
_LABEL_CITATION_RE = re.compile(r"(?<!\w)\[(?P<token>[KMkm]\d+)\](?!\w)")
# A bare [12] only reads as a citation when the turn actually cites something;
# in free chat it is far more likely to be content ("[100] nghìn đồng").
_NUMERIC_CITATION_RE = re.compile(r"(?<!\w)\[(?P<token>\d+)\](?!\w)")
_SOURCE_FOOTER_RE = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]+)?(?:nguồn|sources?)[ \t]*:[ \t]*(?:\n|$)"
)


def expected_citation_labels(
    citations: tuple[KnowledgeCitation, ...],
) -> tuple[str, ...]:
    labels: list[str] = []
    counters = {"knowledge": 0, "memory": 0}
    for citation in citations:
        source = citation.source if citation.source in counters else "knowledge"
        counters[source] += 1
        prefix = "K" if source == "knowledge" else "M"
        labels.append(f"[{prefix}{counters[source]}]")
    return tuple(labels)


def answer_text_without_citation_labels(
    text: str,
    citations: tuple[KnowledgeCitation, ...],
) -> str:
    allowed = frozenset(expected_citation_labels(citations))

    footer_matches = tuple(_SOURCE_FOOTER_RE.finditer(text)) if allowed else ()
    if footer_matches:
        footer = footer_matches[-1]
        if text[: footer.start()].strip():
            text = text[: footer.start()]

    cleaned = _LABEL_CITATION_RE.sub("", text)
    if allowed:
        cleaned = _NUMERIC_CITATION_RE.sub("", cleaned)
    cleaned = re.sub(r"[ \t]+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n[ \t]+\n", "\n\n", cleaned)
    return cleaned.strip()


def citation_records(
    citations: tuple[KnowledgeCitation, ...],
) -> tuple[dict[str, str | int | None], ...]:
    return tuple(
        {
            "label": label.strip("[]"),
            "path": citation.path,
            "title": citation.title,
            "line_start": citation.line_start,
            "line_end": citation.line_end,
            "source": citation.source,
        }
        for label, citation in zip(
            expected_citation_labels(citations),
            citations,
            strict=True,
        )
    )


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
    expected = expected_citation_labels(citations)
    if not expected:
        return AnswerValidationDecision("not_applicable", (), (), "no_evidence_context")
    mentioned = tuple(f"[{match.group('token')}]" for match in _CITATION_LIKE_RE.finditer(text))
    shadow = (
        _shadow_groundedness(text, citations, evidence)
        if evidence
        else _ShadowGroundedness("not_run", None)
    )
    unknown = tuple(label for label in mentioned if label not in expected)
    if unknown:
        return AnswerValidationDecision(
            "invalid",
            expected,
            tuple(label for label in mentioned if label in expected),
            "unknown_provenance_label",
            unknown_labels=unknown,
            groundedness=shadow.status,
            cited_claim_count=len(mentioned),
            groundedness_score=shadow.score,
            grounded_claim_count=shadow.grounded_claim_count,
            ungrounded_claim_count=shadow.ungrounded_claim_count,
        )
    found = tuple(label for label in expected if label in text)
    if not found:
        return AnswerValidationDecision(
            "missing",
            expected,
            (),
            "no_provenance_label",
            groundedness=shadow.status,
            uncited_claim_count=1,
            groundedness_score=shadow.score,
            grounded_claim_count=shadow.grounded_claim_count,
            ungrounded_claim_count=shadow.ungrounded_claim_count,
        )
    if len(found) != len(expected):
        return AnswerValidationDecision(
            "partial",
            expected,
            found,
            "partial_provenance_labels",
            groundedness=shadow.status,
            cited_claim_count=len(mentioned),
            uncited_claim_count=max(0, 1 - len(mentioned)),
            groundedness_score=shadow.score,
            grounded_claim_count=shadow.grounded_claim_count,
            ungrounded_claim_count=shadow.ungrounded_claim_count,
        )
    return AnswerValidationDecision(
        "valid",
        expected,
        found,
        "labels_present",
        groundedness=shadow.status,
        cited_claim_count=len(mentioned),
        groundedness_score=shadow.score,
        grounded_claim_count=shadow.grounded_claim_count,
        ungrounded_claim_count=shadow.ungrounded_claim_count,
    )


def _shadow_groundedness(
    text: str,
    citations: tuple[KnowledgeCitation, ...],
    evidence: tuple[Any, ...],
) -> _ShadowGroundedness:
    """Return a telemetry-only lexical claim/evidence signal.

    This is deliberately not an answer gate. It supplies a reproducible shadow
    metric while a held-out human/model calibration set is still being built.
    Citation provenance remains the deterministic contract.
    """
    if not evidence or not citations:
        return _ShadowGroundedness("shadow", None)
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
        return _ShadowGroundedness("shadow", None)
    score = sum(scores) / len(scores)
    grounded_count = sum(item >= 0.5 for item in scores)
    ungrounded_count = len(scores) - grounded_count
    if score >= 0.5 and ungrounded_count == 0:
        status: Literal["supported", "mixed", "unsupported"] = "supported"
    elif score > 0.0:
        status = "mixed"
    else:
        status = "unsupported"
    return _ShadowGroundedness(status, score, grounded_count, ungrounded_count)


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


__all__ = [
    "AnswerValidationDecision",
    "answer_text_without_citation_labels",
    "citation_records",
    "expected_citation_labels",
    "validate_grounded_answer",
]
