from __future__ import annotations

from soca.core.answer_validation import validate_grounded_answer
from soca.knowledge import KnowledgeCitation


def test_answer_validation_accepts_knowledge_and_memory_provenance() -> None:
    citations = (
        KnowledgeCitation("wiki/a.md", "A"),
        KnowledgeCitation("memory/b.md", "B", source="memory"),
    )
    decision = validate_grounded_answer("Theo [K1] và [M1] thì đúng.", citations)
    assert decision.status == "valid"


def test_answer_validation_reports_missing_provenance_without_blocking() -> None:
    decision = validate_grounded_answer("Một câu trả lời không nguồn.", (KnowledgeCitation("wiki/a.md", "A"),))
    assert decision.status == "missing"


def test_answer_validation_reports_partial_provenance() -> None:
    citations = (
        KnowledgeCitation("wiki/a.md", "A"),
        KnowledgeCitation("wiki/b.md", "B"),
        KnowledgeCitation("memory/c.md", "C", source="memory"),
    )

    decision = validate_grounded_answer("Theo [K1] và [M1] thì đúng.", citations)

    assert decision.status == "partial"
    assert decision.expected_labels == ("[K1]", "[K2]", "[M1]")
    assert decision.found_labels == ("[K1]", "[M1]")
    assert decision.reason == "partial_provenance_labels"


def test_answer_validation_rejects_unknown_citation_ids() -> None:
    citations = (KnowledgeCitation("wiki/a.md", "A"),)

    decision = validate_grounded_answer("Theo [K2] thì đúng.", citations)

    assert decision.status == "invalid"
    assert decision.unknown_labels == ("[K2]",)
