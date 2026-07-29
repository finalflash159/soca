from __future__ import annotations

from soca.core.answer_validation import validate_grounded_answer
from soca.knowledge import KnowledgeCitation, KnowledgeDocument, KnowledgeHit


def test_answer_validation_accepts_knowledge_and_memory_provenance() -> None:
    citations = (
        KnowledgeCitation("wiki/a.md", "A"),
        KnowledgeCitation("memory/b.md", "B", source="memory"),
    )
    decision = validate_grounded_answer("Theo [K1] và [M1] thì đúng.", citations)
    assert decision.status == "valid"


def test_answer_validation_records_shadow_groundedness_against_selected_evidence() -> None:
    citations = (KnowledgeCitation("wiki/bayes.md", "Bayes"),)
    evidence = (
        KnowledgeHit(
            KnowledgeDocument(
                "wiki/bayes.md",
                "wiki/bayes.md",
                "Bayes",
                "Định lý Bayes cập nhật xác suất bằng bằng chứng.",
            ),
            score=0.9,
            snippet="Định lý Bayes cập nhật xác suất bằng bằng chứng.",
            retrieval_backend="dense",
            dense_score=0.9,
        ),
    )

    decision = validate_grounded_answer(
        "Định lý Bayes cập nhật xác suất bằng bằng chứng [K1].",
        citations,
        evidence=evidence,
    )

    assert decision.status == "valid"
    assert decision.groundedness == "supported"
    assert decision.groundedness_score is not None


def test_shadow_groundedness_does_not_change_provenance_status() -> None:
    citation = KnowledgeCitation("wiki/bayes.md", "Bayes")
    evidence = (
        KnowledgeHit(
            KnowledgeDocument("wiki/bayes.md", "wiki/bayes.md", "Bayes", "Bayes."),
            score=0.9,
            snippet="Bayes.",
            retrieval_backend="dense",
            dense_score=0.9,
        ),
    )

    decision = validate_grounded_answer(
        "Thời tiết ngày mai chắc chắn tốt [K1].",
        (citation,),
        evidence=evidence,
    )

    assert decision.status == "valid"
    assert decision.groundedness == "unsupported"


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


def test_answer_validation_rejects_malformed_citation_mixed_with_valid_label() -> None:
    citations = (KnowledgeCitation("wiki/a.md", "A"),)

    decision = validate_grounded_answer("Theo [K1] nhưng cũng có [K0].", citations)

    assert decision.status == "invalid"
    assert decision.unknown_labels == ("[K0]",)


def test_answer_validation_rejects_zero_padded_citation_label() -> None:
    citations = (KnowledgeCitation("wiki/a.md", "A"),)

    decision = validate_grounded_answer("Theo [K01] thì đúng.", citations)

    assert decision.status == "invalid"
    assert decision.unknown_labels == ("[K01]",)
