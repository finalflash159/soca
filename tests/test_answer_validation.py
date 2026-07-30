from __future__ import annotations

from soca.core.answer_validation import (
    answer_text_without_citation_labels,
    expected_citation_labels,
    validate_grounded_answer,
)
from soca.knowledge import KnowledgeCitation, KnowledgeDocument, KnowledgeHit


def test_answer_validation_accepts_knowledge_and_memory_provenance() -> None:
    citations = (
        KnowledgeCitation("wiki/a.md", "A"),
        KnowledgeCitation("memory/b.md", "B", source="memory"),
    )
    decision = validate_grounded_answer("Theo [K1] và [M1] thì đúng.", citations)
    assert decision.status == "valid"


def test_presentation_removes_only_validated_labels_from_answer_text() -> None:
    citations = (
        KnowledgeCitation("wiki/a.md", "A"),
        KnowledgeCitation("memory/b.md", "B", source="memory"),
    )

    text = answer_text_without_citation_labels(
        "Bayes cập nhật xác suất [K1]. Sở thích đã lưu [M1]. Nhãn lạ [K9].",
        citations,
    )

    assert text == "Bayes cập nhật xác suất. Sở thích đã lưu. Nhãn lạ [K9]."


def test_expected_citation_labels_follow_each_source_sequence() -> None:
    citations = (
        KnowledgeCitation("wiki/a.md", "A"),
        KnowledgeCitation("memory/a.md", "A", source="memory"),
        KnowledgeCitation("wiki/b.md", "B"),
        KnowledgeCitation("memory/b.md", "B", source="memory"),
    )

    assert expected_citation_labels(citations) == ("[K1]", "[M1]", "[K2]", "[M2]")


def test_display_text_removes_structured_source_footer() -> None:
    citations = (KnowledgeCitation("wiki/attention.md", "Attention"),)
    text = (
        "Attention tập trung vào phần liên quan [K1].\n\n"
        "Nguồn:\n"
        "[K1] Attention · wiki/attention.md"
    )

    assert (
        answer_text_without_citation_labels(text, citations)
        == "Attention tập trung vào phần liên quan."
    )


def test_display_text_preserves_uncited_source_heading() -> None:
    citations = (KnowledgeCitation("wiki/attention.md", "Attention"),)
    text = "Nguồn:\nĐây là phần nội dung người dùng yêu cầu, không phải citation footer."

    assert answer_text_without_citation_labels(text, citations) == text


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
    assert decision.grounded_claim_count == 1
    assert decision.ungrounded_claim_count == 0


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
    assert decision.grounded_claim_count == 0
    assert decision.ungrounded_claim_count == 1


def test_answer_validation_reports_missing_provenance_without_blocking() -> None:
    decision = validate_grounded_answer(
        "Một câu trả lời không nguồn.", (KnowledgeCitation("wiki/a.md", "A"),)
    )
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
