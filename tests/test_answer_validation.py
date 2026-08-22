from __future__ import annotations

import unicodedata

from soca.core.answer_validation import (
    answer_chunk_without_citation_labels,
    answer_text_without_citation_labels,
    citation_records,
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


def test_citation_records_preserve_new_fingerprints_without_changing_legacy_shape() -> None:
    records = citation_records(
        (
            KnowledgeCitation("wiki/new.md", "New", fingerprint="sha256-new"),
            KnowledgeCitation("wiki/old.md", "Old"),
        )
    )

    assert records[0]["fingerprint"] == "sha256-new"
    assert "fingerprint" not in records[1]


def test_presentation_removes_citation_tags_from_answer_text() -> None:
    citations = (
        KnowledgeCitation("wiki/a.md", "A"),
        KnowledgeCitation("memory/b.md", "B", source="memory"),
    )

    text = answer_text_without_citation_labels(
        (
            "Bayes cập nhật xác suất [K1]. Sở thích đã lưu [M1]. "
            "Nhãn lạ [K9], nguồn số [1], nhưng giữ [TODO] và array[0]."
        ),
        citations,
    )

    assert text == (
        "Bayes cập nhật xác suất. Sở thích đã lưu. "
        "Nhãn lạ, nguồn số, nhưng giữ [TODO] và array[0]."
    )


def test_presentation_keeps_bracketed_numbers_when_turn_has_no_citations() -> None:
    text = answer_text_without_citation_labels(
        "Giá là [100] nghìn đồng, vụ án số [2119], nhãn lạ [K9] thì bỏ.",
        (),
    )

    assert text == "Giá là [100] nghìn đồng, vụ án số [2119], nhãn lạ thì bỏ."


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


def test_display_text_removes_empty_source_footer_when_citations_are_structured() -> None:
    citations = (KnowledgeCitation("wiki/review.md", "Review tuần 30/2026"),)
    text = "Các việc chưa hoàn thành được liệt kê ở trên.\n\nNguồn:\n\n"

    assert (
        answer_text_without_citation_labels(text, citations)
        == "Các việc chưa hoàn thành được liệt kê ở trên."
    )


def test_display_text_removes_markdown_source_footer_when_citations_are_structured() -> None:
    citations = (KnowledgeCitation("wiki/review.md", "Review tuần 30/2026"),)
    text = "Đã đối chiếu xong [K1].\n\n**Nguồn:**\n- [K1] Review tuần 30/2026"

    assert answer_text_without_citation_labels(text, citations) == "Đã đối chiếu xong."


def test_display_text_removes_source_footer_when_provider_sends_nfd_text() -> None:
    # Some providers stream Vietnamese diacritics NFD-decomposed. The footer
    # regex's "nguồn" literal is NFC, so an unnormalized response used to
    # leak the raw "Nguồn:" footer into both chat display and speech.
    citations = (KnowledgeCitation("wiki/review.md", "Review tuần 30/2026"),)
    text = unicodedata.normalize(
        "NFD", "Đã ổn [K1].\n\nNguồn:\n[K1] Review tuần 30/2026 · wiki/review.md"
    )

    result = answer_text_without_citation_labels(text, citations)

    assert result == "Đã ổn."
    assert "guồn" not in result.lower()


def test_display_text_preserves_source_heading_without_structured_citations() -> None:
    text = "Nguồn:\nĐây là phần nội dung người dùng yêu cầu, không phải citation footer."

    assert answer_text_without_citation_labels(text, ()) == text


def test_display_text_removes_source_footer_with_non_label_content() -> None:
    citations = (KnowledgeCitation("wiki/attention.md", "Attention"),)
    text = (
        "Attention tập trung vào phần liên quan.\n\n"
        "Nguồn:\n"
        "\u200bK1 — Attention"
    )

    assert (
        answer_text_without_citation_labels(text, citations)
        == "Attention tập trung vào phần liên quan."
    )


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


def test_chunk_cleaner_preserves_the_whitespace_that_joins_chunks() -> None:
    """Clients concatenate chunks, so the edges carry the sentence spacing."""
    chunks = ("Protein giữ cơ bắp [K1]. ", "Nó tạo cảm giác no [K2].")

    cleaned = [answer_chunk_without_citation_labels(chunk) for chunk in chunks]

    assert cleaned == ["Protein giữ cơ bắp. ", "Nó tạo cảm giác no."]
    assert "".join(cleaned) == "Protein giữ cơ bắp. Nó tạo cảm giác no."


def test_chunk_cleaner_matches_the_whole_answer_cleaner_when_no_footer_exists() -> None:
    """The prompt forbids a source footer, so both cleaners must agree."""
    answer = "Protein giữ cơ bắp [K1]. Nó tạo cảm giác no [M1]."

    assert answer_chunk_without_citation_labels(answer) == (
        answer_text_without_citation_labels(answer, ())
    )


def test_chunk_cleaner_leaves_a_source_footer_for_the_whole_answer_cleaner() -> None:
    """Whether a line is the *last* footer cannot be decided from one chunk."""
    chunk = "Nguồn:\n"

    assert answer_chunk_without_citation_labels(chunk) == chunk


def test_chunk_cleaner_normalizes_decomposed_vietnamese_before_stripping() -> None:
    decomposed = unicodedata.normalize("NFD", "Nó tạo cảm giác no [K1].")

    cleaned = answer_chunk_without_citation_labels(decomposed)

    assert cleaned == "Nó tạo cảm giác no."
