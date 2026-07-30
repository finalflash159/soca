from __future__ import annotations

import pytest

from soca.core.text_chunking import (
    chunk_text_for_tts,
    normalize_text_for_tts,
    split_first_clause,
    split_sentences,
)


def test_split_sentences_keeps_terminal_punctuation() -> None:
    text = "Xin chao. Ban khoe khong? Toi on!"

    assert split_sentences(text) == [
        "Xin chao.",
        "Ban khoe khong?",
        "Toi on!",
    ]


def test_split_sentences_handles_newlines() -> None:
    text = "Dong mot\nDong hai\n\nDong ba"

    assert split_sentences(text) == ["Dong mot", "Dong hai", "Dong ba"]


def test_split_sentences_handles_unicode_ellipsis() -> None:
    text = "Cho minh nghi mot chut\u2026 Duoc roi."

    assert split_sentences(text) == ["Cho minh nghi mot chut\u2026", "Duoc roi."]


def test_split_sentences_ignores_empty_text() -> None:
    assert split_sentences("") == []
    assert split_sentences("   \n\t ") == []


def test_split_sentences_does_not_split_without_boundary_space() -> None:
    text = "Phien ban 1.0 van chay tot."

    assert split_sentences(text) == ["Phien ban 1.0 van chay tot."]


def test_split_sentences_keeps_numbered_list_marker_with_item() -> None:
    text = "Tình hình:\n1. Ăn đủ đạm.\n2. Ngủ sớm."

    assert split_sentences(text) == [
        "Tình hình:",
        "1. Ăn đủ đạm.",
        "2. Ngủ sớm.",
    ]


def test_chunk_text_for_tts_merges_short_leading_sentence() -> None:
    text = "Xin chào! Tôi sẵn sàng giúp bạn. Bạn cần gì hôm nay?"

    assert chunk_text_for_tts(text, min_chars=24) == [
        "Xin chào! Tôi sẵn sàng giúp bạn. Bạn cần gì hôm nay?",
    ]


def test_chunk_text_for_tts_does_not_emit_bare_numbered_marker() -> None:
    text = "Tình hình:\n1. Ăn đủ đạm.\n2. Ngủ sớm."

    chunks = chunk_text_for_tts(text, min_chars=12)

    assert all(not chunk.endswith(" 1.") for chunk in chunks)
    assert "1. Ăn đủ đạm." in " ".join(chunks)


def test_normalize_text_for_tts_strips_markdown_emphasis() -> None:
    text = "**Tình hình:** cần ăn **đủ đạm** và _ngủ sớm_."

    assert normalize_text_for_tts(text) == "Tình hình: cần ăn đủ đạm và ngủ sớm."


def test_normalize_text_for_tts_keeps_link_label_and_code_text() -> None:
    text = "- Xem [ghi chú](wiki/dinh-duong/chat-dam.md) và `protein`."

    assert normalize_text_for_tts(text) == "Xem ghi chú và protein."


def test_normalize_text_for_tts_removes_citation_tags_only() -> None:
    text = "Theo [K1] và [2], attention quan trọng; giữ [TODO] và array[0]."

    assert normalize_text_for_tts(text) == (
        "Theo và, attention quan trọng; giữ [TODO] và array[0]."
    )


def test_first_clause_keeps_discourse_marker_whole() -> None:
    first, rest = split_first_clause(
        "Tuy nhiên, nếu bạn cần nhanh hơn thì dùng cấu hình gọn.",
        min_chars=8,
        min_words=2,
    )

    assert first == "Tuy nhiên,"
    assert rest == "nếu bạn cần nhanh hơn thì dùng cấu hình gọn."


def test_first_clause_rejects_tiny_fragment_without_forcing_space() -> None:
    text = "Ừ, mình sẽ giải thích kỹ hơn nhé"

    first, rest = split_first_clause(text, min_chars=12, min_words=2)

    assert first is None
    assert rest == text


def test_first_clause_waits_for_token_after_punctuation() -> None:
    text = "Mình sẽ kiểm tra, "

    assert split_first_clause(text) == (None, text)


def test_first_clause_skips_number_and_time_internal_punctuation() -> None:
    assert split_first_clause("Giá là 1,000 đồng, bạn có thể mua.")[0] == (
        "Giá là 1,000 đồng,"
    )
    assert split_first_clause("Hẹn lúc 12:30, mình sẽ nhắc bạn.")[0] == (
        "Hẹn lúc 12:30,"
    )


def test_first_clause_allows_boundary_after_url() -> None:
    first, rest = split_first_clause(
        "Xem https://soca.local/a, rồi báo mình.",
        min_chars=8,
    )

    assert first == "Xem https://soca.local/a,"
    assert rest == "rồi báo mình."


def test_first_clause_ignores_punctuation_inside_markdown() -> None:
    first, _ = split_first_clause(
        "Dùng `a, b` để thử, rồi báo kết quả.",
        min_chars=8,
    )

    assert first == "Dùng `a, b` để thử,"


def test_first_clause_validates_thresholds() -> None:
    with pytest.raises(ValueError, match="min_chars"):
        split_first_clause("Xin chào, bạn.", min_chars=0)
    with pytest.raises(ValueError, match="max_scan_chars"):
        split_first_clause("Xin chào, bạn.", min_chars=12, max_scan_chars=8)
