from __future__ import annotations

from soca.core.text_chunking import chunk_text_for_tts, normalize_text_for_tts, split_sentences


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
