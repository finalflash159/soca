from __future__ import annotations

from soca.core.text_chunking import chunk_text_for_tts, split_sentences


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


def test_chunk_text_for_tts_merges_short_leading_sentence() -> None:
    text = "Xin chào! Tôi sẵn sàng giúp bạn. Bạn cần gì hôm nay?"

    assert chunk_text_for_tts(text, min_chars=24) == [
        "Xin chào! Tôi sẵn sàng giúp bạn. Bạn cần gì hôm nay?",
    ]
