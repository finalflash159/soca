"""Guards against the exact failure mode this module exists to prevent:
english_indices (recording time) and scoring alignment must tokenize
identically, or English-word indices point at the wrong token."""

from __future__ import annotations

from local.codeswitch_text import english_indices, normalize, tokens


def test_normalize_strips_punctuation_and_collapses_whitespace():
    assert normalize("Cái model này,   train bằng PyTorch!") == (
        "cái model này train bằng pytorch"
    )


def test_normalize_is_nfc():
    # NFD-composed "à" (a + combining grave) must normalize the same as NFC "à".
    nfd = "chào"
    nfc = "chào"
    assert normalize(nfd) == normalize(nfc)


def test_english_indices_match_token_positions():
    sentence = "mở cái repo trên github ra xem giúp tôi"
    idx = english_indices(sentence)
    words = tokens(sentence)
    assert [words[i] for i in idx] == ["repo", "github"]


def test_english_indices_ignores_vietnamese_words_that_look_like_terms():
    # "cache" and "log" are EN_TERMS; make sure a plain Vietnamese sentence
    # with no English words returns an empty list, not a false positive.
    assert english_indices("xin chào bạn khoẻ không") == []
