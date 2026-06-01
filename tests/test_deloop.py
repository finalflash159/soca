"""Unit tests for soca.asr.deloop."""

from __future__ import annotations

from soca.asr.deloop import has_excessive_repetition, remove_consecutive_repeats


class TestRemoveConsecutiveRepeats:
    def test_simple_pair_repeat(self):
        cleaned, looping = remove_consecutive_repeats("xin chào xin chào")
        assert cleaned == "xin chào"
        assert looping is True

    def test_triple_window_repeat(self):
        cleaned, looping = remove_consecutive_repeats("thấy thế nào thấy thế nào thấy thế nào")
        assert cleaned == "thấy thế nào"
        assert looping is True

    def test_no_repeat_short_text(self):
        cleaned, looping = remove_consecutive_repeats("đặt báo thức lúc bảy giờ")
        assert cleaned == "đặt báo thức lúc bảy giờ"
        assert looping is False

    def test_no_repeat_long_text(self):
        text = "hôm nay trời rất đẹp và tôi muốn đi dạo một vòng quanh hồ"
        cleaned, looping = remove_consecutive_repeats(text)
        assert cleaned == text
        assert looping is False

    def test_under_four_tokens_returns_as_is(self):
        # Too short for any window detection.
        cleaned, looping = remove_consecutive_repeats("xin chào")
        assert cleaned == "xin chào"
        assert looping is False

    def test_empty_string(self):
        cleaned, looping = remove_consecutive_repeats("")
        assert cleaned == ""
        assert looping is False

    def test_partial_repeat_then_continuation(self):
        # Loop ends, then new content. Cleanup keeps the new content.
        cleaned, looping = remove_consecutive_repeats("ờ ờ ờ ờ mấy giờ rồi")
        assert "mấy giờ rồi" in cleaned
        assert looping is True

    def test_legitimate_short_word_pair(self):
        # "rất rất" is legitimate emphasis, but it's caught as a 1-token repeat.
        # However we only check windows >= 2, so single-word doubles pass through.
        cleaned, looping = remove_consecutive_repeats("tôi rất rất thích nó")
        assert cleaned == "tôi rất rất thích nó"
        assert looping is False


class TestHasExcessiveRepetition:
    def test_high_repetition(self):
        assert has_excessive_repetition("a a a a b") is True

    def test_low_repetition(self):
        assert has_excessive_repetition("hôm nay trời đẹp lắm") is False

    def test_under_four_tokens(self):
        assert has_excessive_repetition("a a a") is False
