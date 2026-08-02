"""Unit tests for soca.asr.hallucination_heuristics."""

from __future__ import annotations

from soca.asr.hallucination_heuristics import (
    check_heuristics,
    compression_ratio,
    is_filler_only,
    looks_like_context_echo,
    max_contiguous_context_tokens,
    n_gram_repetition,
    repetition_ratio,
)


class TestIsFillerOnly:
    def test_single_filler(self):
        assert is_filler_only("ờ") is True

    def test_multiple_fillers(self):
        assert is_filler_only("ờ ạ ừm") is True

    def test_filler_plus_content(self):
        assert is_filler_only("ờ mấy giờ rồi") is False

    def test_normal_text(self):
        assert is_filler_only("xin chào") is False

    def test_empty(self):
        assert is_filler_only("") is False


class TestRepetitionRatio:
    def test_all_unique(self):
        assert repetition_ratio("hôm nay trời đẹp lắm") == 0.0

    def test_all_same(self):
        # "a a a a" -> 1 unique / 4 total = 0.25 unique → 0.75 repetition.
        assert repetition_ratio("a a a a") == 0.75

    def test_under_four_tokens_returns_zero(self):
        assert repetition_ratio("a a a") == 0.0


class TestNgramRepetition:
    def test_perfect_3gram_loop(self):
        # "thấy thế nào thấy thế nào" → 4 trigrams, 3 unique → score 0.25
        score = n_gram_repetition("thấy thế nào thấy thế nào", n=3)
        assert score > 0.2

    def test_no_3gram_repeat(self):
        score = n_gram_repetition("hôm nay trời đẹp lắm bạn nhé", n=3)
        assert score == 0.0

    def test_under_threshold_length(self):
        # Only 4 tokens, need 6 for 3-gram repetition check
        assert n_gram_repetition("a b c d", n=3) == 0.0


class TestCheckHeuristics:
    def test_clean_text(self):
        result = check_heuristics("Mấy giờ rồi", speech_duration_ms=1500)
        assert result.is_hallucination is False
        assert result.reason == "clean"

    def test_empty_text(self):
        result = check_heuristics("", speech_duration_ms=1500)
        assert result.is_hallucination is False
        assert result.reason == "empty"

    def test_filler_only_rejected(self):
        result = check_heuristics("ờ ạ", speech_duration_ms=1500)
        assert result.is_hallucination is True
        assert result.reason == "filler_only"

    def test_unigram_repetition_rejected(self):
        # 5 of 6 same word -> repetition ratio 0.833 > 0.50
        result = check_heuristics("a a a a a b", speech_duration_ms=1500)
        assert result.is_hallucination is True
        assert "unigram_rep" in result.reason

    def test_ngram_loop_rejected(self):
        # Loop is caught — could be by unigram or ngram check, both indicate hallucination.
        result = check_heuristics("thấy thế nào thấy thế nào thấy thế nào", speech_duration_ms=2000)
        assert result.is_hallucination is True

    def test_text_too_long_for_short_speech(self):
        # 200ms speech but 100 chars of text -> 50 chars/100ms, way over 8
        long_text = "a" * 100
        result = check_heuristics(long_text, speech_duration_ms=200)
        assert result.is_hallucination is True
        assert "text_too_long" in result.reason

    def test_density_check_skipped_for_long_speech(self):
        # Long speech, density check should not apply
        # Use unique words to avoid triggering repetition
        text = " ".join(f"từ{i}" for i in range(20))
        result = check_heuristics(text, speech_duration_ms=5000)
        assert result.is_hallucination is False

    def test_custom_thresholds(self):
        # Loose both thresholds -> not rejected
        result = check_heuristics(
            "a a a a a b",
            speech_duration_ms=1500,
            repetition_thresh=0.95,
            ngram_repetition_thresh=0.99,
        )
        assert result.is_hallucination is False

    def test_compression_ratio_empty_text(self):
        assert compression_ratio("") == 0.0

    def test_compression_ratio_repeated_text_is_higher(self):
        repeated = "xin chào " * 80
        normal = "bật đèn phòng khách rồi đặt hẹn giờ năm phút"
        assert compression_ratio(repeated) > compression_ratio(normal)


class TestLooksLikeContextEcho:
    CONTEXT = (
        "Cuộc hội thoại về lập trình. Giữ nguyên cách viết các thuật ngữ tiếng Anh: "
        "GitHub, PyTorch, TensorFlow, PostgreSQL, Docker, Kubernetes."
    )

    def test_verbatim_context_is_flagged(self):
        assert looks_like_context_echo(self.CONTEXT, self.CONTEXT) is True

    def test_case_changed_partial_echo_is_flagged(self):
        echoed = (
            "cuộc hội thoại về lập trình github pytorch tensorflow postgresql docker kubernetes"
        )
        assert looks_like_context_echo(echoed, self.CONTEXT) is True

    def test_non_contiguous_shared_vocabulary_is_not_flagged(self):
        text = "github mở giúp tôi rồi kiểm tra docker"
        assert looks_like_context_echo(text, self.CONTEXT) is False

    def test_reports_longest_contiguous_span(self):
        assert (
            max_contiguous_context_tokens(
                "giữ nguyên cách viết thuật ngữ",
                self.CONTEXT,
            )
            == 4
        )

    def test_normal_transcript_is_not_flagged(self):
        text = "mở cái repo trên github ra xem giúp tôi"
        assert looks_like_context_echo(text, self.CONTEXT) is False

    def test_short_sentence_sharing_a_couple_of_terms_is_not_flagged(self):
        # Must not punish legitimate short sentences just for using terms
        # that also appear in the context.
        assert looks_like_context_echo("mở github", self.CONTEXT) is False

    def test_empty_context_never_flags_anything(self):
        assert looks_like_context_echo(self.CONTEXT, "") is False
        assert looks_like_context_echo(self.CONTEXT, "   ") is False
