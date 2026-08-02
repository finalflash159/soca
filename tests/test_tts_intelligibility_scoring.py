from __future__ import annotations

import pytest

from eval.tts_intelligibility.scoring import (
    ItemVerdict,
    aggregate,
    normalize_for_match,
    score_item,
    word_error_rate,
)


class TestNormalizeForMatch:
    def test_lowercases_and_collapses_whitespace(self) -> None:
        assert normalize_for_match("  Một   Trăm\tNghìn ") == "một trăm nghìn"

    def test_strips_punctuation_that_asr_does_not_emit(self) -> None:
        assert normalize_for_match("Giá là: một trăm nghìn đồng.") == "giá là một trăm nghìn đồng"

    def test_keeps_vietnamese_diacritics(self) -> None:
        assert normalize_for_match("Đường") == "đường"

    def test_is_nfc_normalized(self) -> None:
        decomposed = "é"  # e + combining acute
        composed = "é"
        assert normalize_for_match(decomposed) == normalize_for_match(composed)

    def test_empty_input_returns_empty(self) -> None:
        assert normalize_for_match("   ") == ""


class TestWordErrorRate:
    def test_identical_strings_score_zero(self) -> None:
        assert word_error_rate("một hai ba", "một hai ba") == 0.0

    def test_single_substitution(self) -> None:
        assert word_error_rate("một hai ba", "một bốn ba") == pytest.approx(1 / 3)

    def test_empty_reference_with_output_is_one(self) -> None:
        # No reference words: any output is fully wrong, but the rate must stay
        # finite so aggregation does not blow up on a degenerate corpus row.
        assert word_error_rate("", "gì đó") == 1.0

    def test_empty_reference_and_empty_output_is_zero(self) -> None:
        assert word_error_rate("", "") == 0.0

    def test_normalizes_before_comparing(self) -> None:
        assert word_error_rate("Một Hai Ba.", "một   hai ba") == 0.0


class TestScoreItem:
    def test_contains_mode_passes_when_expected_phrase_is_present(self) -> None:
        verdict = score_item(
            item_id="a1",
            corpus="normalizer",
            text_in="Giá là 100.000đ",
            expected="một trăm nghìn đồng",
            heard="giá là một trăm nghìn đồng",
            mode="contains",
        )
        assert verdict.passed is True
        assert verdict.wer == pytest.approx(0.0)

    def test_contains_mode_fails_when_phrase_is_absent(self) -> None:
        verdict = score_item(
            item_id="a2",
            corpus="normalizer",
            text_in="Gọi +84 912",
            expected="không chín một hai",
            heard="gọi cộng tám mươi tư chín trăm mười hai",
            mode="contains",
        )
        assert verdict.passed is False

    def test_exact_mode_compares_whole_utterance(self) -> None:
        verdict = score_item(
            item_id="c1",
            corpus="control",
            text_in="Hôm nay trời đẹp",
            expected="hôm nay trời đẹp",
            heard="hôm nay trời đẹp",
            mode="exact",
        )
        assert verdict.passed is True

    def test_exact_mode_fails_on_any_difference(self) -> None:
        verdict = score_item(
            item_id="c2",
            corpus="control",
            text_in="Hôm nay trời đẹp",
            expected="hôm nay trời đẹp",
            heard="hôm nay trời xấu",
            mode="exact",
        )
        assert verdict.passed is False
        assert verdict.wer > 0.0

    def test_term_mode_detects_letter_spelling_of_a_technical_word(self) -> None:
        # The Valtec failure this corpus exists to catch: an English term read
        # out letter by letter instead of as a word.
        verdict = score_item(
            item_id="b1",
            corpus="lexicon",
            text_in="Mô hình dùng embedding để so sánh",
            expected="embedding",
            heard="mô hình dùng i em bi i đi ai en gi để so sánh",
            mode="term",
        )
        assert verdict.passed is False

    def test_term_mode_passes_when_the_term_survives_the_round_trip(self) -> None:
        verdict = score_item(
            item_id="b2",
            corpus="lexicon",
            text_in="Mô hình dùng embedding để so sánh",
            expected="embedding",
            heard="mô hình dùng embedding để so sánh",
            mode="term",
        )
        assert verdict.passed is True

    def test_term_mode_rejects_a_longer_word_containing_the_term(self) -> None:
        # "logits" is a different rendering than "logit". A substring check
        # would pass it and hide exactly the defect this corpus hunts, so the
        # match has to respect word boundaries.
        verdict = score_item(
            item_id="b4",
            corpus="lexicon",
            text_in="Mô hình dùng logit để so sánh",
            expected="logit",
            heard="mô hình dùng logits để so sánh",
            mode="term",
        )
        assert verdict.passed is False

    def test_term_mode_rejects_a_term_glued_to_its_neighbour(self) -> None:
        verdict = score_item(
            item_id="b5",
            corpus="lexicon",
            text_in="Mô hình dùng cosine để so sánh",
            expected="cosine",
            heard="mô hình dùng cosinesim để so sánh",
            mode="term",
        )
        assert verdict.passed is False

    def test_contains_mode_requires_the_phrase_words_to_be_consecutive(self) -> None:
        verdict = score_item(
            item_id="a3",
            corpus="normalizer",
            text_in="Khoảng 2 giờ 20 phút",
            expected="hai giờ hai mươi phút",
            heard="khoảng hai giờ rồi hai mươi phút",
            mode="contains",
        )
        assert verdict.passed is False

    def test_an_empty_transcript_never_passes(self) -> None:
        verdict = score_item(
            item_id="b6",
            corpus="lexicon",
            text_in="Mô hình dùng softmax để so sánh",
            expected="softmax",
            heard="",
            mode="term",
        )
        assert verdict.passed is False

    def test_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            score_item(
                item_id="x",
                corpus="normalizer",
                text_in="a",
                expected="a",
                heard="a",
                mode="nonsense",  # type: ignore[arg-type]
            )

    def test_verdict_carries_confidence_when_provided(self) -> None:
        verdict = score_item(
            item_id="b3",
            corpus="lexicon",
            text_in="logits",
            expected="logits",
            heard="logits",
            mode="term",
            avg_logprob=-0.42,
        )
        assert verdict.avg_logprob == pytest.approx(-0.42)


class TestAggregate:
    @staticmethod
    def _verdict(corpus: str, passed: bool, wer: float) -> ItemVerdict:
        return ItemVerdict(
            item_id=f"{corpus}-{passed}-{wer}",
            corpus=corpus,
            text_in="x",
            expected="x",
            heard="x",
            passed=passed,
            wer=wer,
        )

    def test_groups_by_corpus_and_counts_passes(self) -> None:
        summary = aggregate(
            [
                self._verdict("normalizer", True, 0.0),
                self._verdict("normalizer", False, 0.5),
                self._verdict("control", True, 0.0),
            ]
        )
        assert summary["normalizer"].total == 2
        assert summary["normalizer"].passed == 1
        assert summary["control"].total == 1

    def test_reports_mean_wer_per_corpus(self) -> None:
        summary = aggregate(
            [
                self._verdict("lexicon", True, 0.0),
                self._verdict("lexicon", False, 0.4),
            ]
        )
        assert summary["lexicon"].mean_wer == pytest.approx(0.2)

    def test_pass_rate_is_none_for_an_empty_corpus(self) -> None:
        summary = aggregate([])
        assert summary == {}

    def test_failures_are_listed_for_inspection(self) -> None:
        bad = self._verdict("lexicon", False, 1.0)
        summary = aggregate([self._verdict("lexicon", True, 0.0), bad])
        assert [item.item_id for item in summary["lexicon"].failures] == [bad.item_id]
