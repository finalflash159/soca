from __future__ import annotations

from local.score_codeswitch import score_system


def _row(sentence_id: str, reference: str, english_words: list[str]) -> dict:
    from local.codeswitch_text import tokens

    words = tokens(reference)
    return {
        "id": sentence_id,
        "reference": reference,
        "english_indices": [words.index(w) for w in english_words],
    }


def test_perfect_hypothesis_scores_zero_wer_and_full_en_recall():
    rows = [_row("cs_000", "mở cái repo trên github ra xem", ["repo", "github"])]
    predictions = {"cs_000": "mở cái repo trên github ra xem"}

    stats = score_system(rows, predictions)

    assert stats["wer"] == 0.0
    assert stats["en_recall"] == 1.0
    assert stats["cs_wer"] == 0.0
    assert stats["en_total"] == 2
    assert stats["en_correct"] == 2


def test_mangled_english_word_is_recorded_as_a_miss():
    rows = [_row("cs_000", "cái model này train bằng pytorch", ["pytorch"])]
    predictions = {"cs_000": "cái ba đồ này chai bằng thai touch"}

    stats = score_system(rows, predictions)

    assert stats["en_total"] == 1
    assert stats["en_correct"] == 0
    assert stats["cs_wer"] == 1.0
    assert stats["top_misses"] == [("pytorch", 1)]


def test_empty_hypothesis_counts_as_full_miss_not_a_crash():
    rows = [_row("cs_000", "mở cái repo trên github", ["repo", "github"])]
    predictions = {"cs_000": ""}

    stats = score_system(rows, predictions)

    assert stats["en_recall"] == 0.0
    assert stats["wer"] == 1.0


def test_missing_prediction_id_falls_back_to_empty_hypothesis():
    rows = [_row("cs_000", "mở cái repo trên github", ["repo", "github"])]

    stats = score_system(rows, predictions={})

    assert stats["en_recall"] == 0.0
    assert stats["num_utterances"] == 1
