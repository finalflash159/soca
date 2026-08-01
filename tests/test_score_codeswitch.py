from __future__ import annotations

import pytest

from local.codeswitch_text import english_indices
from local.score_codeswitch import score_system


def _row(sentence_id: str, reference: str) -> dict:
    return {
        "id": sentence_id,
        "reference": reference,
        "english_indices": english_indices(reference),
    }


def test_perfect_hypothesis_scores_zero_wer_and_full_en_recall():
    rows = [_row("cs_000", "mở cái repo trên github ra xem")]
    predictions = {"cs_000": "mở cái repo trên github ra xem"}

    stats = score_system(rows, predictions)

    assert stats["wer"] == 0.0
    assert stats["en_recall"] == 1.0
    assert stats["cs_wer"] == 0.0
    assert stats["en_total"] == 2
    assert stats["en_correct"] == 2


def test_mangled_english_word_is_recorded_as_a_miss():
    # "postgresql" is the only EN_TERMS word here; the rest of the sentence
    # keeps the test isolated to a single term.
    rows = [_row("cs_000", "cái này chạy trên postgresql")]
    predictions = {"cs_000": "cái này chạy trên po tơ grét sồ"}

    stats = score_system(rows, predictions)

    assert stats["en_total"] == 1
    assert stats["en_correct"] == 0
    assert stats["cs_wer"] == 1.0
    assert stats["top_misses"] == [("postgresql", 1)]


def test_empty_hypothesis_counts_as_full_miss_not_a_crash():
    rows = [_row("cs_000", "mở cái repo trên github")]
    predictions = {"cs_000": ""}

    stats = score_system(rows, predictions)

    assert stats["en_recall"] == 0.0
    assert stats["wer"] == 1.0


def test_missing_prediction_id_falls_back_to_empty_hypothesis():
    rows = [_row("cs_000", "mở cái repo trên github")]

    stats = score_system(rows, predictions={})

    assert stats["en_recall"] == 0.0
    assert stats["num_utterances"] == 1


def test_stale_manifest_english_indices_raise_instead_of_silently_shrinking_denominator():
    """EN_TERMS can change after a manifest was recorded (it has before —
    'repo' was added later). A manifest whose stored english_indices no
    longer matches a fresh recompute must fail loudly, not silently drop
    the mismatched index and inflate en_recall.
    """
    row = _row("cs_000", "mở cái repo trên github")
    row["english_indices"] = [row["english_indices"][0]]  # drop "github" -> stale

    with pytest.raises(ValueError, match="english_indices mismatch"):
        score_system([row], {"cs_000": "mở cái repo trên github"})
