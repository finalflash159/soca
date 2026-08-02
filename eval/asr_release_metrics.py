from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from jiwer import process_characters, process_words


def normalize_transcript(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).casefold()
    return " ".join(
        "".join(char if char.isalnum() or char.isspace() else " " for char in normalized).split()
    )


@dataclass(frozen=True, slots=True)
class ErrorRates:
    wer: float
    cer: float
    word_errors: int
    reference_words: int
    character_errors: int
    reference_characters: int


def error_rates(references: Sequence[str], hypotheses: Sequence[str]) -> ErrorRates:
    if len(references) != len(hypotheses) or not references:
        raise ValueError("references and hypotheses must have equal non-zero length")
    refs = [normalize_transcript(value) for value in references]
    hyps = [normalize_transcript(value) for value in hypotheses]
    words = process_words(refs, hyps)
    characters = process_characters(refs, hyps)
    return ErrorRates(
        wer=float(words.wer),
        cer=float(characters.cer),
        word_errors=words.substitutions + words.deletions + words.insertions,
        reference_words=words.hits + words.substitutions + words.deletions,
        character_errors=(
            characters.substitutions + characters.deletions + characters.insertions
        ),
        reference_characters=(
            characters.hits + characters.substitutions + characters.deletions
        ),
    )


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    auroc: float
    average_precision: float
    positives: int
    negatives: int


@dataclass(frozen=True, slots=True)
class CodeSwitchMetrics:
    term_recall: float
    cs_wer: float
    reference_terms: int
    correct_terms: int


def code_switch_metrics(
    references: Sequence[str],
    hypotheses: Sequence[str],
    english_indices_by_item: Sequence[Sequence[int]],
) -> CodeSwitchMetrics:
    if not references or len(references) != len(hypotheses) or len(references) != len(
        english_indices_by_item
    ):
        raise ValueError("code-switch inputs must have equal non-zero length")
    reference_terms = 0
    correct_terms = 0
    for reference, hypothesis, indices in zip(
        references, hypotheses, english_indices_by_item, strict=True
    ):
        normalized_reference = normalize_transcript(reference)
        normalized_hypothesis = normalize_transcript(hypothesis)
        reference_tokens = normalized_reference.split()
        if any(index < 0 or index >= len(reference_tokens) for index in indices):
            raise ValueError("code-switch reference index is out of range")
        output = process_words(normalized_reference, normalized_hypothesis)
        correct: set[int] = set()
        for alignment in output.alignments:
            for chunk in alignment:
                if chunk.type == "equal":
                    correct.update(range(chunk.ref_start_idx, chunk.ref_end_idx))
        reference_terms += len(indices)
        correct_terms += sum(index in correct for index in indices)
    if reference_terms == 0:
        raise ValueError("code-switch metrics require at least one labelled term")
    recall = correct_terms / reference_terms
    return CodeSwitchMetrics(
        term_recall=recall,
        cs_wer=1 - recall,
        reference_terms=reference_terms,
        correct_terms=correct_terms,
    )


def ranking_metrics(labels: Sequence[bool], scores: Sequence[float]) -> RankingMetrics:
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must have equal non-zero length")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("ranking scores must be finite")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ranking metrics require both classes")

    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    true_positive = 0
    false_positive = 0
    previous_false_positive_rate = 0.0
    previous_true_positive_rate = 0.0
    auroc = 0.0
    precision_sum = 0.0
    cursor = 0
    while cursor < len(order):
        score = scores[order[cursor]]
        group: list[int] = []
        while cursor < len(order) and scores[order[cursor]] == score:
            group.append(order[cursor])
            cursor += 1
        group_positives = sum(labels[index] for index in group)
        true_positive += group_positives
        false_positive += len(group) - group_positives
        true_positive_rate = true_positive / positives
        false_positive_rate = false_positive / negatives
        auroc += (false_positive_rate - previous_false_positive_rate) * (
            true_positive_rate + previous_true_positive_rate
        ) / 2
        precision = true_positive / (true_positive + false_positive)
        precision_sum += precision * group_positives
        previous_false_positive_rate = false_positive_rate
        previous_true_positive_rate = true_positive_rate
    return RankingMetrics(
        auroc=auroc,
        average_precision=precision_sum / positives,
        positives=positives,
        negatives=negatives,
    )


def select_lower_bound_threshold(
    speech_scores: Sequence[float],
    *,
    max_false_reject_rate: float,
) -> float:
    if not speech_scores:
        raise ValueError("speech scores must not be empty")
    if not 0 <= max_false_reject_rate < 1:
        raise ValueError("max_false_reject_rate must be in [0, 1)")
    if any(not math.isfinite(score) for score in speech_scores):
        raise ValueError("speech scores must be finite")
    ordered = sorted(speech_scores)
    allowed_rejections = math.floor(len(ordered) * max_false_reject_rate)
    return float(ordered[min(allowed_rejections, len(ordered) - 1)])


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile values must not be empty")
    if not 0 <= fraction <= 1:
        raise ValueError("percentile fraction must be in [0, 1]")
    return float(np.percentile(np.asarray(values, dtype=np.float64), fraction * 100))


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if total < 1 or successes < 0 or successes > total:
        raise ValueError("Wilson interval counts are invalid")
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        probability * (1 - probability) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


__all__ = [
    "CodeSwitchMetrics",
    "ErrorRates",
    "RankingMetrics",
    "code_switch_metrics",
    "error_rates",
    "normalize_transcript",
    "percentile",
    "ranking_metrics",
    "select_lower_bound_threshold",
    "wilson_interval",
]
