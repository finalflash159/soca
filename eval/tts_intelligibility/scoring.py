from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

MatchMode = Literal["contains", "exact", "term"]

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize_for_match(text: str) -> str:
    """Fold text to the form both sides can be compared in.

    ASR output carries no punctuation and inconsistent casing, so a raw
    comparison against a written expectation fails for reasons that have
    nothing to do with pronunciation. NFC is applied because Vietnamese
    diacritics have two Unicode spellings and the two must compare equal.
    """
    folded = unicodedata.normalize("NFC", text).casefold()
    folded = _PUNCT_RE.sub(" ", folded)
    return _SPACE_RE.sub(" ", folded).strip()


def _contains_token_sequence(haystack: str, needle: str) -> bool:
    """Whether ``needle`` appears in ``haystack`` as consecutive whole words.

    Substring matching would accept a term the engine never said: "logit" is
    inside "logits" and "cosine" is inside "cosinesim", yet both are different
    renderings and one of them is exactly the defect being hunted. Comparing
    token sequences makes a word boundary mandatory without needing a regex
    that behaves differently for Vietnamese and ASCII terms.
    """
    needle_tokens = needle.split()
    if not needle_tokens:
        return False
    haystack_tokens = haystack.split()
    span = len(needle_tokens)
    return any(
        haystack_tokens[start : start + span] == needle_tokens
        for start in range(len(haystack_tokens) - span + 1)
    )


def _levenshtein_words(reference: list[str], hypothesis: list[str]) -> int:
    if not reference:
        return len(hypothesis)
    previous = list(range(len(reference) + 1))
    for j, hyp_word in enumerate(hypothesis, start=1):
        current = [j]
        for i, ref_word in enumerate(reference, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(
                min(
                    previous[i] + 1,  # deletion from hypothesis
                    current[i - 1] + 1,  # insertion
                    previous[i - 1] + cost,  # substitution
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word error rate after match-normalization, clamped to a finite value.

    An empty reference would divide by zero. It is reported as a full error
    when anything was heard and as zero when nothing was, so a degenerate
    corpus row cannot poison the corpus mean.
    """
    ref_words = normalize_for_match(reference).split()
    hyp_words = normalize_for_match(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _levenshtein_words(ref_words, hyp_words) / len(ref_words)


@dataclass(frozen=True)
class ItemVerdict:
    item_id: str
    corpus: str
    text_in: str
    expected: str
    heard: str
    passed: bool
    wer: float
    avg_logprob: float | None = None


def score_item(
    *,
    item_id: str,
    corpus: str,
    text_in: str,
    expected: str,
    heard: str,
    mode: MatchMode,
    avg_logprob: float | None = None,
) -> ItemVerdict:
    """Decide whether one synthesized utterance survived the round trip.

    ``contains``/``term`` ask whether a specific phrase reached the listener,
    which is what the normalizer and lexicon corpora care about. ``exact``
    compares the whole utterance and is reserved for the control corpus, where
    any drift is evidence about the measuring chain rather than the term.
    """
    if mode not in ("contains", "exact", "term"):
        raise ValueError(f"Unknown match mode: {mode!r}. Valid modes: contains, exact, term")

    normalized_heard = normalize_for_match(heard)
    normalized_expected = normalize_for_match(expected)

    if mode == "exact":
        passed = normalized_heard == normalized_expected
        wer = word_error_rate(expected, heard)
    else:
        passed = _contains_token_sequence(normalized_heard, normalized_expected)
        # For a phrase-level expectation the whole-utterance WER would be
        # dominated by the carrier sentence, so the rate is measured on the
        # expectation alone: 0.0 when it survived, 1.0 when it did not.
        wer = 0.0 if passed else 1.0

    return ItemVerdict(
        item_id=item_id,
        corpus=corpus,
        text_in=text_in,
        expected=expected,
        heard=heard,
        passed=passed,
        wer=wer,
        avg_logprob=avg_logprob,
    )


@dataclass(frozen=True)
class CorpusSummary:
    corpus: str
    total: int
    passed: int
    mean_wer: float
    failures: tuple[ItemVerdict, ...] = field(default=())

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


def aggregate(verdicts: list[ItemVerdict]) -> dict[str, CorpusSummary]:
    grouped: dict[str, list[ItemVerdict]] = {}
    for verdict in verdicts:
        grouped.setdefault(verdict.corpus, []).append(verdict)

    summaries: dict[str, CorpusSummary] = {}
    for corpus, items in grouped.items():
        summaries[corpus] = CorpusSummary(
            corpus=corpus,
            total=len(items),
            passed=sum(1 for item in items if item.passed),
            mean_wer=sum(item.wer for item in items) / len(items),
            failures=tuple(item for item in items if not item.passed),
        )
    return summaries


__all__ = [
    "CorpusSummary",
    "ItemVerdict",
    "MatchMode",
    "aggregate",
    "normalize_for_match",
    "score_item",
    "word_error_rate",
]
