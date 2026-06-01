"""Lightweight post-ASR heuristic checks for Whisper hallucinations.

Complementary to BoH matching — these catch hallucinations NOT in the
empirical phrase list (novel loops, filler-only outputs, output too long
for input duration).

Reference: Barański et al. (ICASSP 2025); thresholds calibrated from p99
on real Vietnamese speech (see local/calibrate_thresholds.py).
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass

# Vietnamese conversational fillers. A real command can START with one
# ("ờ mấy giờ rồi?"), but if the ENTIRE output is filler tokens we treat
# it as hallucination of near-silence.
COMMON_FILLERS = frozenset({"ờ", "ừm", "à", "ạ", "uhm", "hmm", "ừ", "dạ"})


@dataclass(frozen=True)
class HeuristicCheck:
    """Result of a multi-heuristic hallucination check.

    `reason` is a short machine-readable tag (e.g. "ngram_rep:0.62") so
    upstream code can route or log per-rule rejections.
    """

    is_hallucination: bool
    reason: str = ""
    confidence: float = 0.0


def is_filler_only(text: str) -> bool:
    """True iff text consists exclusively of filler tokens.

    "ờ" → True. "ờ mấy giờ" → False.
    """
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return False
    return all(token in COMMON_FILLERS for token in tokens)


def repetition_ratio(text: str) -> float:
    """Unigram repetition ratio: 1 - unique_tokens/total_tokens.

    0.0 = all tokens unique. 1.0 = all tokens identical.
    Returns 0.0 for inputs with <4 tokens (too short to be meaningful).
    """
    tokens = text.lower().split()
    if len(tokens) < 4:
        return 0.0
    return 1.0 - len(set(tokens)) / len(tokens)


def n_gram_repetition(text: str, n: int = 3) -> float:
    """N-gram repetition ratio. Catches "thấy thế nào thấy thế nào" cases.

    More sensitive than unigram repetition for structural loops because
    individual tokens may all be unique within a window but the window
    itself repeats.
    """
    tokens = text.lower().split()
    if len(tokens) < n * 2:
        return 0.0
    ngrams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - len(set(ngrams)) / max(len(ngrams), 1)


def compression_ratio(text: str) -> float:
    """OpenAI Whisper-style UTF-8 compression ratio.

    Highly repetitive hallucinations compress well, so their raw/compressed ratio
    is higher. Empty text has no useful signal and returns 0.0.
    """
    raw = text.encode("utf-8")
    if not raw:
        return 0.0
    return len(raw) / len(zlib.compress(raw))


def check_heuristics(
    text: str,
    speech_duration_ms: float,
    # Defaults calibrated from p99 + 15% margin on 200 FLEURS vi_vn samples
    # (see data/asr/threshold_calibration.json). Re-run local.calibrate_thresholds
    # after any dataset change. Conservative rounding leaves ~1pp headroom.
    repetition_thresh: float = 0.35,
    ngram_repetition_thresh: float = 0.12,
    min_speech_for_long_text_ms: float = 500,
    chars_per_100ms_max: float = 2.5,
) -> HeuristicCheck:
    """Multi-rule hallucination check. Returns the first matching reason.

    Order matters: cheapest checks first.

    1. Empty → clean (not hallucination, just nothing).
    2. Filler-only → reject.
    3. Unigram repetition above `repetition_thresh` → reject.
    4. 3-gram repetition above `ngram_repetition_thresh` → reject.
    5. If speech < `min_speech_for_long_text_ms` AND text density >
       `chars_per_100ms_max` chars/100ms → reject. Natural Vietnamese
       speech is ~2 chars/100ms; threshold 8 ≈ 4x typical, only triggers
       on clearly impossible density (e.g. 100 char output from 200ms).

    Default thresholds are conservative starting points. Calibrate from
    p99 of real speech distribution via `local/calibrate_thresholds.py`.
    """
    text_clean = text.strip()
    if not text_clean:
        return HeuristicCheck(False, "empty", 0.0)

    if is_filler_only(text_clean):
        return HeuristicCheck(True, "filler_only", 0.85)

    rep = repetition_ratio(text_clean)
    if rep > repetition_thresh:
        return HeuristicCheck(True, f"unigram_rep:{rep:.2f}", min(0.9, rep))

    ngram_rep = n_gram_repetition(text_clean, n=3)
    if ngram_rep > ngram_repetition_thresh:
        return HeuristicCheck(True, f"ngram_rep:{ngram_rep:.2f}", min(0.95, ngram_rep + 0.3))

    if speech_duration_ms < min_speech_for_long_text_ms:
        chars_per_100ms = len(text_clean) / max(speech_duration_ms / 100, 0.1)
        if chars_per_100ms > chars_per_100ms_max:
            return HeuristicCheck(
                True,
                f"text_too_long:{chars_per_100ms:.1f}_chars_per_100ms",
                0.85,
            )

    return HeuristicCheck(False, "clean", 0.0)
