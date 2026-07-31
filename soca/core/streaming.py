from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np

from soca.core.text_chunking import is_numbered_list_marker, split_first_clause
from soca.tts import TTSResult

_SENTENCE_END_RE = re.compile(r"([.!?\u2026]+)(\s+|$)")

StreamingEventType = Literal[
    "asr",
    "asr_partial",
    "repair",
    "runtime",
    "llm_token",
    "sentence",
    "tts",
    "playback_started",
    "audio",
    "interrupted",
    "done",
    "error",
]


@dataclass(frozen=True)
class StreamingEvent:
    type: StreamingEventType
    text: str = ""
    audio: np.ndarray | None = None
    sample_rate: int | None = None
    tts: TTSResult | None = None
    latency_ms: float | None = None
    metadata: dict | None = None


def audio_duration_ms(sample_count: int, sample_rate: int | None) -> float | None:
    """Duration of a PCM buffer, or ``None`` when the sample rate is unusable.

    A non-positive sample rate means the TTS engine or sink reported a broken
    format. Returning ``None`` keeps the caller from publishing a fabricated
    ``0.0`` duration that downstream consumers would read as "already spoken".
    """
    if sample_rate is None or sample_rate <= 0 or sample_count <= 0:
        return None
    return sample_count / sample_rate * 1000.0


def pop_ready_sentence(buffer: str, min_chars: int = 24) -> tuple[str | None, str]:
    if len(buffer.strip()) < min_chars:
        return None, buffer

    for match in _SENTENCE_END_RE.finditer(buffer):
        if is_numbered_list_marker(buffer, match.start(1), match.group(1)):
            continue
        end_pos = match.end()
        sentence = buffer[:end_pos].strip()
        if len(sentence) >= min_chars:
            return sentence, buffer[end_pos:].lstrip()

    return None, buffer


def pop_ready_first_clause(
    buffer: str,
    *,
    min_chars: int = 12,
    min_words: int = 2,
    max_scan_chars: int = 80,
) -> tuple[str | None, str]:
    clause, remainder = split_first_clause(
        buffer,
        min_chars=min_chars,
        min_words=min_words,
        max_scan_chars=max_scan_chars,
    )
    if clause is None:
        return None, buffer
    return clause, remainder
