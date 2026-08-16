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


_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})", re.MULTILINE)


def pop_ready_block(buffer: str, max_chars: int = 400) -> tuple[str | None, str]:
    """Pop one complete markdown block, for a surface that renders markdown.

    The sentence and first-clause splitters exist to get audio out early and to
    give TTS audible pauses. On a screen they are wrong twice over: they strip
    the newlines that make a list a list, and `pop_ready_first_clause` cuts
    mid-sentence at a comma, which is a fine place to breathe and a poor place
    to stop reading.

    A block ends at a blank line, so nothing has to be invented — the model
    already wrote the boundary. Fenced code is exempt: a blank line inside a
    snippet belongs to the snippet, and the block ends at the closing fence.

    `max_chars` is the escape hatch. A single long paragraph carries no blank
    line at all, and without this nothing would appear until the turn closed;
    past that length the block is released at the next sentence end instead.
    The cost is that one long paragraph may show as two while streaming, which
    `chat/done.text` then corrects.
    """
    if not buffer.strip():
        return None, buffer

    fences = 0
    offset = 0
    for line in buffer.splitlines(keepends=True):
        offset += len(line)
        if _FENCE_RE.match(line):
            fences += 1
            # A closing fence completes the block on its own line.
            if fences % 2 == 0:
                block = buffer[:offset].strip()
                if block:
                    return block, buffer[offset:].lstrip("\n")
            continue
        if fences % 2 == 1:
            continue
        if not line.strip():
            block = buffer[:offset].strip()
            if block:
                return block, buffer[offset:].lstrip("\n")

    if fences % 2 == 0 and len(buffer.strip()) >= max_chars:
        return pop_ready_sentence(buffer, min_chars=max_chars)
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
