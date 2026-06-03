from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np

from soca.core.text_chunking import is_numbered_list_marker
from soca.tts import TTSResult

_SENTENCE_END_RE = re.compile(r"([.!?\u2026]+)(\s+|$)")

StreamingEventType = Literal[
    "asr",
    "repair",
    "runtime",
    "llm_token",
    "sentence",
    "tts",
    "audio",
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
