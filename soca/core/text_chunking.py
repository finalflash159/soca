from __future__ import annotations

import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\u2026])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []

    parts = _SENTENCE_SPLIT_RE.split(cleaned)
    return [part.strip() for part in parts if part.strip()]


def chunk_text_for_tts(text: str, min_chars: int = 24) -> list[str]:
    """Split text into TTS-friendly chunks without tiny leading fragments."""
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    pending = ""

    for sentence in sentences:
        candidate = f"{pending} {sentence}".strip() if pending else sentence
        if len(candidate) < min_chars:
            pending = candidate
            continue

        chunks.append(candidate)
        pending = ""

    if pending:
        if chunks:
            chunks[-1] = f"{chunks[-1]} {pending}".strip()
        else:
            chunks.append(pending)

    return chunks
