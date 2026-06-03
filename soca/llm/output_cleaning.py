from __future__ import annotations

import re

from .registry import LLMModelConfig

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def clean_model_output(text: str, config: LLMModelConfig) -> str:
    text = text.strip()
    if config.strip_reasoning:
        text = THINK_BLOCK_RE.sub("", text).strip()
    return text


def _split_reasoning_safe_prefix(text: str) -> tuple[str, str]:
    """Hold back suffixes that may become a split '<think>' tag."""
    tag = "<think>"
    lower_text = text.lower()
    for suffix_len in range(min(len(tag) - 1, len(text)), 0, -1):
        if tag.startswith(lower_text[-suffix_len:]):
            return text[:-suffix_len], text[-suffix_len:]
    return text, ""


class StreamingOutputCleaner:
    """Incrementally remove reasoning blocks before text reaches TTS."""

    def __init__(self, config: LLMModelConfig) -> None:
        self.strip_reasoning = config.strip_reasoning
        self._buffer = ""
        self._inside_reasoning = False

    def feed(self, text: str) -> list[str]:
        if not text:
            return []
        if not self.strip_reasoning:
            return [text]

        self._buffer += text
        chunks: list[str] = []

        while self._buffer:
            lower_buffer = self._buffer.lower()

            if self._inside_reasoning:
                close_index = lower_buffer.find("</think>")
                if close_index == -1:
                    return chunks
                self._buffer = self._buffer[close_index + len("</think>") :].lstrip()
                self._inside_reasoning = False
                continue

            open_index = lower_buffer.find("<think>")
            if open_index == -1:
                emit, holdback = _split_reasoning_safe_prefix(self._buffer)
                if emit:
                    chunks.append(emit)
                self._buffer = holdback
                return chunks

            prefix = self._buffer[:open_index].rstrip()
            if prefix:
                chunks.append(prefix)
            self._buffer = self._buffer[open_index + len("<think>") :]
            self._inside_reasoning = True

        return chunks

    def flush(self) -> list[str]:
        if not self.strip_reasoning:
            chunk = self._buffer
            self._buffer = ""
            return [chunk] if chunk else []

        if self._inside_reasoning:
            self._buffer = ""
            return []

        chunk = self._buffer.strip()
        self._buffer = ""
        return [chunk] if chunk else []


__all__ = [
    "StreamingOutputCleaner",
    "clean_model_output",
]
