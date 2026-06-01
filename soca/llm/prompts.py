from __future__ import annotations

import re
from typing import TypedDict

from .registry import LLMModelConfig

COMPLETION_PROMPT_TEMPLATES = {
    "phogpt_completion": "### Câu hỏi: {prompt}\n### Trả lời:",
}

SON_CA_SYSTEM_PROMPT = (
    "Bạn là Sơn Ca, trợ lý ảo tiếng Việt thông minh, thân thiện. "
    "Trả lời súc tích dưới 50 từ. Nếu không biết, hãy nói rằng bạn không biết."
)
SHRIKE7_SYSTEM_PROMPT = SON_CA_SYSTEM_PROMPT

QWEN_NO_THINK_MARKER = "/no_think"
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


class ChatMessage(TypedDict):
    role: str
    content: str


def build_completion_prompt(
    user_msg: str,
    config: LLMModelConfig,
    inject_persona: bool = True,
) -> str:
    try:
        prompt_template = COMPLETION_PROMPT_TEMPLATES[config.prompt_style]
    except KeyError as exc:
        raise ValueError(f"Model does not use completion prompts: {config.model_key}") from exc

    user_msg = user_msg.strip()
    if inject_persona:
        user_msg = f"{SON_CA_SYSTEM_PROMPT}\n\nCâu hỏi của tôi: {user_msg}"

    return prompt_template.format(prompt=user_msg)


def build_chat_messages(
    user_msg: str,
    config: LLMModelConfig,
    inject_persona: bool = True,
) -> list[ChatMessage]:
    if config.prompt_style == "phogpt_completion":
        raise ValueError("Completion-style models use completion prompts, not chat messages.")

    user_msg = user_msg.strip()
    if config.append_no_think and QWEN_NO_THINK_MARKER not in user_msg:
        user_msg = f"{user_msg}\n{QWEN_NO_THINK_MARKER}"

    messages: list[ChatMessage] = []
    if inject_persona:
        messages.append({"role": "system", "content": SON_CA_SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_msg})
    return messages


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


def uses_completion_prompt(config: LLMModelConfig) -> bool:
    return config.prompt_style == "phogpt_completion"
