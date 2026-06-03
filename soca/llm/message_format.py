from __future__ import annotations

from typing import TypedDict

from soca.prompts import SOCA_LLM_SYSTEM_PROMPT, split_embedded_system_prompt

from .registry import LLMModelConfig

COMPLETION_PROMPT_TEMPLATES = {
    "phogpt_completion": "### Câu hỏi: {prompt}\n### Trả lời:",
}

QWEN_NO_THINK_MARKER = "/no_think"


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
        user_msg = f"{SOCA_LLM_SYSTEM_PROMPT}\n\nCâu hỏi của tôi: {user_msg}"

    return prompt_template.format(prompt=user_msg)


def build_chat_messages(
    user_msg: str,
    config: LLMModelConfig,
    inject_persona: bool = True,
) -> list[ChatMessage]:
    if config.prompt_style == "phogpt_completion":
        raise ValueError("Completion-style models use completion prompts, not chat messages.")

    messages: list[ChatMessage] = []
    if inject_persona:
        messages.append({"role": "system", "content": SOCA_LLM_SYSTEM_PROMPT})
        user_content = user_msg.strip()
    else:
        system_prompt, user_content = split_embedded_system_prompt(user_msg)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

    user_content = user_content.strip()
    if config.append_no_think and QWEN_NO_THINK_MARKER not in user_content:
        user_content = f"{user_content}\n{QWEN_NO_THINK_MARKER}"

    messages.append({"role": "user", "content": user_content})
    return messages


def uses_completion_prompt(config: LLMModelConfig) -> bool:
    return config.prompt_style == "phogpt_completion"


__all__ = [
    "ChatMessage",
    "QWEN_NO_THINK_MARKER",
    "build_chat_messages",
    "build_completion_prompt",
    "uses_completion_prompt",
]
