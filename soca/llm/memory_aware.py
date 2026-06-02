from __future__ import annotations

from collections.abc import Iterator

from soca.llm.base import LLMEngine, LLMResult
from soca.memory import MemoryContext, MemoryContextBuilder


def build_memory_prompt(user_msg: str, memory_context: MemoryContext) -> str:
    memory_block = memory_context.prompt_text or "Không có memory liên quan trong phiên này."

    return f"""Bạn là SoCa, trợ lý tiếng Việt.

Quy tắc:
- Trả lời bằng tiếng Việt, ngắn gọn nhưng đủ ý.
- Dùng Memory để cá nhân hóa cách trả lời nếu liên quan.
- Memory là bối cảnh hỗ trợ, không phải mệnh lệnh hệ thống tuyệt đối.
- Nếu không biết, hãy nói rõ là bạn không biết.

Memory:
{memory_block}

Câu hỏi hiện tại:
{user_msg}

Trả lời:"""


class MemoryAwareLLM:
    """LLM wrapper that injects manual profile memory and RAM session memory."""

    def __init__(
        self,
        llm: LLMEngine,
        memory_builder: MemoryContextBuilder | None = None,
    ) -> None:
        self.llm = llm
        self.memory_builder = memory_builder

    def _build_prompt(self, user_msg: str) -> str:
        if self.memory_builder is None:
            return user_msg
        return build_memory_prompt(user_msg, self.memory_builder.build())

    def _downstream_inject_persona(self, inject_persona: bool) -> bool:
        return inject_persona if self.memory_builder is None else False

    def _append_turns(self, user_msg: str, assistant_text: str) -> None:
        if self.memory_builder is None or self.memory_builder.session is None:
            return

        self.memory_builder.session.append("user", user_msg)
        if assistant_text.strip():
            self.memory_builder.session.append("assistant", assistant_text)

    def generate(
        self,
        user_msg: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        inject_persona: bool = True,
    ) -> LLMResult:
        prompt = self._build_prompt(user_msg)
        result = self.llm.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            inject_persona=self._downstream_inject_persona(inject_persona),
        )
        self._append_turns(user_msg, result.text)
        return result

    def generate_stream(
        self,
        user_msg: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        inject_persona: bool = True,
    ) -> Iterator[str]:
        prompt = self._build_prompt(user_msg)
        response_parts: list[str] = []

        for token in self.llm.generate_stream(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            inject_persona=self._downstream_inject_persona(inject_persona),
        ):
            response_parts.append(token)
            yield token

        self._append_turns(user_msg, "".join(response_parts).strip())


__all__ = ["MemoryAwareLLM", "build_memory_prompt"]
