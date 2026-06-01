from __future__ import annotations

from collections.abc import Iterator

from soca.llm import LLMResult, MemoryAwareLLM, build_memory_prompt
from soca.memory import MemoryContext, MemoryContextBuilder, SessionMemory


class FakeLongTermMemory:
    def read_profile(self) -> str:
        return "- Người dùng thích được giải thích kỹ bằng tiếng Việt."


class SpyLLM:
    def __init__(self) -> None:
        self.generate_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    def generate(
        self,
        user_msg: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        inject_persona: bool = True,
    ) -> LLMResult:
        self.generate_calls.append(
            {
                "user_msg": user_msg,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "inject_persona": inject_persona,
            }
        )
        return LLMResult(
            text="Mình sẽ giải thích kỹ.",
            prompt=user_msg,
            n_prompt_tokens=10,
            n_completion_tokens=5,
            ttft_ms=1.0,
            total_latency_ms=2.0,
            tokens_per_second=100.0,
        )

    def generate_stream(
        self,
        user_msg: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        inject_persona: bool = True,
    ) -> Iterator[str]:
        self.stream_calls.append(
            {
                "user_msg": user_msg,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "inject_persona": inject_persona,
            }
        )
        yield "Xin chào. "
        yield "Mình nhớ cách bạn thích được trả lời."


def test_build_memory_prompt_includes_memory_and_user_message() -> None:
    context = MemoryContext(
        profile_text="- thích tiếng Việt",
        session_text="Recent conversation:\nUser: hỏi cũ",
        prompt_text="Long-term memory:\n- thích tiếng Việt",
    )

    prompt = build_memory_prompt("Bạn nhớ gì về tôi?", context)

    assert "Bạn là Sơn Ca" in prompt
    assert "Long-term memory:" in prompt
    assert "thích tiếng Việt" in prompt
    assert "Bạn nhớ gì về tôi?" in prompt
    assert "Memory là bối cảnh hỗ trợ" in prompt


def test_generate_injects_memory_and_updates_session_after_success() -> None:
    spy = SpyLLM()
    session = SessionMemory()
    builder = MemoryContextBuilder(
        long_term=FakeLongTermMemory(),
        session=session,
    )
    llm = MemoryAwareLLM(spy, memory_builder=builder)

    result = llm.generate(
        "Tôi thích kiểu trả lời nào?",
        max_tokens=64,
        temperature=0.2,
        top_p=0.9,
    )

    assert result.text == "Mình sẽ giải thích kỹ."
    assert spy.generate_calls[0]["inject_persona"] is False
    assert spy.generate_calls[0]["max_tokens"] == 64
    assert spy.generate_calls[0]["temperature"] == 0.2
    assert spy.generate_calls[0]["top_p"] == 0.9
    assert "Người dùng thích được giải thích kỹ" in spy.generate_calls[0]["user_msg"]
    assert "Tôi thích kiểu trả lời nào?" in spy.generate_calls[0]["user_msg"]
    assert [turn.role for turn in session.turns] == ["user", "assistant"]
    assert session.turns[0].text == "Tôi thích kiểu trả lời nào?"
    assert session.turns[1].text == "Mình sẽ giải thích kỹ."


def test_generate_stream_yields_tokens_and_updates_session_after_stream_finishes() -> None:
    spy = SpyLLM()
    session = SessionMemory()
    builder = MemoryContextBuilder(
        long_term=FakeLongTermMemory(),
        session=session,
    )
    llm = MemoryAwareLLM(spy, memory_builder=builder)

    stream = llm.generate_stream("Xin chào", max_tokens=32, temperature=0.1)

    first = next(stream)
    assert first == "Xin chào. "
    assert session.turns == ()

    rest = list(stream)

    assert rest == ["Mình nhớ cách bạn thích được trả lời."]
    assert spy.stream_calls[0]["inject_persona"] is False
    assert spy.stream_calls[0]["max_tokens"] == 32
    assert spy.stream_calls[0]["temperature"] == 0.1
    assert [turn.role for turn in session.turns] == ["user", "assistant"]
    assert session.turns[1].text == "Xin chào. Mình nhớ cách bạn thích được trả lời."


def test_without_memory_builder_delegates_original_message() -> None:
    spy = SpyLLM()
    llm = MemoryAwareLLM(spy)

    llm.generate("Tin nhắn gốc")

    assert spy.generate_calls[0]["user_msg"] == "Tin nhắn gốc"
    assert spy.generate_calls[0]["inject_persona"] is True
