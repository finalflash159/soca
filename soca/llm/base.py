from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Protocol


@dataclass
class LLMResult:
    text: str
    prompt: str
    n_prompt_tokens: int
    n_completion_tokens: int
    ttft_ms: float
    total_latency_ms: float
    tokens_per_second: float

    def to_dict(self) -> dict:
        return asdict(self)


class LLMEngine(Protocol):
    def generate(
        self,
        user_msg: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        inject_persona: bool = True,
    ) -> LLMResult:
        ...

    def generate_stream(
        self,
        user_msg: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        inject_persona: bool = True,
    ) -> Iterator[str]:
        ...
