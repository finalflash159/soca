from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class LLMResult:
    text: str
    prompt: str
    n_prompt_tokens: int
    n_completion_tokens: int
    ttft_ms: float
    total_latency_ms: float
    tokens_per_second: float
    provider_trace: Mapping[str, Any] = field(default_factory=dict)

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


@runtime_checkable
class StructuredLLMEngine(Protocol):
    def generate_structured(
        self,
        user_msg: str,
        *,
        schema_name: str,
        schema: Mapping[str, Any],
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        inject_persona: bool = False,
        zero_data_retention: bool = True,
    ) -> LLMResult:
        ...
