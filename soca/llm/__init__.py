"""LLM subpackage — Vietnamese language understanding & response generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import LLMEngine, LLMResult
from .memory_aware import MemoryAwareLLM, build_memory_prompt

if TYPE_CHECKING:
    from .llamacpp_runner import LocalLlamaCppLLM

__all__ = [
    "LLMEngine",
    "LLMResult",
    "LocalLlamaCppLLM",
    "MemoryAwareLLM",
    "build_memory_prompt",
]


def __getattr__(name: str):
    # Lazy import so the remote (API) path does not require llama-cpp-python.
    if name == "LocalLlamaCppLLM":
        from .llamacpp_runner import LocalLlamaCppLLM

        return LocalLlamaCppLLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
