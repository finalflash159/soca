"""LLM subpackage — Vietnamese language understanding & response generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import LLMEngine, LLMResult, StructuredLLMEngine

if TYPE_CHECKING:
    from .llamacpp_runner import LocalLlamaCppLLM
    from .memory_aware import MemoryAwareLLM, build_memory_prompt

__all__ = [
    "LLMEngine",
    "LLMResult",
    "StructuredLLMEngine",
    "LocalLlamaCppLLM",
    "MemoryAwareLLM",
    "build_memory_prompt",
]


def __getattr__(name: str):
    # Lazy import so the remote (API) path does not require llama-cpp-python.
    if name == "LocalLlamaCppLLM":
        from .llamacpp_runner import LocalLlamaCppLLM

        return LocalLlamaCppLLM
    if name == "MemoryAwareLLM":
        from .memory_aware import MemoryAwareLLM

        return MemoryAwareLLM
    if name == "build_memory_prompt":
        from .memory_aware import build_memory_prompt

        return build_memory_prompt
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
