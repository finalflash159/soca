"""LLM subpackage — Vietnamese language understanding & response generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import LLMEngine, LLMResult, StructuredLLMEngine

if TYPE_CHECKING:
    from .llamacpp_runner import LocalLlamaCppLLM

__all__ = [
    "LLMEngine",
    "LLMResult",
    "StructuredLLMEngine",
    "LocalLlamaCppLLM",
]


def __getattr__(name: str):
    # Lazy import so the remote (API) path does not require llama-cpp-python.
    if name == "LocalLlamaCppLLM":
        from .llamacpp_runner import LocalLlamaCppLLM

        return LocalLlamaCppLLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
