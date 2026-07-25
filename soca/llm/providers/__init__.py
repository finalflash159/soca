"""Remote (third-party API) LLM providers for SoCa.

A single OpenAI-compatible client covers OpenAI, Groq, OpenRouter, and Gemini;
each is one entry in :mod:`provider_registry`. Local GGUF models keep their own
registry in :mod:`soca.llm.registry`; this package is the opt-in remote path.
"""

from __future__ import annotations

from .provider_registry import PROVIDER_REGISTRY, LLMProvider, get_provider
from .remote_openai_llm import RemoteLLMError, RemoteOpenAILLM

__all__ = [
    "PROVIDER_REGISTRY",
    "LLMProvider",
    "get_provider",
    "RemoteLLMError",
    "RemoteOpenAILLM",
]
