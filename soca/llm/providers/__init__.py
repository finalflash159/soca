"""Remote (third-party API) LLM providers for SoCa.

A single OpenAI-compatible client covers OpenAI, Groq, OpenRouter, and Gemini;
each is one entry in :mod:`provider_registry`. Local GGUF models keep their own
registry in :mod:`soca.llm.registry`; this package is the opt-in remote path.
"""

from __future__ import annotations

from .model_catalog import RemoteModelInfo, fetch_catalog, search_models
from .pricing_table import PRICING_TABLE_AS_OF, lookup_pricing
from .provider_registry import PROVIDER_REGISTRY, LLMProvider, get_provider
from .remote_openai_llm import (
    ProviderCallTrace,
    RemoteFailureKind,
    RemoteLLMError,
    RemoteOpenAILLM,
    RetryPolicy,
)

__all__ = [
    "PRICING_TABLE_AS_OF",
    "PROVIDER_REGISTRY",
    "LLMProvider",
    "ProviderCallTrace",
    "RemoteFailureKind",
    "RemoteLLMError",
    "RemoteModelInfo",
    "RemoteOpenAILLM",
    "RetryPolicy",
    "fetch_catalog",
    "get_provider",
    "lookup_pricing",
    "search_models",
]
