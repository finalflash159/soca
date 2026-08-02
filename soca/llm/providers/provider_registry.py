"""Remote provider endpoints and protocol capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OutputTokenParameter = Literal["max_tokens", "max_completion_tokens"]
ReasoningTransport = Literal["openai", "openrouter"]


@dataclass(frozen=True)
class LLMProvider:
    key: str
    label: str
    base_url: str
    api_key_env: str
    # OpenRouter is the only provider whose /models endpoint returns pricing.
    has_pricing_api: bool
    models_docs_url: str
    output_token_parameter: OutputTokenParameter
    reasoning_transport: ReasoningTransport
    supports_zero_data_retention_routing: bool = False
    supports_upstream_fallback_control: bool = False


PROVIDER_REGISTRY: dict[str, LLMProvider] = {
    "openai": LLMProvider(
        key="openai",
        label="OpenAI",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        has_pricing_api=False,
        models_docs_url="https://platform.openai.com/docs/api-reference/models",
        output_token_parameter="max_completion_tokens",
        reasoning_transport="openai",
    ),
    "gemini": LLMProvider(
        key="gemini",
        label="Gemini",
        # Google's OpenAI-compatibility shim maps /chat/completions + /models.
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
        has_pricing_api=False,
        models_docs_url="https://ai.google.dev/gemini-api/docs/openai",
        output_token_parameter="max_tokens",
        reasoning_transport="openai",
    ),
    "openrouter": LLMProvider(
        key="openrouter",
        label="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        has_pricing_api=True,
        models_docs_url="https://openrouter.ai/docs/api-reference/list-available-models",
        output_token_parameter="max_tokens",
        reasoning_transport="openrouter",
        supports_zero_data_retention_routing=True,
        supports_upstream_fallback_control=True,
    ),
    "groq": LLMProvider(
        key="groq",
        label="Groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        has_pricing_api=False,
        models_docs_url="https://console.groq.com/docs/models",
        output_token_parameter="max_completion_tokens",
        reasoning_transport="openai",
    ),
}


def get_provider(key: str) -> LLMProvider:
    try:
        return PROVIDER_REGISTRY[key]
    except KeyError as exc:
        valid = ", ".join(sorted(PROVIDER_REGISTRY))
        raise ValueError(f"Unknown LLM provider '{key}'. Valid providers: {valid}.") from exc


__all__ = [
    "LLMProvider",
    "OutputTokenParameter",
    "PROVIDER_REGISTRY",
    "ReasoningTransport",
    "get_provider",
]
