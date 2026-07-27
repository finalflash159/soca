from __future__ import annotations

import pytest

from soca.llm.providers.provider_registry import (
    PROVIDER_REGISTRY,
    LLMProvider,
    get_provider,
)

EXPECTED_KEYS = {"openai", "gemini", "openrouter", "groq"}


def test_registry_has_the_four_providers():
    assert set(PROVIDER_REGISTRY) == EXPECTED_KEYS


def test_every_provider_is_frozen_and_openai_compatible():
    for key, provider in PROVIDER_REGISTRY.items():
        assert isinstance(provider, LLMProvider)
        assert provider.key == key
        assert provider.base_url.startswith("https://")
        assert provider.api_key_env.endswith("_API_KEY")
        # Only OpenRouter exposes pricing through its models API.
        assert isinstance(provider.has_pricing_api, bool)


def test_only_openrouter_advertises_a_pricing_api():
    with_pricing = {k for k, p in PROVIDER_REGISTRY.items() if p.has_pricing_api}
    assert with_pricing == {"openrouter"}


def test_gemini_uses_the_openai_compat_shim():
    gemini = get_provider("gemini")
    assert gemini.base_url.rstrip("/").endswith("/openai")


def test_get_provider_returns_the_matching_provider():
    provider = get_provider("groq")
    assert provider.key == "groq"
    assert "groq.com" in provider.base_url


def test_get_provider_rejects_unknown_key_and_lists_valid_ones():
    with pytest.raises(ValueError) as excinfo:
        get_provider("anthropic")
    message = str(excinfo.value)
    assert "anthropic" in message
    for key in EXPECTED_KEYS:
        assert key in message


def test_provider_is_immutable():
    provider = get_provider("openai")
    with pytest.raises(Exception):
        provider.base_url = "https://evil.example"  # type: ignore[misc]
