"""Tests for remote model catalog fetch + search (no network)."""

from __future__ import annotations

from typing import Any

import pytest

from soca.llm.providers import (
    RemoteModelInfo,
    fetch_catalog,
    get_provider,
    search_models,
)


class FakeHttp:
    """Records the request and returns a canned JSON payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, headers: dict[str, str]) -> dict[str, Any]:
        self.calls.append((url, headers))
        return self.payload


# -- OpenRouter live pricing -------------------------------------------------


def _openrouter_payload() -> dict[str, Any]:
    return {
        "data": [
            {
                "id": "meta-llama/llama-3.3-70b-instruct",
                "name": "Meta: Llama 3.3 70B Instruct",
                "context_length": 131072,
                # per-token price strings, as OpenRouter returns them.
                "pricing": {"prompt": "0.00000059", "completion": "0.00000079"},
            },
            {
                "id": "free/model",
                "name": "A Free Model",
                "context_length": 8192,
                "pricing": {"prompt": "0", "completion": "0"},
            },
        ]
    }


def test_openrouter_parses_live_pricing_per_million_tokens() -> None:
    provider = get_provider("openrouter")
    http = FakeHttp(_openrouter_payload())

    catalog = fetch_catalog(provider, "sk-test", http=http)

    first = catalog[0]
    assert isinstance(first, RemoteModelInfo)
    assert first.id == "meta-llama/llama-3.3-70b-instruct"
    assert first.label == "Meta: Llama 3.3 70B Instruct"
    assert first.context_length == 131072
    assert first.pricing_source == "live"
    # 0.00000059 USD/token * 1e6 = 0.59 USD / 1M tokens.
    assert first.price_prompt_per_1m == pytest.approx(0.59)
    assert first.price_completion_per_1m == pytest.approx(0.79)


def test_openrouter_free_model_prices_are_zero_not_none() -> None:
    provider = get_provider("openrouter")
    catalog = fetch_catalog(provider, "sk-test", http=FakeHttp(_openrouter_payload()))

    free = catalog[1]
    assert free.pricing_source == "live"
    assert free.price_prompt_per_1m == 0.0
    assert free.price_completion_per_1m == 0.0


def test_fetch_sends_bearer_token_and_models_endpoint() -> None:
    provider = get_provider("openrouter")
    http = FakeHttp(_openrouter_payload())

    fetch_catalog(provider, "sk-secret", http=http)

    url, headers = http.calls[0]
    assert url.endswith("/models")
    assert provider.base_url.rstrip("/") in url
    assert headers["Authorization"] == "Bearer sk-secret"


# -- Table / unknown fallback for non-pricing providers ----------------------


def _openai_payload() -> dict[str, Any]:
    return {
        "data": [
            {"id": "gpt-4o-mini", "object": "model"},
            {"id": "some-unlisted-model", "object": "model"},
        ]
    }


def test_openai_known_model_uses_static_table() -> None:
    provider = get_provider("openai")
    catalog = fetch_catalog(provider, "sk-test", http=FakeHttp(_openai_payload()))

    known = catalog[0]
    assert known.id == "gpt-4o-mini"
    assert known.pricing_source == "table"
    assert known.price_prompt_per_1m is not None
    assert known.price_completion_per_1m is not None


def test_openai_unknown_model_is_marked_unknown_not_fabricated() -> None:
    provider = get_provider("openai")
    catalog = fetch_catalog(provider, "sk-test", http=FakeHttp(_openai_payload()))

    unknown = catalog[1]
    assert unknown.id == "some-unlisted-model"
    assert unknown.pricing_source == "unknown"
    assert unknown.price_prompt_per_1m is None
    assert unknown.price_completion_per_1m is None


def test_gemini_models_prefix_is_stripped_and_priced_from_table() -> None:
    # Gemini's /models returns ids like "models/gemini-2.0-flash"; the prefix
    # must be normalized so both the id and the table lookup are clean.
    provider = get_provider("gemini")
    payload = {"data": [{"id": "models/gemini-2.0-flash"}]}

    catalog = fetch_catalog(provider, "sk-test", http=FakeHttp(payload))

    assert catalog[0].id == "gemini-2.0-flash"
    assert catalog[0].pricing_source == "table"
    assert catalog[0].price_prompt_per_1m is not None


def test_groq_context_window_key_is_read() -> None:
    provider = get_provider("groq")
    payload = {"data": [{"id": "llama-3.1-8b-instant", "context_window": 131072}]}

    catalog = fetch_catalog(provider, "sk-test", http=FakeHttp(payload))

    assert catalog[0].context_length == 131072


def test_missing_context_is_none() -> None:
    provider = get_provider("openai")
    payload = {"data": [{"id": "gpt-4o-mini"}]}

    catalog = fetch_catalog(provider, "sk-test", http=FakeHttp(payload))

    assert catalog[0].context_length is None


# -- Label fallback ----------------------------------------------------------


def test_label_falls_back_to_id_when_name_missing() -> None:
    provider = get_provider("groq")
    payload = {"data": [{"id": "llama-3.1-8b-instant"}]}

    catalog = fetch_catalog(provider, "sk-test", http=FakeHttp(payload))

    assert catalog[0].label == "llama-3.1-8b-instant"


# -- search_models -----------------------------------------------------------


def _sample_catalog() -> list[RemoteModelInfo]:
    return [
        RemoteModelInfo("meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B", 131072, 0.59, 0.79, "live"),
        RemoteModelInfo("qwen/qwen-2.5-72b-instruct", "Qwen 2.5 72B", 32768, 0.4, 0.4, "live"),
        RemoteModelInfo("gpt-4o-mini", "GPT-4o mini", None, 0.15, 0.6, "table"),
    ]


def test_search_empty_query_returns_full_catalog() -> None:
    catalog = _sample_catalog()
    assert search_models(catalog, "") == catalog
    assert search_models(catalog, "   ") == catalog


def test_search_is_case_insensitive_substring_on_id_and_label() -> None:
    catalog = _sample_catalog()
    result = search_models(catalog, "QWEN")
    assert [m.id for m in result] == ["qwen/qwen-2.5-72b-instruct"]


def test_search_matches_label_text() -> None:
    catalog = _sample_catalog()
    result = search_models(catalog, "mini")
    assert [m.id for m in result] == ["gpt-4o-mini"]


def test_search_tokens_are_anded() -> None:
    catalog = _sample_catalog()
    # Both tokens must appear (in id or label), order-independent.
    result = search_models(catalog, "llama 70b")
    assert [m.id for m in result] == ["meta-llama/llama-3.3-70b-instruct"]


def test_search_no_match_returns_empty() -> None:
    assert search_models(_sample_catalog(), "nonexistent") == []
