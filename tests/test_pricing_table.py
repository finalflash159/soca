"""Tests for the static pricing table (OpenAI / Groq / Gemini)."""

from __future__ import annotations

from soca.llm.providers import pricing_table


def test_as_of_label_is_a_nonempty_string() -> None:
    assert isinstance(pricing_table.PRICING_TABLE_AS_OF, str)
    assert pricing_table.PRICING_TABLE_AS_OF.strip()


def test_lookup_known_model_returns_prompt_and_completion_floats() -> None:
    price = pricing_table.lookup_pricing("openai", "gpt-4o-mini")
    assert price is not None
    prompt_per_1m, completion_per_1m = price
    assert isinstance(prompt_per_1m, float)
    assert isinstance(completion_per_1m, float)
    # Completion is never cheaper than prompt for these providers.
    assert completion_per_1m >= prompt_per_1m >= 0.0


def test_lookup_unknown_model_returns_none() -> None:
    assert pricing_table.lookup_pricing("openai", "no-such-model-xyz") is None


def test_lookup_unknown_provider_returns_none() -> None:
    assert pricing_table.lookup_pricing("not-a-provider", "gpt-4o-mini") is None


def test_openrouter_has_no_static_prices() -> None:
    # OpenRouter pricing is fetched live; it must not be in the static table.
    table = pricing_table.load_pricing_table()
    assert "openrouter" not in table["prices"]


def test_all_table_entries_are_two_nonnegative_numbers() -> None:
    table = pricing_table.load_pricing_table()
    for provider_key, models in table["prices"].items():
        for model_id, pair in models.items():
            assert len(pair) == 2, f"{provider_key}/{model_id} must be [prompt, completion]"
            prompt, completion = pair
            assert prompt >= 0 and completion >= 0
