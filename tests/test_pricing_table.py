"""Tests for the static pricing table (OpenAI / Groq / Gemini)."""

from __future__ import annotations

from soca.llm.providers import pricing_table


def test_as_of_label_is_a_nonempty_string() -> None:
    assert isinstance(pricing_table.PRICING_TABLE_AS_OF, str)
    assert pricing_table.PRICING_TABLE_AS_OF.strip()


def test_lookup_returns_floats_when_model_is_present(monkeypatch) -> None:
    # Independent of the shipped (empty) table: verify the lookup mechanism
    # coerces stored numbers to a (float, float) tuple.
    fake = {"as_of": "test", "prices": {"openai": {"demo-model": [1, 2]}}}
    monkeypatch.setattr(pricing_table, "load_pricing_table", lambda: fake)

    price = pricing_table.lookup_pricing("openai", "demo-model")

    assert price == (1.0, 2.0)
    assert all(isinstance(x, float) for x in price)


def test_shipped_table_is_empty_by_design() -> None:
    # Decision (2026-07): ship no hand-typed prices; unknown beats a stale guess.
    table = pricing_table.load_pricing_table()
    assert all(models == {} for models in table["prices"].values())


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
