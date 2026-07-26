"""Static, hand-maintained pricing table for providers without a pricing API.

OpenRouter returns per-token prices from its `/models` endpoint (parsed live in
:mod:`model_catalog`). OpenAI, Groq, and Gemini do not, so their list prices are
curated here from a committed JSON file (config, not a heavy artifact) with an
``as of`` label. Any model absent from the table is reported as pricing unknown
rather than a fabricated number.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_PRICING_PATH = Path(__file__).with_name("pricing_table.json")


@lru_cache(maxsize=1)
def load_pricing_table() -> dict[str, Any]:
    """Load and cache the committed pricing table JSON."""
    with _PRICING_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


PRICING_TABLE_AS_OF: str = load_pricing_table()["as_of"]


def lookup_pricing(provider_key: str, model_id: str) -> tuple[float, float] | None:
    """Return ``(prompt_per_1m, completion_per_1m)`` or ``None`` if unknown.

    Never guesses: an unknown provider or model yields ``None`` so callers can
    surface "giá: không rõ" instead of a made-up figure.
    """
    prices = load_pricing_table()["prices"]
    provider_prices = prices.get(provider_key)
    if not provider_prices:
        return None
    pair = provider_prices.get(model_id)
    if pair is None:
        return None
    prompt_per_1m, completion_per_1m = pair
    return float(prompt_per_1m), float(completion_per_1m)


__all__ = ["PRICING_TABLE_AS_OF", "load_pricing_table", "lookup_pricing"]
