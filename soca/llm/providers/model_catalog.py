"""Fetch and search the remote model catalog for a provider.

Every provider exposes an OpenAI-style ``/models`` endpoint, but only OpenRouter
returns per-token pricing there. For OpenRouter the price is parsed live; for the
others it is merged from the committed static table (:mod:`pricing_table`) and
falls back to ``pricing_source="unknown"`` when a model is not listed — we never
invent a price.

The HTTP getter is injectable so the whole path is testable without network.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .pricing_table import lookup_pricing
from .provider_registry import LLMProvider
from .remote_openai_llm import RemoteLLMError

# A getter takes (url, headers) and returns the parsed JSON body as a dict.
HttpGetter = Callable[[str, dict[str, str]], dict[str, Any]]

_HTTP_TIMEOUT_S = 30.0
# OpenRouter prices are per-token strings; scale to price per 1M tokens.
_TOKENS_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class RemoteModelInfo:
    id: str
    label: str
    context_length: int | None
    price_prompt_per_1m: float | None
    price_completion_per_1m: float | None
    # "live" (from provider API) | "table" (static table) | "unknown".
    pricing_source: str


def fetch_catalog(
    provider: LLMProvider,
    api_key: str,
    *,
    http: HttpGetter | None = None,
) -> list[RemoteModelInfo]:
    """Return the provider's models with pricing merged in.

    OpenRouter pricing is read live from the response; other providers are
    matched against the static table. Network/HTTP failures are raised as a
    friendly :class:`RemoteLLMError`.
    """
    getter = http or _default_http_get
    url = provider.base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = getter(url, headers)

    raw_models = payload.get("data") or []
    return [_to_model_info(provider, raw) for raw in raw_models]


def search_models(catalog: list[RemoteModelInfo], query: str) -> list[RemoteModelInfo]:
    """Filter a catalog by a keyword query (case-insensitive, tokens AND-ed).

    An empty/whitespace query returns the full catalog. Each whitespace-separated
    token must appear as a substring of the model id or label.
    """
    tokens = query.lower().split()
    if not tokens:
        return catalog

    def matches(model: RemoteModelInfo) -> bool:
        haystack = f"{model.id} {model.label}".lower()
        return all(token in haystack for token in tokens)

    return [model for model in catalog if matches(model)]


# -- internals ---------------------------------------------------------------


def _normalize_model_id(raw_id: str) -> str:
    """Strip Gemini's ``models/`` prefix so ids match the API + pricing table."""
    prefix = "models/"
    return raw_id[len(prefix):] if raw_id.startswith(prefix) else raw_id


def _to_model_info(provider: LLMProvider, raw: dict[str, Any]) -> RemoteModelInfo:
    model_id = _normalize_model_id(raw["id"])
    label = raw.get("name") or model_id
    context_length = raw.get("context_length") or raw.get("context_window")

    prompt_per_1m, completion_per_1m, source = _resolve_pricing(provider, model_id, raw)

    return RemoteModelInfo(
        id=model_id,
        label=label,
        context_length=context_length,
        price_prompt_per_1m=prompt_per_1m,
        price_completion_per_1m=completion_per_1m,
        pricing_source=source,
    )


def _resolve_pricing(
    provider: LLMProvider,
    model_id: str,
    raw: dict[str, Any],
) -> tuple[float | None, float | None, str]:
    if provider.has_pricing_api:
        pricing = raw.get("pricing") or {}
        prompt = _per_token_to_per_1m(pricing.get("prompt"))
        completion = _per_token_to_per_1m(pricing.get("completion"))
        if prompt is not None or completion is not None:
            return prompt, completion, "live"
        return None, None, "unknown"

    table = lookup_pricing(provider.key, model_id)
    if table is None:
        return None, None, "unknown"
    return table[0], table[1], "table"


def _per_token_to_per_1m(value: Any) -> float | None:
    """Convert an OpenRouter per-token price string to USD per 1M tokens."""
    if value is None or value == "":
        return None
    try:
        return float(value) * _TOKENS_PER_MILLION
    except (TypeError, ValueError):
        return None


def _default_http_get(url: str, headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=headers)  # noqa: S310 - trusted provider URL
    try:
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_S) as response:  # noqa: S310
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise _http_error_to_remote(exc.code) from exc
    except urllib.error.URLError as exc:
        raise RemoteLLMError(
            "Không kết nối được để lấy danh sách model. Kiểm tra mạng và thử lại.",
            category="network",
        ) from exc


def _http_error_to_remote(status: int) -> RemoteLLMError:
    if status in (401, 403):
        return RemoteLLMError("API key sai hoặc hết hạn.", category="auth")
    if status == 429:
        return RemoteLLMError(
            "Đã hết quota hoặc bị giới hạn tốc độ khi lấy danh sách model.",
            category="rate_limit",
        )
    return RemoteLLMError(
        f"Lỗi khi lấy danh sách model (HTTP {status}).", category="unknown"
    )


__all__ = ["HttpGetter", "RemoteModelInfo", "fetch_catalog", "search_models"]
