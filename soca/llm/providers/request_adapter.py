from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .provider_registry import LLMProvider

ReasoningParameter = Literal["reasoning", "reasoning_effort"]


@dataclass(frozen=True)
class EffectiveGeneration:
    requested_max_tokens: int
    max_tokens: int
    reasoning_enabled: bool | None
    reasoning_parameter: ReasoningParameter | None


class ProviderRequestAdapter:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def generation_options(
        self,
        *,
        max_tokens: int,
        reasoning_enabled: bool | None,
        reasoning_parameter: ReasoningParameter | None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            self.provider.output_token_parameter: max_tokens,
        }
        if self.provider.supports_upstream_fallback_control:
            options["extra_body"] = {"provider": {"allow_fallbacks": False}}
        if reasoning_enabled is None or reasoning_parameter is None:
            return options

        if reasoning_parameter == "reasoning":
            if self.provider.reasoning_transport != "openrouter":
                raise ValueError(
                    f"provider {self.provider.key} does not accept the unified reasoning object"
                )
            extra_body = dict(options.get("extra_body", {}))
            extra_body["reasoning"] = (
                {"enabled": True, "exclude": True}
                if reasoning_enabled
                else {"effort": "none"}
            )
            options["extra_body"] = extra_body
            return options

        options["reasoning_effort"] = "medium" if reasoning_enabled else "none"
        return options

    def structured_options(self, *, zero_data_retention: bool) -> dict[str, Any]:
        if not self.provider.supports_zero_data_retention_routing:
            return {}
        return {
            "extra_body": {
                "provider": {
                    # Keep the selected model and explicit no-fallback policy,
                    # but do not force OpenRouter's provider-level
                    # `require_parameters` filter here. Some live model
                    # catalogs advertise response_format while their eligible
                    # endpoint rejects that routing filter with HTTP 404. The
                    # response is still schema-validated by the local parser;
                    # unsupported output is a typed failure, never a fallback.
                    "data_collection": "deny" if zero_data_retention else "allow",
                }
            }
        }

    @staticmethod
    def merge_options(*values: Mapping[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        extra_body: dict[str, Any] = {}
        for value in values:
            for key, item in value.items():
                if key == "extra_body":
                    if not isinstance(item, Mapping):
                        raise TypeError("provider extra_body must be a mapping")
                    _merge_mapping(extra_body, item)
                else:
                    merged[key] = item
        if extra_body:
            merged["extra_body"] = extra_body
        return merged


def _merge_mapping(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        current = target.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            _merge_mapping(current, value)
        else:
            target[key] = value


__all__ = ["EffectiveGeneration", "ProviderRequestAdapter", "ReasoningParameter"]
