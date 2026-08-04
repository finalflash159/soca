from __future__ import annotations

import pytest

from soca.llm.providers.provider_registry import get_provider
from soca.llm.providers.request_adapter import ProviderRequestAdapter


@pytest.mark.parametrize(
    ("provider_key", "parameter"),
    [
        ("openai", "max_completion_tokens"),
        ("groq", "max_completion_tokens"),
        ("gemini", "max_tokens"),
        ("openrouter", "max_tokens"),
    ],
)
def test_provider_uses_its_declared_output_token_parameter(
    provider_key: str, parameter: str
) -> None:
    adapter = ProviderRequestAdapter(get_provider(provider_key))

    options = adapter.generation_options(
        max_tokens=4_096,
        reasoning_enabled=None,
        reasoning_parameter=None,
    )

    assert options[parameter] == 4_096
    if provider_key == "openrouter":
        assert options["extra_body"] == {"provider": {"allow_fallbacks": False}}
    else:
        assert set(options) == {parameter}


def test_openrouter_unified_reasoning_uses_extra_body() -> None:
    adapter = ProviderRequestAdapter(get_provider("openrouter"))

    options = adapter.generation_options(
        max_tokens=4_096,
        reasoning_enabled=True,
        reasoning_parameter="reasoning",
    )

    assert options["extra_body"] == {
        "provider": {"allow_fallbacks": False},
        "reasoning": {"enabled": True, "exclude": True}
    }


@pytest.mark.parametrize("provider_key", ["openai", "gemini", "groq"])
def test_openai_compatible_reasoning_effort_is_a_top_level_parameter(
    provider_key: str,
) -> None:
    adapter = ProviderRequestAdapter(get_provider(provider_key))

    options = adapter.generation_options(
        max_tokens=4_096,
        reasoning_enabled=False,
        reasoning_parameter="reasoning_effort",
    )

    assert options["reasoning_effort"] == "none"


def test_adapter_rejects_unified_reasoning_on_an_incompatible_provider() -> None:
    adapter = ProviderRequestAdapter(get_provider("openai"))

    with pytest.raises(ValueError, match="does not accept"):
        adapter.generation_options(
            max_tokens=4_096,
            reasoning_enabled=True,
            reasoning_parameter="reasoning",
        )


def test_openrouter_structured_options_merge_without_losing_reasoning() -> None:
    adapter = ProviderRequestAdapter(get_provider("openrouter"))

    options = adapter.merge_options(
        adapter.generation_options(
            max_tokens=4_096,
            reasoning_enabled=True,
            reasoning_parameter="reasoning",
        ),
        adapter.structured_options(zero_data_retention=True),
    )

    assert options["extra_body"]["reasoning"]["enabled"] is True
    assert options["extra_body"]["provider"] == {
        "allow_fallbacks": False,
        "data_collection": "deny",
    }
