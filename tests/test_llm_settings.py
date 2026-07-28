"""Tests for LlmSettings (immutable config + JSON round-trip)."""

from __future__ import annotations

import json
import stat
from dataclasses import FrozenInstanceError

import pytest

from soca.config.llm_settings import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_SETTINGS,
    MAX_MAX_TOKENS,
    MIN_MAX_TOKENS,
    LlmSettings,
    load_settings,
    save_settings,
)

# -- construction + validation ----------------------------------------------


def test_defaults_are_local_backend() -> None:
    assert DEFAULT_SETTINGS.backend == "local"
    # A fresh install must never silently point at a paid remote provider.
    assert isinstance(DEFAULT_SETTINGS, LlmSettings)
    assert DEFAULT_SETTINGS.max_tokens == DEFAULT_MAX_TOKENS == 4_096


def test_load_migrates_the_old_implicit_token_default(tmp_path) -> None:
    path = tmp_path / "llm.json"
    path.write_text(
        json.dumps(
            {
                "backend": "remote",
                "model_id": "z-ai/glm-5.2",
                "provider_key": "openrouter",
                "max_tokens": 160,
                "temperature": 0.2,
                "top_p": 0.95,
            }
        ),
        encoding="utf-8",
    )

    assert load_settings(path).max_tokens == DEFAULT_MAX_TOKENS


@pytest.mark.parametrize("legacy_max_tokens", [1, 200, 512, 1_024, 2_047])
def test_load_migrates_positive_legacy_token_values_without_losing_selection(
    tmp_path, legacy_max_tokens: int
) -> None:
    path = tmp_path / "llm.json"
    path.write_text(
        json.dumps(
            {
                "backend": "remote",
                "model_id": "z-ai/glm-5.2",
                "provider_key": "openrouter",
                "max_tokens": legacy_max_tokens,
                "temperature": 0.2,
                "top_p": 0.95,
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(path)

    assert settings.max_tokens == MIN_MAX_TOKENS
    assert settings.backend == "remote"
    assert settings.provider_key == "openrouter"
    assert settings.model_id == "z-ai/glm-5.2"


def test_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="backend"):
        LlmSettings(backend="cloud")


def test_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="provider"):
        LlmSettings(backend="remote", provider_key="not-a-provider", model_id="x")


def test_remote_requires_a_model_id() -> None:
    with pytest.raises(ValueError, match="model"):
        LlmSettings(backend="remote", provider_key="groq", model_id="")


def test_rejects_out_of_range_generation_params() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        LlmSettings(max_tokens=0)
    with pytest.raises(ValueError, match="max_tokens"):
        LlmSettings(max_tokens=MIN_MAX_TOKENS - 1)
    with pytest.raises(ValueError, match="max_tokens"):
        LlmSettings(max_tokens=MAX_MAX_TOKENS + 1)
    with pytest.raises(ValueError, match="temperature"):
        LlmSettings(temperature=-0.1)
    with pytest.raises(ValueError, match="top_p"):
        LlmSettings(top_p=0)
    with pytest.raises(ValueError, match="top_p"):
        LlmSettings(top_p=1.5)
    with pytest.raises(ValueError, match="reasoning_supported"):
        LlmSettings(model_reasoning_supported=1)  # type: ignore[arg-type]


# -- immutability / with_* copies -------------------------------------------


def test_with_methods_return_new_copies() -> None:
    base = LlmSettings()
    remote = base.with_backend("remote").with_provider("groq").with_model("llama-3.1-8b-instant")

    assert base.backend == "local"  # original untouched
    assert remote.backend == "remote"
    assert remote.provider_key == "groq"
    assert remote.model_id == "llama-3.1-8b-instant"
    assert remote is not base


def test_with_generation_returns_new_copy() -> None:
    base = LlmSettings()
    tuned = base.with_generation(
        max_tokens=8192,
        reasoning_enabled=True,
        temperature=0.5,
        top_p=0.9,
    )

    assert base.max_tokens != 8192
    assert tuned.max_tokens == 8192
    assert tuned.reasoning_enabled is True
    assert tuned.temperature == 0.5
    assert tuned.top_p == 0.9


def test_settings_are_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        LlmSettings().backend = "remote"  # type: ignore[misc]


# -- persistence ------------------------------------------------------------


def test_load_missing_file_returns_defaults(tmp_path) -> None:
    settings = load_settings(tmp_path / "llm.json")
    assert settings == DEFAULT_SETTINGS


def test_save_then_load_round_trips(tmp_path) -> None:
    path = tmp_path / "sub" / "llm.json"  # parent created on save
    original = LlmSettings(
        backend="remote",
        provider_key="openrouter",
        model_id="openai/gpt-4o-mini",
        max_tokens=8192,
        reasoning_enabled=True,
        temperature=0.3,
        top_p=0.8,
    )

    save_settings(original, path)
    loaded = load_settings(path)

    assert loaded == original


def test_effective_generation_respects_model_capabilities() -> None:
    requested = LlmSettings(
        backend="remote",
        model_id="reasoning/model",
        max_tokens=500_000,
        reasoning_enabled=False,
        model_max_output_tokens=65_536,
        model_reasoning_supported=True,
        model_reasoning_mandatory=True,
        model_reasoning_parameter="reasoning",
    )

    assert requested.max_tokens == 500_000
    assert requested.effective_max_tokens == 65_536
    assert requested.effective_reasoning_enabled is True


def test_unknown_reasoning_capability_uses_model_default() -> None:
    settings = LlmSettings(reasoning_enabled=True)

    assert settings.effective_reasoning_enabled is None


def test_saved_file_is_owner_only_readable(tmp_path) -> None:
    path = tmp_path / "llm.json"
    save_settings(LlmSettings(), path)

    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600


def test_saved_file_never_contains_a_key(tmp_path) -> None:
    path = tmp_path / "llm.json"
    save_settings(LlmSettings(backend="remote", provider_key="groq", model_id="m"), path)

    raw = path.read_text(encoding="utf-8")
    assert "api_key" not in raw.lower()
    assert "sk-" not in raw


def test_load_invalid_json_fails_fast(tmp_path) -> None:
    path = tmp_path / "llm.json"
    path.write_text("{ not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="llm.json"):
        load_settings(path)


def test_load_rejects_bad_schema(tmp_path) -> None:
    path = tmp_path / "llm.json"
    path.write_text(json.dumps({"backend": "nope"}), encoding="utf-8")

    with pytest.raises(ValueError):
        load_settings(path)
