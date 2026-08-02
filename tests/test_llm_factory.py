"""Tests for build_llm_engine (routes local vs remote from settings)."""

from __future__ import annotations

import pytest

from soca.config.llm_settings import LlmSettings
from soca.config.secret_store import SecretStore
from soca.llm.factory import build_llm_engine
from soca.llm.providers import RemoteLLMError


class FakeSecrets:
    def __init__(self, keys: dict[str, str]) -> None:
        self._keys = keys

    def get_key(self, provider_key: str) -> str | None:
        return self._keys.get(provider_key)


def test_local_backend_calls_local_factory_with_model() -> None:
    calls: list[dict] = []

    def fake_local(*, model_key: str, n_threads: int, n_gpu_layers: int):
        calls.append({"model_key": model_key, "n_threads": n_threads, "n_gpu_layers": n_gpu_layers})
        return object()

    settings = LlmSettings(backend="local", model_id="phogpt-4b")
    engine = build_llm_engine(
        settings, FakeSecrets({}), local_factory=fake_local, n_threads=4, n_gpu_layers=0
    )

    assert engine is not None
    assert calls == [{"model_key": "phogpt-4b", "n_threads": 4, "n_gpu_layers": 0}]


def test_remote_backend_builds_engine_with_resolved_key() -> None:
    captured: dict = {}

    def fake_remote(provider, model, api_key, **generation):
        captured["provider_key"] = provider.key
        captured["model"] = model
        captured["api_key"] = api_key
        captured["generation"] = generation
        return object()

    settings = LlmSettings(backend="remote", provider_key="groq", model_id="llama-3.1-8b-instant")
    secrets = FakeSecrets({"groq": "sk-live-key"})

    engine = build_llm_engine(settings, secrets, remote_factory=fake_remote)

    assert engine is not None
    assert captured == {
        "provider_key": "groq",
        "model": "llama-3.1-8b-instant",
        "api_key": "sk-live-key",
        "generation": {
            "reasoning_enabled": None,
            "reasoning_parameter": None,
            "max_output_tokens": 4_096,
        },
    }


def test_remote_factory_receives_reconciled_model_limits_and_reasoning() -> None:
    captured: dict = {}

    def fake_remote(provider, model, api_key, **generation):
        del provider, model, api_key
        captured.update(generation)
        return object()

    settings = LlmSettings(
        backend="remote",
        provider_key="openrouter",
        model_id="reasoning/model",
        max_tokens=500_000,
        reasoning_enabled=False,
        model_max_output_tokens=16_384,
        model_reasoning_supported=True,
        model_reasoning_mandatory=True,
        model_reasoning_parameter="reasoning",
    )

    build_llm_engine(
        settings,
        FakeSecrets({"openrouter": "sk-live-key"}),
        remote_factory=fake_remote,
    )

    assert captured == {
        "reasoning_enabled": True,
        "reasoning_parameter": "reasoning",
        "max_output_tokens": 16_384,
    }


def test_remote_factory_programming_type_error_is_not_hidden() -> None:
    def broken_remote(provider, model, api_key, **generation):
        del provider, model, api_key, generation
        raise TypeError("factory implementation bug")

    settings = LlmSettings(backend="remote", provider_key="groq", model_id="model")

    with pytest.raises(TypeError, match="factory implementation bug"):
        build_llm_engine(
            settings,
            FakeSecrets({"groq": "sk-live-key"}),
            remote_factory=broken_remote,
        )


def test_remote_without_key_raises_auth_error() -> None:
    settings = LlmSettings(backend="remote", provider_key="openai", model_id="gpt-4o-mini")
    secrets = FakeSecrets({})  # no key for openai

    with pytest.raises(RemoteLLMError) as exc:
        build_llm_engine(settings, secrets, remote_factory=lambda *a: object())

    assert exc.value.category == "auth"


def test_real_secret_store_type_is_accepted() -> None:
    # build_llm_engine should accept a real SecretStore (structural: get_key).
    settings = LlmSettings(backend="local", model_id="phogpt-4b")
    engine = build_llm_engine(
        settings,
        SecretStore(keyring_module=None, env={}, dotenv_path=None),
        local_factory=lambda **kw: object(),
    )
    assert engine is not None
