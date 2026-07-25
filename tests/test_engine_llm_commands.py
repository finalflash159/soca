"""Protocol tests for remote LLM configuration commands (no network or keyring)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from soca.app.engine import SocaEngine, _ProtocolWriter, run_engine
from soca.app.text_runtime import TextRuntimeConfig
from soca.config.llm_settings import LlmSettings
from soca.llm.providers import RemoteModelInfo


class ProtocolCapture:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self._lock = threading.Lock()
        self._seen = threading.Condition(self._lock)

    def write(self, text: str) -> None:
        with self._seen:
            self.lines.extend(line for line in text.splitlines() if line.strip())
            self._seen.notify_all()

    def flush(self) -> None:
        return None

    def wait_for(self, substring: str, timeout: float = 2.0) -> None:
        with self._seen:
            found = self._seen.wait_for(
                lambda: any(substring in line for line in self.lines), timeout=timeout
            )
        assert found, self.lines

    def events(self) -> list[dict[str, object]]:
        with self._lock:
            return [json.loads(line) for line in self.lines]


class FakeSecrets:
    def __init__(self, keys: dict[str, str] | None = None) -> None:
        self.keys = dict(keys or {})

    def get_key(self, provider_key: str) -> str | None:
        return self.keys.get(provider_key)

    def set_key(self, provider_key: str, value: str) -> None:
        self.keys[provider_key] = value

    def has_key(self, provider_key: str) -> bool:
        return self.get_key(provider_key) is not None

    @staticmethod
    def mask(value: str) -> str:
        return f"sk-...{value[-4:]}"


def _catalog(provider, api_key: str) -> list[RemoteModelInfo]:
    assert api_key.startswith("sk-")
    return [
        RemoteModelInfo(
            id="llama-3.1-8b-instant",
            label="Llama 3.1 8B Instant",
            context_length=131072,
            price_prompt_per_1m=None,
            price_completion_per_1m=None,
            pricing_source="unknown",
        ),
        RemoteModelInfo(
            id="qwen/qwen3-32b",
            label="Qwen3 32B",
            context_length=32768,
            price_prompt_per_1m=0.12,
            price_completion_per_1m=0.24,
            pricing_source="live",
        ),
    ]


def _run(
    commands: list[dict[str, object]],
    *,
    settings: LlmSettings | None = None,
    secrets: FakeSecrets | None = None,
) -> tuple[ProtocolCapture, list[LlmSettings]]:
    capture = ProtocolCapture()
    saved: list[LlmSettings] = []

    def stdin():
        for command in commands:
            yield json.dumps(command) + "\n"
        yield '{"cmd": "quit"}\n'

    run_engine(
        voice_config=None,
        text_config=TextRuntimeConfig(no_memory=True, vault=Path("/tmp/soca-test-vault")),
        profile="baseline",
        no_model=True,
        stdin=stdin(),
        stdout=capture,
        llm_settings_loader=lambda: settings or LlmSettings(),
        llm_settings_saver=saved.append,
        secret_store=secrets or FakeSecrets({"groq": "sk-existing-1234"}),
        catalog_fetcher=_catalog,
    )
    return capture, saved


def test_llm_providers_reports_all_providers_and_only_key_presence() -> None:
    capture, _ = _run([{"cmd": "llm_providers"}])

    event = next(item for item in capture.events() if item["event"] == "llm_providers")
    providers = event["providers"]
    assert isinstance(providers, list)
    groq = next(item for item in providers if item["key"] == "groq")
    assert groq["has_key"] is True
    assert "sk-existing" not in json.dumps(event)


def test_llm_models_filters_catalog_by_query() -> None:
    capture, _ = _run([{"cmd": "llm_models", "provider": "groq", "query": "qwen"}])

    event = next(item for item in capture.events() if item["event"] == "llm_catalog")
    assert event["provider"] == "groq"
    assert [model["id"] for model in event["models"]] == ["qwen/qwen3-32b"]


def test_llm_set_key_validates_then_masks_without_echoing_secret() -> None:
    secret = "sk-super-secret-9876"
    capture, _ = _run(
        [{"cmd": "llm_set_key", "provider": "openrouter", "key": secret}],
        secrets=FakeSecrets(),
    )

    event = next(item for item in capture.events() if item["event"] == "llm_key_status")
    assert event == {
        "event": "llm_key_status",
        "provider": "openrouter",
        "ok": True,
        "masked": "sk-...9876",
    }
    assert secret not in "\n".join(capture.lines)


def test_llm_select_persists_remote_config_and_emits_active_config() -> None:
    capture, saved = _run(
        [
            {
                "cmd": "llm_select",
                "backend": "remote",
                "provider": "groq",
                "model": "llama-3.1-8b-instant",
            }
        ]
    )

    assert saved == [
        LlmSettings(
            backend="remote",
            provider_key="groq",
            model_id="llama-3.1-8b-instant",
        )
    ]
    event = next(item for item in capture.events() if item["event"] == "llm_config")
    assert event["backend"] == "remote"
    assert event["provider"] == "groq"
    assert event["model"] == "llama-3.1-8b-instant"


def test_llm_select_invalidates_the_lazy_text_runtime() -> None:
    saved: list[LlmSettings] = []
    engine = SocaEngine(
        voice_config=None,
        text_config=TextRuntimeConfig(no_memory=True, vault=Path("/tmp/soca-test-vault")),
        profile="baseline",
        no_model=True,
        writer=_ProtocolWriter(ProtocolCapture()),
        llm_settings_loader=LlmSettings,
        llm_settings_saver=saved.append,
        secret_store=FakeSecrets({"groq": "sk-existing-1234"}),
        catalog_fetcher=_catalog,
    )
    engine.text_bundle = object()  # type: ignore[assignment]

    engine.dispatch(
        {
            "cmd": "llm_select",
            "backend": "remote",
            "provider": "groq",
            "model": "llama-3.1-8b-instant",
        }
    )

    assert engine.text_bundle is None
    assert saved[0].backend == "remote"
