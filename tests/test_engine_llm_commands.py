"""Protocol tests for remote LLM configuration commands (no network or keyring)."""

from __future__ import annotations

import io
import json
import threading
from collections.abc import Callable
from pathlib import Path

from soca.app.engine import SocaEngine, _ProtocolWriter, run_engine
from soca.app.text_runtime import TextRuntimeConfig
from soca.config.llm_settings import LlmSettings
from soca.llm.providers import LLMProvider, RemoteModelInfo

CatalogFetcher = Callable[[LLMProvider, str], list[RemoteModelInfo]]


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
    catalog_fetcher: CatalogFetcher = _catalog,
    no_model: bool = True,
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
        no_model=no_model,
        stdin=stdin(),
        stdout=capture,
        llm_settings_loader=lambda: settings or LlmSettings(),
        llm_settings_saver=saved.append,
        secret_store=secrets or FakeSecrets({"groq": "sk-existing-1234"}),
        catalog_fetcher=catalog_fetcher,
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

    event = next(item for item in reversed(capture.events()) if item["event"] == "llm_catalog")
    assert event["provider"] == "groq"
    assert event["loading"] is False
    assert [model["id"] for model in event["models"]] == ["qwen/qwen3-32b"]
    assert "supported_parameters" not in event["models"][0]


def test_llm_models_converts_an_unexpected_catalog_failure_to_a_safe_error() -> None:
    def broken_catalog(provider: LLMProvider, api_key: str) -> list[RemoteModelInfo]:
        raise RuntimeError("internal catalog diagnostic")

    capture, _ = _run([{"cmd": "llm_models", "provider": "groq"}], catalog_fetcher=broken_catalog)

    event = next(item for item in capture.events() if item["event"] == "engine_error")
    assert event["provider"] == "groq"
    assert "không thể lấy danh sách model" in event["message"].lower()
    assert "internal catalog diagnostic" not in event["message"]


def test_llm_set_key_validates_then_masks_without_echoing_secret() -> None:
    secret = "sk-super-secret-9876"
    capture, _ = _run(
        [{"cmd": "llm_set_key", "provider": "openrouter", "key": secret}],
        secrets=FakeSecrets(),
    )

    event = next(
        item
        for item in reversed(capture.events())
        if item["event"] == "llm_key_status" and not item.get("pending")
    )
    assert event == {
        "event": "llm_key_status",
        "provider": "openrouter",
        "ok": True,
        "masked": "sk-...9876",
    }
    assert secret not in "\n".join(capture.lines)


def test_llm_set_key_converts_an_unexpected_catalog_failure_to_a_safe_status() -> None:
    def broken_catalog(provider: LLMProvider, api_key: str) -> list[RemoteModelInfo]:
        raise RuntimeError("internal catalog diagnostic")

    capture, _ = _run(
        [{"cmd": "llm_set_key", "provider": "groq", "key": "sk-test-key"}],
        secrets=FakeSecrets(),
        catalog_fetcher=broken_catalog,
    )

    event = next(
        item
        for item in reversed(capture.events())
        if item["event"] == "llm_key_status" and not item.get("pending")
    )
    assert event["ok"] is False
    assert "không thể xác thực api key" in event["message"].lower()
    assert "internal catalog diagnostic" not in event["message"]


def test_llm_select_survives_an_unexpected_active_model_lookup_failure() -> None:
    def broken_catalog(provider: LLMProvider, api_key: str) -> list[RemoteModelInfo]:
        raise RuntimeError("internal catalog diagnostic")

    capture, saved = _run(
        [
            {
                "cmd": "llm_select",
                "backend": "remote",
                "provider": "groq",
                "model": "llama-3.1-8b-instant",
            }
        ],
        catalog_fetcher=broken_catalog,
    )

    assert saved[0].backend == "remote"
    event = [item for item in capture.events() if item["event"] == "llm_config"][-1]
    assert event["pricing"] is None


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

    assert saved[0] == LlmSettings(
        backend="remote",
        provider_key="groq",
        model_id="llama-3.1-8b-instant",
    )
    assert saved[-1].model_context_window == 131_072
    event = [item for item in capture.events() if item["event"] == "llm_config"][-1]
    assert event["backend"] == "remote"
    assert event["provider"] == "groq"
    assert event["model"] == "llama-3.1-8b-instant"
    assert event["runtime_ready"] is True
    assert event["runtime_reason"] is None
    assert event["local_model_path"] is None


def test_remote_readiness_is_independent_of_local_model_presence() -> None:
    capture, _ = _run(
        [
            {"cmd": "llm_models", "provider": "groq"},
            {
                "cmd": "llm_select",
                "backend": "remote",
                "provider": "groq",
                "model": "llama-3.1-8b-instant",
            },
        ],
        no_model=False,
    )

    event = [item for item in capture.events() if item["event"] == "llm_config"][-1]
    assert event["backend"] == "remote"
    assert event["runtime_ready"] is True
    assert event["local_model_path"] is None


def test_switching_from_remote_to_local_does_not_persist_remote_model_id() -> None:
    capture, saved = _run(
        [
            {
                "cmd": "llm_select",
                "backend": "remote",
                "provider": "groq",
                "model": "llama-3.1-8b-instant",
            },
            {"cmd": "llm_select", "backend": "local", "provider": "groq"},
        ]
    )

    assert saved[-1].backend == "local"
    assert saved[-1].model_id == LlmSettings().model_id
    config_events = [item for item in capture.events() if item["event"] == "llm_config"]
    assert config_events[-1]["backend"] == "local"
    assert config_events[-1]["model"] == LlmSettings().model_id


def test_select_rejects_model_not_in_a_loaded_remote_catalog() -> None:
    capture, saved = _run(
        [
            {"cmd": "llm_models", "provider": "groq"},
            {
                "cmd": "llm_select",
                "backend": "remote",
                "provider": "groq",
                "model": "not-in-catalog",
            },
        ]
    )

    assert saved == []
    error = next(item for item in capture.events() if item["event"] == "engine_error")
    assert "không có trong danh mục" in error["message"].lower()


def test_key_validation_emits_catalog_and_refreshes_remote_readiness() -> None:
    capture, _ = _run(
        [{"cmd": "llm_set_key", "provider": "openrouter", "key": "sk-new-1234"}],
        settings=LlmSettings(
            backend="remote",
            provider_key="openrouter",
            model_id="qwen/qwen3-32b",
        ),
        secrets=FakeSecrets(),
    )

    events = capture.events()
    assert any(
        item["event"] == "llm_key_status" and item.get("ok") is True for item in events
    )
    catalog = [item for item in events if item["event"] == "llm_catalog"][-1]
    assert catalog["provider"] == "openrouter"
    assert catalog["loading"] is False
    config = [item for item in events if item["event"] == "llm_config"][-1]
    assert config["runtime_ready"] is True


def test_remote_config_emits_pending_before_a_fast_catalog_can_emit_ready() -> None:
    started = threading.Event()
    release = threading.Event()
    capture = ProtocolCapture()
    saved: list[LlmSettings] = []

    def catalog(provider: LLMProvider, api_key: str) -> list[RemoteModelInfo]:
        started.set()
        assert release.wait(timeout=2.0)
        return _catalog(provider, api_key)

    engine = SocaEngine(
        voice_config=None,
        text_config=TextRuntimeConfig(no_memory=True, vault=Path("/tmp/soca-test-vault")),
        profile="baseline",
        no_model=True,
        writer=_ProtocolWriter(capture),
        llm_settings_loader=LlmSettings,
        llm_settings_saver=saved.append,
        secret_store=FakeSecrets({"groq": "sk-existing-1234"}),
        catalog_fetcher=catalog,
    )
    try:
        engine.dispatch(
            {
                "cmd": "llm_select",
                "backend": "remote",
                "provider": "groq",
                "model": "llama-3.1-8b-instant",
            }
        )
        assert started.wait(timeout=2.0)
        initial = [item for item in capture.events() if item["event"] == "llm_config"]
        assert initial
        assert initial[-1]["runtime_ready"] is False
    finally:
        release.set()
        engine.shutdown()

    final = [item for item in capture.events() if item["event"] == "llm_config"]
    assert final[-1]["runtime_ready"] is True


def test_llm_config_reports_missing_local_model_without_touching_remote_key_state(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr("soca.llm.registry.MODELS_ROOT", tmp_path / "empty-models")
    capture, _ = _run(
        [{"cmd": "llm_config"}], settings=LlmSettings(backend="local"), no_model=False
    )

    event = next(item for item in capture.events() if item["event"] == "llm_config")
    assert event["runtime_ready"] is False
    assert event["runtime_reason"].startswith("Chưa tìm thấy model local tại ")
    assert event["local_model_path"].endswith("Arcee-VyLinh.Q4_K_M.gguf")


def test_llm_select_invalidates_previous_prompt_manifest() -> None:
    output = io.StringIO()
    saved: list[LlmSettings] = []
    engine = SocaEngine(
        voice_config=None,
        text_config=TextRuntimeConfig(no_memory=True, vault=Path("/tmp/soca-test-vault")),
        profile="baseline",
        no_model=True,
        writer=_ProtocolWriter(output),
        llm_settings_loader=lambda: LlmSettings(),
        llm_settings_saver=saved.append,
        secret_store=FakeSecrets({"groq": "sk-existing-1234"}),
        catalog_fetcher=_catalog,
    )
    engine._last_prompt_manifest = {
        "model_id": "old-model",
        "context_window": 2_048,
        "effective_output_tokens": 256,
    }

    try:
        engine.dispatch(
            {
                "cmd": "llm_select",
                "backend": "remote",
                "provider": "groq",
                "model": "llama-3.1-8b-instant",
            }
        )
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        context = [event for event in events if event["event"] == "context"][-1]
        assert context["estimated"] is True
        assert context["prompt_manifest"]["model_id"] == "llama-3.1-8b-instant"
        assert context["prompt_hash"]
    finally:
        engine.shutdown()


def test_llm_select_persists_requested_generation_and_caps_effective_output() -> None:
    model = RemoteModelInfo(
        id="reasoning/model",
        label="Reasoning Model",
        context_length=200_000,
        price_prompt_per_1m=None,
        price_completion_per_1m=None,
        pricing_source="unknown",
        supported_parameters=("reasoning",),
        max_output_tokens=32_768,
        reasoning_supported=True,
        reasoning_mandatory=True,
        reasoning_parameter="reasoning",
    )

    def catalog(provider: LLMProvider, api_key: str) -> list[RemoteModelInfo]:
        return [model]

    capture, saved = _run(
        [
            {"cmd": "llm_models", "provider": "groq"},
            {
                "cmd": "llm_select",
                "backend": "remote",
                "provider": "groq",
                "model": "reasoning/model",
                "max_tokens": 500_000,
                "reasoning_enabled": False,
            },
        ],
        catalog_fetcher=catalog,
    )

    selected = saved[-1]
    assert selected.max_tokens == 500_000
    assert selected.effective_max_tokens == 32_768
    assert selected.reasoning_enabled is False
    assert selected.effective_reasoning_enabled is True
    event = [item for item in capture.events() if item["event"] == "llm_config"][-1]
    assert event["effective_max_tokens"] == 32_768
    assert event["effective_reasoning_enabled"] is True


def test_llm_select_invalidates_the_lazy_text_runtime() -> None:
    class Bundle:
        def close(self) -> None:
            return None

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
    engine.text_bundle = Bundle()  # type: ignore[assignment]

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


def test_llm_select_invalidates_idle_voice_runtime() -> None:
    class IdleController:
        def __init__(self) -> None:
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

    engine = SocaEngine(
        voice_config=None,
        text_config=TextRuntimeConfig(no_memory=True, vault=Path("/tmp/soca-test-vault")),
        profile="baseline",
        no_model=True,
        writer=_ProtocolWriter(io.StringIO()),
        llm_settings_loader=LlmSettings,
        llm_settings_saver=lambda _settings: None,
        secret_store=FakeSecrets({"groq": "sk-existing-1234"}),
        catalog_fetcher=_catalog,
    )
    controller = IdleController()
    engine.voice_controller = controller  # type: ignore[assignment]

    engine.dispatch(
        {
            "cmd": "llm_select",
            "backend": "remote",
            "provider": "groq",
            "model": "llama-3.1-8b-instant",
        }
    )

    assert controller.stopped is True
    assert engine.voice_controller is None


def test_llm_select_rejects_hot_swap_while_voice_is_active() -> None:
    class AliveThread:
        def is_alive(self) -> bool:
            return True

    saved: list[LlmSettings] = []
    engine = SocaEngine(
        voice_config=None,
        text_config=TextRuntimeConfig(no_memory=True, vault=Path("/tmp/soca-test-vault")),
        profile="baseline",
        no_model=True,
        writer=_ProtocolWriter(io.StringIO()),
        llm_settings_loader=LlmSettings,
        llm_settings_saver=saved.append,
        secret_store=FakeSecrets({"groq": "sk-existing-1234"}),
        catalog_fetcher=_catalog,
    )
    engine.voice_stop_event = threading.Event()
    engine._voice_threads = [AliveThread()]  # type: ignore[list-item]

    engine.dispatch(
        {
            "cmd": "llm_select",
            "backend": "remote",
            "provider": "groq",
            "model": "llama-3.1-8b-instant",
        }
    )

    assert saved == []
    assert engine.llm_settings.backend == "local"
