from __future__ import annotations

from pathlib import Path

from soca.app.text_runtime import (
    TextRuntimeConfig,
    build_text_runtime,
    normalize_text_turn,
)
from soca.config.llm_settings import LlmSettings
from soca.memory import SessionMemory


def _config(vault: Path, **overrides) -> TextRuntimeConfig:
    base = {
        "vault": vault,
        "no_llm": True,  # keep tests model-free
    }
    base.update(overrides)
    return TextRuntimeConfig(**base)


def test_text_runtime_default_llm_follows_default_runtime_profile() -> None:
    config = TextRuntimeConfig()

    assert config.profile_key == "baseline"
    assert config.llm_model == "arcee_vylinh_3b_q4_k_m"


def test_session_memory_enabled_without_vault(tmp_path: Path) -> None:
    missing_vault = tmp_path / "no-vault"
    bundle = build_text_runtime(_config(missing_vault, no_memory=False))

    # RAM session memory must exist even though the vault is missing.
    assert bundle.session_memory is not None
    assert bundle.memory_status.startswith("session-only")
    assert "vault_missing" in bundle.memory_status


def test_no_memory_disables_session_even_with_vault(tmp_path: Path) -> None:
    (tmp_path / "wiki").mkdir()
    bundle = build_text_runtime(_config(tmp_path, no_memory=True))

    assert bundle.session_memory is None
    assert bundle.memory_status == "disabled"


def test_session_memory_records_turn_without_vault(tmp_path: Path) -> None:
    bundle = build_text_runtime(_config(tmp_path / "absent", no_memory=False))

    # A deterministic tool turn (time question) still appends to session memory.
    bundle.runtime.run_text_turn("mấy giờ rồi?", source="test")

    assert bundle.session_memory is not None
    assert [turn.role for turn in bundle.session_memory.turns] == ["user", "assistant"]


def test_text_runtime_uses_injected_session_memory(tmp_path: Path) -> None:
    shared = SessionMemory()
    bundle = build_text_runtime(
        _config(tmp_path / "absent", no_memory=False),
        session_memory=shared,
    )

    assert bundle.session_memory is shared
    bundle.runtime.run_text_turn("mấy giờ rồi?", source="test")
    assert len(shared.turns) == 2


def test_text_runtime_uses_persisted_remote_selection(monkeypatch, tmp_path: Path) -> None:
    persisted = LlmSettings(
        backend="remote",
        provider_key="groq",
        model_id="llama-3.1-8b-instant",
    )
    captured: dict[str, object] = {}

    def fake_engine(settings, secrets, **kwargs):
        captured["settings"] = settings
        captured["secrets"] = secrets
        captured["local_factory"] = kwargs["local_factory"]
        return object()

    monkeypatch.setattr("soca.app.text_runtime.load_settings", lambda: persisted)
    bundle = build_text_runtime(
        _config(tmp_path / "absent", no_llm=False),
        secret_store=object(),
        engine_factory=fake_engine,
    )

    assert captured["settings"] == persisted
    assert bundle.llm_status == "enabled:groq:llama-3.1-8b-instant"


def test_text_runtime_uses_persisted_local_model_without_cli_override(
    monkeypatch, tmp_path: Path
) -> None:
    persisted = LlmSettings(model_id="qwen3_0_6b_q8_0")
    captured: dict[str, object] = {}

    def fake_engine(settings, secrets, **kwargs):
        captured["settings"] = settings
        return object()

    monkeypatch.setattr("soca.app.text_runtime.load_settings", lambda: persisted)
    bundle = build_text_runtime(
        _config(tmp_path / "absent", no_llm=False),
        secret_store=object(),
        engine_factory=fake_engine,
    )

    assert captured["settings"] == persisted
    assert bundle.llm_status == "enabled:qwen3_0_6b_q8_0"


def test_normalize_text_turn_extracts_knowledge_prefix() -> None:
    assert normalize_text_turn("xin chào") == ("xin chào", {})
    assert normalize_text_turn("  /k chất đạm là gì?  ") == (
        "chất đạm là gì?",
        {"use_knowledge": True},
    )
