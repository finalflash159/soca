from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from soca.config import LlmSettings
from soca.core import (
    RuntimeOptions,
    build_voice_runtime,
    resolve_voice_runtime_config,
    voice_runtime,
)
from soca.core.knowledge_setup import (
    KnowledgeRuntimeSetup,
    build_knowledge_runtime_setup,
)
from soca.knowledge.factory import RetrievalConfig


def test_resolver_does_not_accept_tts_model_override() -> None:
    parameters = inspect.signature(resolve_voice_runtime_config).parameters
    assert "tts_model" not in parameters


def test_baseline_resolves_former_quality_stack_with_valtec(tmp_path: Path) -> None:
    config = resolve_voice_runtime_config(profile_key="baseline", vault=tmp_path)

    assert config.asr_model == "phowhisper_small"
    assert config.llm_model == "arcee_vylinh_3b_q4_k_m"
    assert not hasattr(config, "tts_model")
    assert config.tts_voice == "NF"


def test_voice_defaults_to_the_shared_semantic_policy(tmp_path: Path) -> None:
    config = resolve_voice_runtime_config(profile_key="baseline", vault=tmp_path)
    assert config.tool_router_mode == "cascade"
    assert config.semantic_router_enabled is True
    assert config.semantic_router_threshold == 0.58
    assert config.semantic_router_margin == 0.0
    assert config.semantic_router_examples is not None
    assert config.semantic_router_examples.name == "turn_routing_vi.jsonl"
    assert config.semantic_router_examples.parent.name == "p0"


@pytest.mark.parametrize("profile_key", ["quality", "edge"])
def test_removed_runtime_profiles_are_rejected(
    profile_key: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Unknown voice runtime profile"):
        resolve_voice_runtime_config(profile_key=profile_key, vault=tmp_path)


@pytest.mark.parametrize("voice", ["NF", "SF", "NM1", "SM", "NM2"])
def test_explicit_valtec_voice_override_is_allowed(
    voice: str,
    tmp_path: Path,
) -> None:
    config = resolve_voice_runtime_config(
        profile_key="baseline",
        tts_voice=voice,
        vault=tmp_path,
    )

    assert config.tts_voice == voice


def test_first_clause_defaults_to_profile_and_can_be_overridden(tmp_path: Path) -> None:
    default = resolve_voice_runtime_config(profile_key="baseline", vault=tmp_path)
    assert default.first_clause_enabled is True  # baseline profile default

    off = resolve_voice_runtime_config(
        profile_key="baseline",
        first_clause_enabled=False,
        vault=tmp_path,
    )
    assert off.first_clause_enabled is False
    # Override only touches the toggle, not the other clause knobs.
    assert off.first_clause_min_chars == default.first_clause_min_chars


def test_unknown_valtec_voice_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown Valtec voice"):
        resolve_voice_runtime_config(
            profile_key="baseline",
            tts_voice="not-a-valtec-voice",
            vault=tmp_path,
        )


def test_voice_runtime_uses_shared_knowledge_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "wiki").mkdir()
    shared_setup = build_knowledge_runtime_setup(
        tmp_path,
        knowledge_limit=RuntimeOptions().knowledge_limit,
        retrieval_config=RetrievalConfig(
            mode="chunk_sparse",
            watcher_enabled=False,
        ),
    )
    setup_calls: list[tuple[Path, int]] = []

    def fake_knowledge_setup(
        vault: Path,
        *,
        knowledge_limit: int,
        retrieval_config=None,
    ) -> KnowledgeRuntimeSetup:
        setup_calls.append((vault, knowledge_limit))
        return shared_setup

    monkeypatch.setattr(
        voice_runtime,
        "build_knowledge_runtime_setup",
        fake_knowledge_setup,
        raising=False,
    )
    monkeypatch.setattr(voice_runtime, "SpeechDetector", lambda: object())
    monkeypatch.setattr(voice_runtime, "VietnameseASR", lambda **kwargs: object())
    monkeypatch.setattr(voice_runtime, "RobustASR", lambda **kwargs: object())
    monkeypatch.setattr(
        voice_runtime,
        "load_confidence_guard_calibration",
        lambda _model_key: None,
    )
    def fake_engine_factory(settings, secrets, **kwargs):
        del settings, secrets, kwargs
        return object()
    monkeypatch.setattr(voice_runtime, "create_tts_engine", lambda **kwargs: object())
    monkeypatch.setattr(voice_runtime, "VoicePipeline", lambda **kwargs: object())
    monkeypatch.setattr(voice_runtime, "default_repair_catalog", lambda: object())
    config = resolve_voice_runtime_config(
        profile_key="baseline",
        vault=tmp_path,
        adaptive_endpoint=False,
        no_memory=True,
    )

    bundle = build_voice_runtime(config, engine_factory=fake_engine_factory)

    assert setup_calls == [
        (config.vault, RuntimeOptions().knowledge_limit),
    ]
    assert bundle.knowledge_status == "enabled:chunk_sparse"
    assert bundle.assistant_runtime.knowledge_builder is shared_setup.builder
    assert bundle.assistant_runtime.tool_runtime.get("knowledge.search") is shared_setup.search_tool
    assert bundle.assistant_runtime.tool_runtime.get("knowledge.read") is shared_setup.read_tool


def test_voice_runtime_uses_selected_remote_llm_without_local_construction(
    tmp_path: Path,
) -> None:
    config = resolve_voice_runtime_config(
        profile_key="baseline",
        vault=tmp_path,
        adaptive_endpoint=False,
        no_memory=True,
    )
    settings = LlmSettings(
        backend="remote",
        provider_key="openrouter",
        model_id="google/gemini-3.5-flash-lite",
    )
    calls: list[tuple[LlmSettings, object]] = []

    def fake_engine_factory(selected, secrets, **kwargs):
        calls.append((selected, secrets))
        del kwargs
        return object()

    bundle = build_voice_runtime(
        config,
        llm_settings=settings,
        secret_store=object(),  # type: ignore[arg-type]
        engine_factory=fake_engine_factory,
    )

    assert len(calls) == 1
    assert calls[0][0] is settings
    assert bundle.llm_settings == settings
