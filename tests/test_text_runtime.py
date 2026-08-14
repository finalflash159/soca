from __future__ import annotations

import json
from pathlib import Path

import pytest

from soca.app.text_runtime import (
    TextRuntimeConfig,
    build_text_runtime,
    normalize_text_turn,
)
from soca.config.llm_settings import LlmSettings
from soca.knowledge.hybrid_source import HybridKnowledgeSource
from soca.llm import LLMResult
from soca.memory import SessionMemory


class FakeKnowledgeLLM:
    def __init__(self) -> None:
        self.structured_calls: list[str] = []

    def generate(
        self,
        user_msg: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        inject_persona: bool = True,
    ) -> LLMResult:
        return LLMResult(
            text="Protein hỗ trợ duy trì cơ bắp [K1].",
            prompt=user_msg,
            n_prompt_tokens=10,
            n_completion_tokens=6,
            ttft_ms=1.0,
            total_latency_ms=2.0,
            tokens_per_second=100.0,
        )

    def generate_structured(self, user_msg: str, *, schema_name: str, **kwargs) -> LLMResult:
        del kwargs
        self.structured_calls.append(schema_name)
        return LLMResult(
            text=json.dumps(
                {
                    "sufficient": True,
                    "confidence": 0.99,
                    "reason_code": "answer_explicitly_supported",
                }
            ),
            prompt=user_msg,
            n_prompt_tokens=20,
            n_completion_tokens=8,
            ttft_ms=1.0,
            total_latency_ms=2.0,
            tokens_per_second=100.0,
        )


def _config(vault: Path, **overrides) -> TextRuntimeConfig:
    base = {
        "vault": vault,
        "no_llm": True,  # keep tests model-free
        "knowledge_retrieval_mode": "chunk_sparse",
        "tool_router_mode": "deterministic",
        "semantic_router_enabled": False,
    }
    base.update(overrides)
    return TextRuntimeConfig(**base)


def test_text_runtime_default_llm_follows_default_runtime_profile() -> None:
    config = TextRuntimeConfig()

    assert config.profile_key == "baseline"
    assert config.llm_model == "arcee_vylinh_3b_q4_k_m"
    assert config.knowledge_retrieval_mode == "hybrid"
    assert config.knowledge_dense_backend == "aiteamvn_v2"
    assert config.tool_router_response_mode == "json_schema"
    assert config.sufficient_context_enabled is False


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


def test_text_runtime_does_not_register_removed_memory_write_tool(tmp_path: Path) -> None:
    (tmp_path / "wiki").mkdir()
    (tmp_path / "memory").mkdir()
    bundle = build_text_runtime(_config(tmp_path, no_llm=True, no_memory=False))
    assert bundle.runtime.tool_runtime.get("memory.search") is not None


def test_text_runtime_uses_shared_source_and_k_query_returns_citation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "protein.md").write_text(
        "# Chất đạm\n\nProtein hỗ trợ duy trì cơ bắp và cảm giác no.",
        encoding="utf-8",
    )
    bundle = build_text_runtime(
        _config(
            tmp_path,
            no_llm=False,
            no_memory=True,
            knowledge_limit=2,
            sufficient_context_enabled=True,
        ),
        secret_store=object(),
        engine_factory=lambda *args, **kwargs: FakeKnowledgeLLM(),
    )

    source = bundle.runtime.knowledge_builder.source
    search_tool = bundle.runtime.tool_runtime.get("knowledge.search")
    read_tool = bundle.runtime.tool_runtime.get("knowledge.read")
    user_text, metadata = normalize_text_turn("/k chất đạm")
    source.search("warmup", limit=1)
    result = bundle.runtime.run_text_turn(
        user_text,
        source="test",
        metadata=metadata,
    )

    assert bundle.knowledge_status == "enabled:chunk_sparse"
    assert isinstance(source, HybridKnowledgeSource)
    assert search_tool is not None and search_tool.source is source
    assert read_tool is not None and read_tool.source is source
    assert bundle.runtime.options.knowledge_limit == 2
    assert bundle.runtime.options.require_sufficient_context is True
    assert bundle.runtime.sufficiency_assessor is not None
    assert [citation.path for citation in result.citations] == ["wiki/protein.md"]
    assert result.trace is not None


def test_no_evidence_memory_request_without_vault_returns_empty_evidence(tmp_path: Path) -> None:
    bundle = build_text_runtime(_config(tmp_path / "absent", no_memory=False))

    result = bundle.runtime.run_text_turn("memory: something", source="test")

    assert bundle.session_memory is not None
    assert result.blocked is False
    assert result.trace is not None and result.trace.used_llm is False
    assert result.response_text == "Mình chưa tìm thấy ghi chú phù hợp trong memory."
    assert bundle.session_memory.turns == ()


def test_text_runtime_uses_injected_session_memory(tmp_path: Path) -> None:
    shared = SessionMemory()
    bundle = build_text_runtime(
        _config(tmp_path / "absent", no_memory=False),
        session_memory=shared,
    )

    assert bundle.session_memory is shared
    result = bundle.runtime.run_text_turn("memory: something", source="test")
    assert result.blocked is False
    assert result.trace is not None and result.trace.used_llm is False
    assert shared.turns == ()


def test_text_runtime_uses_persisted_remote_selection(monkeypatch, tmp_path: Path) -> None:
    persisted = LlmSettings(
        backend="remote",
        provider_key="groq",
        model_id="llama-3.1-8b-instant",
        max_tokens=8_192,
        model_max_output_tokens=2_048,
    )
    captured: dict[str, object] = {}

    def fake_engine(settings, secrets, **kwargs):
        captured["settings"] = settings
        captured["secrets"] = secrets
        captured["local_factory"] = kwargs["local_factory"]
        return object()

    vault = tmp_path / "vault"
    (vault / "wiki").mkdir(parents=True)
    monkeypatch.setattr("soca.app.text_runtime.load_settings", lambda: persisted)
    bundle = build_text_runtime(
        _config(vault, no_llm=False, no_memory=True, tool_router_mode="cascade"),
        secret_store=object(),
        engine_factory=fake_engine,
    )

    assert captured["settings"] == persisted
    assert bundle.llm_status == "enabled:groq:llama-3.1-8b-instant"
    assert bundle.runtime.options.max_tokens == 2_048
    llm_router = bundle.runtime.tool_router._llm_router
    assert llm_router._config.max_tokens == 2_048


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
