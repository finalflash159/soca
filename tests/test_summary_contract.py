from __future__ import annotations

import json

import pytest

from soca.llm import LLMResult
from soca.memory.summary import (
    PRODUCTION_SUMMARY_MODEL_KEY,
    PRODUCTION_SUMMARY_RELEASE_GATE,
    SUMMARY_MODEL_REGISTRY,
    LocalSummaryWorkerProcess,
    artifact_from_json,
    build_production_summary_worker,
    build_summary_prompt,
    default_summary_model_root,
    execute_summary_job,
    summary_context_window,
)
from soca.memory.working import WorkingMemory


class _StructuredEngine:
    def generate_structured(self, prompt: str, **kwargs: object) -> LLMResult:
        assert "PREVIOUS_SUMMARY_JSON:" in prompt
        assert "FROZEN_TURNS_JSON:" in prompt
        assert kwargs["max_tokens"] == 2_048
        return LLMResult(
            text=json.dumps(
                {
                    "summary": "Người dùng muốn giữ quyết định TTS.",
                    "user_constraints": ["Trả lời bằng tiếng Việt."],
                    "decisions": ["Giữ TTS local."],
                    "corrections": [],
                    "open_items": ["Đo benchmark."],
                    "continuity_refs": [],
                }
            ),
            prompt=prompt,
            n_prompt_tokens=10,
            n_completion_tokens=10,
            ttft_ms=1.0,
            total_latency_ms=2.0,
            tokens_per_second=5.0,
        )


class _RepairStructuredEngine:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    def generate_structured(self, prompt: str, **kwargs: object) -> LLMResult:
        self.calls += 1
        self.prompts.append(prompt)
        assert kwargs["max_tokens"] == 2_048
        summary = "từ " * 2_100 if self.calls == 1 else "Bối cảnh đã được compact."
        text = json.dumps(
            {
                "summary": summary,
                "user_constraints": [],
                "decisions": [],
                "corrections": [],
                "open_items": [],
                "continuity_refs": [],
            }
        )
        return LLMResult(
            text=text,
            prompt=prompt,
            n_prompt_tokens=10,
            n_completion_tokens=10,
            ttft_ms=1.0,
            total_latency_ms=2.0,
            tokens_per_second=5.0,
        )


def test_summary_registry_is_separate_and_summary_job_uses_structured_local_contract() -> None:
    assert "qwen3_0_6b_q8_0" in SUMMARY_MODEL_REGISTRY
    assert "qwen3_1_7b_q8_0" in SUMMARY_MODEL_REGISTRY
    assert "qwen25_3b_instruct_q4_k_m" in SUMMARY_MODEL_REGISTRY
    assert "qwen3_4b_q4_k_m" in SUMMARY_MODEL_REGISTRY
    assert "qwen3_4b_instruct_2507_q4_k_m" in SUMMARY_MODEL_REGISTRY
    assert "qwen3_8b_q4_k_m" not in SUMMARY_MODEL_REGISTRY
    assert "qwen3_14b_q4_k_m" not in SUMMARY_MODEL_REGISTRY
    assert "arcee_vylinh_3b_q4_k_m" not in SUMMARY_MODEL_REGISTRY
    assert "gemma3_4b_it_qat_q4_0" not in SUMMARY_MODEL_REGISTRY
    assert "sailor2_1b_chat_q4" not in SUMMARY_MODEL_REGISTRY
    assert "sailor2_8b_chat_q4_k_m" not in SUMMARY_MODEL_REGISTRY
    memory = WorkingMemory(token_counter=lambda _: 15_000)
    for index in range(6):
        turn = memory.begin_turn(f"user {index}")
        memory.finish_turn(turn.sequence, f"assistant {index}")
    job = memory.prepare_compaction()
    assert job is not None
    artifact, result = execute_summary_job(job, _StructuredEngine())
    assert artifact.generation == job.generation
    assert artifact.source_through_sequence == 4
    assert result.text


def test_summary_prompt_requires_continuity_even_without_durable_state() -> None:
    memory = WorkingMemory()
    for index in range(5):
        turn = memory.begin_turn(f"Câu hỏi kỹ thuật {index}")
        memory.finish_turn(turn.sequence, f"Giải thích kỹ thuật {index}")
    job = memory.prepare_compaction(force=True)
    assert job is not None

    prompt = build_summary_prompt(job)

    assert "summary không được rỗng" in prompt
    assert "chủ đề, giải thích chính và điểm đang bàn tới" in prompt
    with pytest.raises(ValueError, match="empty continuity summary"):
        artifact_from_json(
            job,
            json.dumps(
                {
                    "summary": "",
                    "user_constraints": [],
                    "decisions": [],
                    "corrections": [],
                    "open_items": [],
                    "continuity_refs": [],
                }
            ),
        )


def test_summary_job_repairs_an_oversized_candidate_before_publishing() -> None:
    memory = WorkingMemory()
    for index in range(5):
        turn = memory.begin_turn(f"Câu hỏi {index}")
        memory.finish_turn(turn.sequence, f"Trả lời {index}")
    job = memory.prepare_compaction(force=True)
    assert job is not None
    engine = _RepairStructuredEngine()

    artifact, _ = execute_summary_job(job, engine)

    assert engine.calls == 2
    assert "REPAIR PASS" in engine.prompts[1]
    assert artifact.summary == "Bối cảnh đã được compact."


def test_unprovisioned_summary_worker_never_auto_downloads_or_stays_loaded(tmp_path) -> None:
    memory = WorkingMemory(token_counter=lambda _: 15_000)
    for index in range(6):
        turn = memory.begin_turn(f"user {index}")
        memory.finish_turn(turn.sequence, f"assistant {index}")
    job = memory.prepare_compaction()
    assert job is not None
    worker = LocalSummaryWorkerProcess(
        SUMMARY_MODEL_REGISTRY["qwen3_1_7b_q8_0"], model_root=tmp_path
    )
    assert worker.start(job) is False
    assert worker.status.state == "idle"
    assert default_summary_model_root().name == "summary"


def test_production_summary_selection_and_revised_release_gate(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SOCA_SUMMARY_MODEL_ROOT", str(tmp_path))
    worker = build_production_summary_worker()

    assert worker.spec.key == PRODUCTION_SUMMARY_MODEL_KEY
    assert worker.model_root == tmp_path
    memory = WorkingMemory(token_counter=lambda _: 15_000)
    for index in range(6):
        turn = memory.begin_turn(f"user {index}")
        memory.finish_turn(turn.sequence, f"assistant {index}")
    job = memory.prepare_compaction()
    assert job is not None
    assert summary_context_window(job, worker.spec) == 5_120
    assert PRODUCTION_SUMMARY_RELEASE_GATE.accepts(
        schema_valid_rate=1.0,
        single_fact_recall=0.80,
        rolling_fact_recall=0.725,
        negative_state_clean_rate=1.0,
        forbidden_surface_match_rate=0.0,
        cold_clean_exit_rate=1.0,
        cold_worker_stopped_rate=1.0,
        cold_peak_rss_mb_max=6031.0,
    )
    assert not PRODUCTION_SUMMARY_RELEASE_GATE.accepts(
        schema_valid_rate=1.0,
        single_fact_recall=0.79,
        rolling_fact_recall=0.725,
        negative_state_clean_rate=1.0,
        forbidden_surface_match_rate=0.0,
        cold_clean_exit_rate=1.0,
        cold_worker_stopped_rate=1.0,
        cold_peak_rss_mb_max=6031.0,
    )
    assert not PRODUCTION_SUMMARY_RELEASE_GATE.accepts(
        schema_valid_rate=1.0,
        single_fact_recall=0.80,
        rolling_fact_recall=0.725,
        negative_state_clean_rate=1.0,
        forbidden_surface_match_rate=0.0,
        cold_clean_exit_rate=1.0,
        cold_worker_stopped_rate=1.0,
        cold_peak_rss_mb_max=8193.0,
    )
