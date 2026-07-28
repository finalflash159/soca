from __future__ import annotations

import json

from soca.llm import LLMResult
from soca.memory.summary import SUMMARY_MODEL_REGISTRY, execute_summary_job
from soca.memory.working import WorkingMemory


class _StructuredEngine:
    def generate_structured(self, prompt: str, **kwargs: object) -> LLMResult:
        assert "Previous summary:" in prompt
        assert kwargs["max_tokens"] == 384
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


def test_summary_registry_is_separate_and_summary_job_uses_structured_local_contract() -> None:
    assert "qwen3_1_7b_q8_0" in SUMMARY_MODEL_REGISTRY
    memory = WorkingMemory(token_counter=lambda _: 1000)
    for index in range(6):
        turn = memory.begin_turn(f"user {index}")
        memory.finish_turn(turn.sequence, f"assistant {index}")
    job = memory.prepare_compaction()
    assert job is not None
    artifact, result = execute_summary_job(job, _StructuredEngine())
    assert artifact.generation == job.generation
    assert artifact.source_through_sequence == 2
    assert result.text
