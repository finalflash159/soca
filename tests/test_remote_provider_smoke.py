from __future__ import annotations

import json

from scripts import smoke_test_remote_llm as smoke
from soca.llm import LLMResult


class FakeRemoteEngine:
    def generate(self, user_msg: str, **kwargs) -> LLMResult:
        del kwargs
        return LLMResult(
            text="Đây là phản hồi kiểm tra.",
            prompt=user_msg,
            n_prompt_tokens=12,
            n_completion_tokens=6,
            ttft_ms=1.0,
            total_latency_ms=2.0,
            tokens_per_second=3_000.0,
            provider_trace={
                "provider": "openrouter",
                "model": "test/model",
                "attempt_count": 1,
                "retry_count": 0,
            },
        )

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def test_smoke_harness_records_chat_and_voice_transcript_without_audio(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(
        smoke,
        "DEFAULT_LLM_ENGINE_FACTORY",
        lambda settings, secrets: FakeRemoteEngine(),
    )

    receipts = smoke.run_provider("openrouter", "test/model", max_tokens=96)
    artifact = tmp_path / "provider-smoke.json"
    smoke._write_artifact(artifact, receipts, max_tokens=96)
    payload = json.loads(artifact.read_text(encoding="utf-8"))

    assert [item.surface for item in receipts] == ["chat", "voice_transcript"]
    assert all(item.provider_called for item in receipts)
    assert all(item.terminal == "achieved" for item in receipts)
    assert payload["run_type"] == "real_provider_smoke"
    assert payload["benchmark_eligible"] is False
    assert payload["scenario"]["revision"].startswith("sha256:")
    assert payload["models"][0]["revision"].startswith("provider-managed")
    assert "no microphone" in payload["configuration"]["voice_scope"]
    assert payload["raw_log"]["committed"] is False
    assert "exclude" in payload["decision"]
    assert len(payload["receipts"]) == 2
