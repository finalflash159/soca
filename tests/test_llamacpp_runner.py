from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from soca.llm import LocalLlamaCppLLM


class FakeLlama:
    instances: list[FakeLlama] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.close_calls = 0
        self.completion_calls: list[dict[str, Any]] = []
        self.chat_calls: list[dict[str, Any]] = []
        FakeLlama.instances.append(self)

    def close(self) -> None:
        self.close_calls += 1

    def __call__(self, prompt: str, **kwargs: Any):
        self.completion_calls.append({"prompt": prompt, **kwargs})
        yield {"choices": [{"text": "Xin chào"}]}
        yield {"choices": [{"text": " bạn."}]}

    def create_chat_completion(self, messages: list[dict[str, str]], **kwargs: Any):
        self.chat_calls.append({"messages": messages, **kwargs})
        yield {"choices": [{"delta": {"content": "Chào"}}]}
        yield {"choices": [{"delta": {"content": " bạn."}}]}

    def tokenize(self, text: bytes, add_bos: bool = True) -> list[bytes]:
        return text.split()


@pytest.fixture(autouse=True)
def fake_llama(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeLlama.instances = []
    monkeypatch.setattr("soca.llm.llamacpp_runner.Llama", FakeLlama)


def touch_model(tmp_path: Path) -> Path:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"fake gguf")
    return model_path


def test_completion_model_uses_completion_api_and_stop_sequences(tmp_path: Path) -> None:
    llm = LocalLlamaCppLLM(
        model_key="phogpt_4b_q4_k_m",
        model_path=touch_model(tmp_path),
    )

    result = llm.generate("Bạn là ai?", max_tokens=16, temperature=0.2)

    fake = FakeLlama.instances[0]
    call = fake.completion_calls[0]
    assert result.text == "Xin chào bạn."
    assert "### Câu hỏi:" in call["prompt"]
    assert call["stop"] == ["### Câu hỏi:"]
    assert call["stream"] is True
    assert fake.chat_calls == []


def test_chat_model_uses_chat_completion_api(tmp_path: Path) -> None:
    llm = LocalLlamaCppLLM(
        model_key="arcee_vylinh_3b_q4_k_m",
        model_path=touch_model(tmp_path),
    )

    result = llm.generate("Xin chào", max_tokens=16)

    fake = FakeLlama.instances[0]
    call = fake.chat_calls[0]
    assert fake.kwargs["n_ctx"] == 32768
    assert result.text == "Chào bạn."
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][1] == {"role": "user", "content": "Xin chào"}
    assert call["stream"] is True
    assert fake.completion_calls == []


def test_chat_format_is_passed_when_registry_requires_fallback(tmp_path: Path) -> None:
    LocalLlamaCppLLM(
        model_key="vinallama_2_7b_q5_0",
        model_path=touch_model(tmp_path),
    )

    assert FakeLlama.instances[0].kwargs["chat_format"] == "chatml"


def test_missing_model_error_points_to_generic_downloader(tmp_path: Path) -> None:
    missing_model = tmp_path / "missing.gguf"

    with pytest.raises(FileNotFoundError, match="scripts/download_llm.py --model phogpt_4b_q4_k_m"):
        LocalLlamaCppLLM(model_key="phogpt_4b_q4_k_m", model_path=missing_model)


def test_close_releases_native_model_handle_idempotently(tmp_path: Path) -> None:
    llm = LocalLlamaCppLLM(
        model_key="arcee_vylinh_3b_q4_k_m",
        model_path=touch_model(tmp_path),
    )

    llm.close()
    llm.close()

    assert llm._native_closed is True
    assert FakeLlama.instances[0].close_calls == 1


def test_qwen_stream_output_strips_think_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ThinkingFakeLlama(FakeLlama):
        def create_chat_completion(self, messages: list[dict[str, str]], **kwargs: Any):
            self.chat_calls.append({"messages": messages, **kwargs})
            yield {"choices": [{"delta": {"content": "<think>hidden</think>"}}]}
            yield {"choices": [{"delta": {"content": "Trả lời ngắn."}}]}

    monkeypatch.setattr("soca.llm.llamacpp_runner.Llama", ThinkingFakeLlama)

    llm = LocalLlamaCppLLM(
        model_key="qwen3_0_6b_q8_0",
        model_path=touch_model(tmp_path),
    )

    result = llm.generate("GPU là gì?", max_tokens=16)

    fake = ThinkingFakeLlama.instances[0]
    assert fake.chat_calls[0]["messages"][-1]["content"].endswith("/no_think")
    assert result.text == "Trả lời ngắn."
