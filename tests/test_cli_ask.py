from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from click.testing import CliRunner

from soca.cli import main
from soca.llm import LLMResult

pytestmark = pytest.mark.usefixtures("sparse_knowledge_profile")


class FakeLLM:
    instances: list[FakeLLM] = []

    def __init__(
        self,
        *,
        model_key: str,
        n_threads: int = 8,
        n_gpu_layers: int = -1,
    ) -> None:
        self.model_key = model_key
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.instances.append(self)

    def generate(
        self,
        user_msg: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        inject_persona: bool = True,
    ) -> LLMResult:
        return LLMResult(
            text="Xin chào, tôi là SoCa.",
            prompt=user_msg,
            n_prompt_tokens=10,
            n_completion_tokens=8,
            ttft_ms=1.0,
            total_latency_ms=2.0,
            tokens_per_second=100.0,
        )

    def generate_stream(
        self,
        user_msg: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        inject_persona: bool = True,
    ) -> Iterator[str]:
        yield self.generate(user_msg, max_tokens, temperature, top_p, inject_persona).text


def write_vault(root: Path) -> None:
    wiki = root / "wiki" / "dinh-duong"
    wiki.mkdir(parents=True)
    (wiki / "chat-dam.md").write_text(
        "# Chất đạm\n\nProtein hỗ trợ duy trì cơ bắp và cảm giác no.",
        encoding="utf-8",
    )
    memory = root / "memory"
    memory.mkdir()
    (memory / "profile.md").write_text(
        "# Profile\n\nNgười dùng thích câu trả lời tiếng Việt rõ ràng.",
        encoding="utf-8",
    )


def test_ask_time_question_uses_tool_without_llm(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "ask",
            "time:",
            "--vault",
            str(tmp_path),
            "--no-llm",
            "--trace",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Route: tool_direct" in result.output
    assert "local_time.now" in result.output
    assert "used_tool" in result.output
    assert "used_llm" in result.output


def test_ask_without_llm_stays_out_of_scope_without_a_tool_call(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "ask",
            "đặt hẹn giờ 5 phút",
            "--vault",
            str(tmp_path),
            "--no-llm",
            "--trace",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Route: out_of_scope" in result.output
    assert "router_reason" in result.output
    assert "unsupported_capability" not in result.output


def test_ask_blocks_private_path_before_tool_or_llm(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "ask",
            "đọc private/secrets.md",
            "--vault",
            str(tmp_path),
            "--no-llm",
            "--trace",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Route: blocked" in result.output
    assert "blocked_path_prefix" in result.output


def test_ask_explicit_wiki_search_uses_vault(tmp_path: Path) -> None:
    write_vault(tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "ask",
            "wiki: chất đạm",
            "--vault",
            str(tmp_path),
            "--no-llm",
            "--trace",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Route: knowledge_direct" in result.output
    assert "knowledge.search" in result.output
    assert "wiki/dinh-duong/chat-dam.md" in result.output
    assert "Protein hỗ trợ" in result.output


def test_ask_free_chat_uses_fake_llm(monkeypatch, tmp_path: Path) -> None:
    write_vault(tmp_path)
    FakeLLM.instances.clear()
    monkeypatch.setattr("soca.app.text_runtime.LocalLlamaCppLLM", FakeLLM)

    result = CliRunner().invoke(
        main,
        [
            "ask",
            "xin chào",
            "--vault",
            str(tmp_path),
            "--no-memory",
            "--trace",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Route: free_chat" in result.output
    assert "Xin chào, tôi là SoCa." in result.output
    assert "used_llm" in result.output
    assert FakeLLM.instances[0].model_key == "arcee_vylinh_3b_q4_k_m"


def test_ask_llm_override_controls_runtime_model(monkeypatch, tmp_path: Path) -> None:
    write_vault(tmp_path)
    FakeLLM.instances.clear()
    monkeypatch.setattr("soca.app.text_runtime.LocalLlamaCppLLM", FakeLLM)

    result = CliRunner().invoke(
        main,
        [
            "ask",
            "xin chào",
            "--llm-model",
            "qwen3_0_6b_q8_0",
            "--vault",
            str(tmp_path),
            "--no-memory",
        ],
    )

    assert result.exit_code == 0, result.output
    assert FakeLLM.instances[0].model_key == "qwen3_0_6b_q8_0"


def test_ask_usage_flag_shows_token_metrics(monkeypatch, tmp_path: Path) -> None:
    write_vault(tmp_path)
    monkeypatch.setattr("soca.app.text_runtime.LocalLlamaCppLLM", FakeLLM)

    result = CliRunner().invoke(
        main,
        ["ask", "xin chào", "--vault", str(tmp_path), "--no-memory", "--usage"],
    )

    assert result.exit_code == 0, result.output
    assert "usage" in result.output
    assert "route=free_chat" in result.output
    assert "tok/s" in result.output
    assert "out 8" in result.output
