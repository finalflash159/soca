from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from click.testing import CliRunner

from soca.cli import main
from soca.llm import LLMResult


class FakeChatLLM:
    instances: list[FakeChatLLM] = []

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
        self.calls: list[str] = []
        self.instances.append(self)

    def generate(
        self,
        user_msg: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        inject_persona: bool = True,
    ) -> LLMResult:
        self.calls.append(user_msg)
        return LLMResult(
            text=f"Phản hồi số {len(self.calls)}.",
            prompt=user_msg,
            n_prompt_tokens=12,
            n_completion_tokens=5,
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
    (wiki / "bua-sang.md").write_text(
        "# Bữa sáng\n\nBữa sáng nên có đạm, rau hoặc trái cây, và tinh bột vừa đủ.",
        encoding="utf-8",
    )
    memory = root / "memory"
    memory.mkdir()
    (memory / "profile.md").write_text(
        "# Profile\n\nNgười dùng thích giải thích rõ bằng tiếng Việt.",
        encoding="utf-8",
    )


def test_chat_reuses_one_runtime_across_multiple_turns(monkeypatch, tmp_path: Path) -> None:
    write_vault(tmp_path)
    FakeChatLLM.instances.clear()
    monkeypatch.setattr("soca.app.text_runtime.LocalLlamaCppLLM", FakeChatLLM)

    result = CliRunner().invoke(
        main,
        [
            "chat",
            "--vault",
            str(tmp_path),
            "--no-memory",
        ],
        input="xin chào\nbạn nhớ lượt trước không?\n/exit\n",
    )

    assert result.exit_code == 0, result.output
    assert "SoCa · chat" in result.output
    assert "Route: free_chat" in result.output
    assert "Phản hồi số 1." in result.output
    # The second turn misses the semantic examples and exercises the LLM
    # router's JSON/repair fallback before the free-chat generation.
    assert "Phản hồi số 4." in result.output
    assert len(FakeChatLLM.instances) == 1
    assert FakeChatLLM.instances[0].model_key == "arcee_vylinh_3b_q4_k_m"
    assert len(FakeChatLLM.instances[0].calls) == 4


def test_chat_llm_override_controls_runtime_model(monkeypatch, tmp_path: Path) -> None:
    write_vault(tmp_path)
    FakeChatLLM.instances.clear()
    monkeypatch.setattr("soca.app.text_runtime.LocalLlamaCppLLM", FakeChatLLM)

    result = CliRunner().invoke(
        main,
        [
            "chat",
            "--llm-model",
            "qwen3_0_6b_q8_0",
            "--vault",
            str(tmp_path),
            "--no-memory",
        ],
        input="xin chào\n/exit\n",
    )

    assert result.exit_code == 0, result.output
    assert len(FakeChatLLM.instances) == 1
    assert FakeChatLLM.instances[0].model_key == "qwen3_0_6b_q8_0"


def test_chat_memory_commands_show_and_clear_session(monkeypatch, tmp_path: Path) -> None:
    write_vault(tmp_path)
    FakeChatLLM.instances.clear()
    monkeypatch.setattr("soca.app.text_runtime.LocalLlamaCppLLM", FakeChatLLM)

    result = CliRunner().invoke(
        main,
        [
            "chat",
            "--vault",
            str(tmp_path),
        ],
        input="xin chào\n/memory\n/clear\n/memory\n/exit\n",
    )

    assert result.exit_code == 0, result.output
    assert "Session memory" in result.output
    assert "Recent conversation:" in result.output
    assert "Session memory cleared" in result.output
    assert "<empty>" in result.output


def test_chat_can_run_tool_only_without_llm(tmp_path: Path) -> None:
    write_vault(tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "chat",
            "--vault",
            str(tmp_path),
            "--no-memory",
            "--no-llm",
            "--trace",
        ],
            input="time:\n/exit\n",
    )

    assert result.exit_code == 0, result.output
    assert "Route: tool_direct" in result.output
    assert "local_time.now" in result.output
    assert "used_tool" in result.output


def test_chat_help_and_trace_toggle(tmp_path: Path) -> None:
    write_vault(tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "chat",
            "--vault",
            str(tmp_path),
            "--no-memory",
            "--no-llm",
        ],
        input="/help\n/trace\nđặt hẹn giờ 5 phút\n/exit\n",
    )

    assert result.exit_code == 0, result.output
    assert "Lệnh chat" in result.output
    assert "Trace: on" in result.output
    assert "Route: blocked" in result.output
    assert "unsupported_capability" not in result.output


def test_chat_usage_flag_and_session_command(monkeypatch, tmp_path: Path) -> None:
    write_vault(tmp_path)
    FakeChatLLM.instances.clear()
    monkeypatch.setattr("soca.app.text_runtime.LocalLlamaCppLLM", FakeChatLLM)

    result = CliRunner().invoke(
        main,
        ["chat", "--vault", str(tmp_path), "--no-memory", "--usage"],
        input="xin chào\n/usage\n/exit\n",
    )

    assert result.exit_code == 0, result.output
    assert "route=free_chat" in result.output  # per-turn usage line from --usage
    assert "Session Usage" in result.output  # /usage table
    assert "completion tokens" in result.output
