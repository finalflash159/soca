from __future__ import annotations

import sys
import types

from click.testing import CliRunner

from soca.cli import main


def install_fake_tui(monkeypatch):
    calls: list[dict[str, object]] = []

    module = types.ModuleType("soca.app.tui")

    def run_tui(**kwargs):
        calls.append(kwargs)
        return 0

    module.run_tui = run_tui
    monkeypatch.setitem(sys.modules, "soca.app.tui", module)
    return calls


def test_ui_quick_chat_mode_uses_positional_mode(monkeypatch) -> None:
    calls = install_fake_tui(monkeypatch)

    result = CliRunner().invoke(main, ["ui", "chat", "--no-memory", "--no-model"])

    assert result.exit_code == 0, result.output
    assert calls[0]["mode"] == "chat"
    assert calls[0]["profile"] == "baseline"
    assert calls[0]["no_model"] is True
    assert calls[0]["text_runtime"].no_memory is True
    assert calls[0]["text_runtime"].llm_model == "arcee_vylinh_3b_q4_k_m"
    assert calls[0]["voice_runtime"].llm_model == "arcee_vylinh_3b_q4_k_m"


def test_ui_quick_voice_profile_uses_positional_profile(monkeypatch) -> None:
    calls = install_fake_tui(monkeypatch)

    result = CliRunner().invoke(main, ["ui", "voice", "quality", "--no-model"])

    assert result.exit_code == 0, result.output
    assert calls[0]["mode"] == "voice"
    assert calls[0]["profile"] == "quality"
    assert calls[0]["text_runtime"].llm_model == "arcee_vylinh_3b_q4_k_m"
    assert calls[0]["voice_runtime"].llm_model == "arcee_vylinh_3b_q4_k_m"


def test_ui_hidden_llm_model_override_applies_to_text_and_voice_runtime(monkeypatch) -> None:
    calls = install_fake_tui(monkeypatch)

    result = CliRunner().invoke(
        main,
        ["ui", "voice", "quality", "--llm-model", "qwen3_0_6b_q8_0", "--no-model"],
    )

    assert result.exit_code == 0, result.output
    assert calls[0]["text_runtime"].llm_model == "qwen3_0_6b_q8_0"
    assert calls[0]["voice_runtime"].llm_model == "qwen3_0_6b_q8_0"


def test_ui_help_shows_quick_examples_not_hidden_compat_options() -> None:
    result = CliRunner().invoke(main, ["ui", "--help"])

    assert result.exit_code == 0, result.output
    assert "soca ui chat" in result.output
    assert "--mode [" not in result.output
    assert "--profile [" not in result.output
