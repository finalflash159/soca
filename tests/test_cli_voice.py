from __future__ import annotations

from click.testing import CliRunner

from soca.cli import main


def test_voice_command_delegates_to_app_voice_loop(monkeypatch) -> None:
    captured = {}

    def fake_run_voice_loop(
        config,
        *,
        no_speak_rejections: bool = False,
        press_enter_to_record: bool = False,
        warmup: bool = True,
    ) -> int:
        captured["config"] = config
        captured["no_speak_rejections"] = no_speak_rejections
        captured["press_enter_to_record"] = press_enter_to_record
        captured["warmup"] = warmup
        return 0

    monkeypatch.setattr("soca.cli.run_voice_loop", fake_run_voice_loop)

    result = CliRunner().invoke(
        main,
        [
            "voice",
            "--profile",
            "quality",
            "--asr-model",
            "phowhisper_base",
            "--tts-model",
            "valtec_multispeaker",
            "--voice",
            "SF",
            "--max-tokens",
            "96",
            "--no-speak-rejections",
            "--press-enter-to-record",
            "--no-warmup",
            "--no-memory",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["no_speak_rejections"] is True
    assert captured["press_enter_to_record"] is True
    assert captured["warmup"] is False

    config = captured["config"]
    assert config.profile_key == "quality"
    assert config.asr_model == "phowhisper_base"
    assert config.llm_model == "arcee_vylinh_3b_q4_k_m"
    assert config.tts_model == "valtec_multispeaker"
    assert config.tts_voice == "SF"
    assert config.max_tokens == 96
    assert config.no_memory is True


def test_voice_command_shows_help() -> None:
    result = CliRunner().invoke(main, ["voice", "--help"])

    assert result.exit_code == 0
    assert "Run the interactive Soca microphone voice loop" in result.output
    assert "--profile" in result.output
    assert "--tts-model" in result.output
    assert "--press-enter-to-record" in result.output
    assert "--no-warmup" in result.output
