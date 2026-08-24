from __future__ import annotations

from click.testing import CliRunner

from soca.cli import main


def test_voice_command_delegates_to_app_voice_loop(monkeypatch) -> None:
    captured = {}

    def fake_run_voice_loop(
        config,
        *,
        no_speak_repairs: bool = False,
        no_speak_rejections: bool = False,
        press_enter_to_record: bool = False,
        warmup: bool = True,
        show_usage: bool = False,
        player=None,
    ) -> int:
        captured["config"] = config
        captured["no_speak_repairs"] = no_speak_repairs
        captured["no_speak_rejections"] = no_speak_rejections
        captured["press_enter_to_record"] = press_enter_to_record
        captured["warmup"] = warmup
        captured["show_usage"] = show_usage
        captured["player"] = player
        return 0

    # Stub the duplex sink so the CLI does not load the real models here.
    player_sentinel = object()
    monkeypatch.setattr("soca.core.duplex_aec_sink.DuplexAecSink", lambda **kwargs: player_sentinel)
    monkeypatch.setattr("soca.app.voice_loop.run_voice_loop", fake_run_voice_loop)

    result = CliRunner().invoke(
        main,
        [
            "voice",
            "--profile",
            "qwen-release",
            "--voice",
            "SF",
            "--max-tokens",
            "96",
            "--no-speak-repairs",
            "--press-enter-to-record",
            "--no-warmup",
            "--no-memory",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["no_speak_repairs"] is True
    assert captured["press_enter_to_record"] is True
    assert captured["warmup"] is False

    assert captured["player"] is player_sentinel  # console barge-in wired via DuplexAecSink

    config = captured["config"]
    assert config.profile_key == "qwen-release"
    assert config.asr_model == "qwen3_asr_0_6b"
    assert config.llm_model == "arcee_vylinh_3b_q4_k_m"
    assert not hasattr(config, "tts_model")
    assert config.tts_voice == "SF"
    assert config.max_tokens == 96
    assert config.no_memory is True


def test_voice_command_accepts_quick_profile_argument(monkeypatch) -> None:
    captured = {}

    def fake_run_voice_loop(
        config,
        *,
        no_speak_repairs: bool = False,
        no_speak_rejections: bool = False,
        press_enter_to_record: bool = False,
        warmup: bool = True,
        show_usage: bool = False,
        player=None,
    ) -> int:
        captured["config"] = config
        captured["warmup"] = warmup
        return 0

    monkeypatch.setattr("soca.core.duplex_aec_sink.DuplexAecSink", lambda **kwargs: object())
    monkeypatch.setattr("soca.app.voice_loop.run_voice_loop", fake_run_voice_loop)

    result = CliRunner().invoke(main, ["voice", "qwen-release", "--no-warmup"])

    assert result.exit_code == 0, result.output
    assert captured["config"].profile_key == "qwen-release"
    assert captured["warmup"] is False


def test_voice_command_shows_help() -> None:
    result = CliRunner().invoke(main, ["voice", "--help"])

    assert result.exit_code == 0
    assert "Run the interactive SoCa microphone voice loop" in result.output
    assert "Quick form: soca voice [profile]" in result.output
    assert "soca voice qwen-release" in result.output
    assert "soca voice quality" not in result.output
    assert "soca voice edge" not in result.output
    assert "--profile" not in result.output
    assert "--tts-model" not in result.output
    assert "--press-enter-to-record" in result.output
    assert "--no-warmup" in result.output
