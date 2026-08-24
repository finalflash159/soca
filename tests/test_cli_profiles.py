from __future__ import annotations

from click.testing import CliRunner

from soca.cli import main


def test_profiles_command_lists_only_qwen_voice_profiles(monkeypatch) -> None:
    def fail_voice_loop(*_args, **_kwargs):
        raise AssertionError("profiles must not run the voice loop")

    monkeypatch.setattr("soca.app.voice_loop.run_voice_loop", fail_voice_loop)

    result = CliRunner().invoke(main, ["profiles"])

    assert result.exit_code == 0, result.output
    assert "SoCa Runtime Profiles" in result.output
    assert "qwen-release" in result.output
    assert "quality" not in result.output
    assert "edge" not in result.output
    assert "qwen3_asr_0_6b" in result.output
    assert "phowhisper_small" not in result.output
    assert "arcee_vylinh_3b_q4_k_m" in result.output
    assert "valtec_multispeaker" in result.output


def test_default_profile_uses_benchmarked_knowledge_retrieval() -> None:
    from soca.core.profiles import get_voice_runtime_profile

    profile = get_voice_runtime_profile("qwen-release")

    assert profile.knowledge_retrieval_mode == "hybrid"
    assert profile.knowledge_dense_backend == "aiteamvn_v2"


def test_profiles_command_can_show_artifact_paths() -> None:
    result = CliRunner().invoke(main, ["profiles", "--show-paths"])

    assert result.exit_code == 0, result.output
    assert "Profile Artifact Paths" in result.output
    assert "models" in result.output


def test_profiles_command_surfaces_profile_validation_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        "soca.app.profiles.validate_voice_runtime_profiles",
        lambda: ["qwen-release: unknown ASR model 'not_real'"],
    )

    result = CliRunner().invoke(main, ["profiles"])

    assert result.exit_code == 0, result.output
    assert "qwen-release" in result.output
    assert "invalid" in result.output
    assert "unknown ASR model" in result.output


def test_profiles_command_surfaces_global_validation_errors(monkeypatch) -> None:
    # A whole-config invariant error has no "<profile>:" prefix; it must still
    # mark the profile table invalid instead of being silently dropped.
    monkeypatch.setattr(
        "soca.app.profiles.validate_voice_runtime_profiles",
        lambda: ["runtime profiles must contain exactly ['qwen-release'], got ['qwen-release', 'extra']"],
    )

    result = CliRunner().invoke(main, ["profiles"])

    assert result.exit_code == 0, result.output
    assert "invalid" in result.output
    # Rich wraps table cells, so normalize whitespace before matching the message.
    normalized = " ".join(result.output.split())
    assert "runtime profiles must contain exactly" in normalized


def test_status_command_shows_lightweight_runtime_overview() -> None:
    result = CliRunner().invoke(main, ["status"])

    assert result.exit_code == 0, result.output
    assert "SoCa Status" in result.output
    assert "Primary command" in result.output
    assert "uv run soca voice" in result.output
    assert "--profile baseline" not in result.output
    assert "Runtime profiles" in result.output


def test_voice_command_rejects_tts_model_override() -> None:
    result = CliRunner().invoke(
        main,
        ["voice", "qwen-release", "--tts-model", "other_tts"],
    )

    assert result.exit_code != 0
    assert "No such option: --tts-model" in result.output


def test_voice_command_rejects_removed_profile() -> None:
    result = CliRunner().invoke(main, ["voice", "quality"])

    assert result.exit_code != 0
    assert "Invalid value" in result.output


def test_root_help_keeps_ui_and_engine_commands() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0, result.output
    assert "ui" in result.output
    assert "engine" in result.output
