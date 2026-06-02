from __future__ import annotations

from click.testing import CliRunner

from soca.cli import main


def test_profiles_command_lists_runtime_profiles_without_running_voice_loop(monkeypatch) -> None:
    def fail_voice_loop(*_args, **_kwargs):
        raise AssertionError("profiles must not run the voice loop")

    monkeypatch.setattr("soca.cli.run_voice_loop", fail_voice_loop)

    result = CliRunner().invoke(main, ["profiles"])

    assert result.exit_code == 0, result.output
    assert "SoCa Runtime Profiles" in result.output
    assert "baseline" in result.output
    assert "quality" in result.output
    assert "edge" in result.output
    assert "phowhisper_base" in result.output
    assert "arcee_vylinh_3b_q4_k_m" in result.output
    assert "valtec_multispeaker" in result.output


def test_profiles_command_can_show_artifact_paths() -> None:
    result = CliRunner().invoke(main, ["profiles", "--show-paths"])

    assert result.exit_code == 0, result.output
    assert "Profile Artifact Paths" in result.output
    assert "models" in result.output


def test_profiles_command_surfaces_profile_validation_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        "soca.app.profiles.validate_voice_runtime_profiles",
        lambda: ["baseline: unknown ASR model 'not_real'"],
    )

    result = CliRunner().invoke(main, ["profiles"])

    assert result.exit_code == 0, result.output
    assert "baseline" in result.output
    assert "invalid" in result.output
    assert "unknown ASR model" in result.output


def test_status_command_shows_lightweight_runtime_overview() -> None:
    result = CliRunner().invoke(main, ["status"])

    assert result.exit_code == 0, result.output
    assert "SoCa Status" in result.output
    assert "Primary command" in result.output
    assert "uv run soca voice --profile baseline" in result.output
    assert "Runtime profiles" in result.output

