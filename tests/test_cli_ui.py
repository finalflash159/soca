from __future__ import annotations

from click.testing import CliRunner

from soca.cli import main
from soca.knowledge.vault import default_vault_root


def test_ui_help_shows_quick_examples_not_hidden_compat_options() -> None:
    result = CliRunner().invoke(main, ["ui", "--help"])

    assert result.exit_code == 0, result.output
    assert "soca ui chat" in result.output
    assert "--mode [" not in result.output
    assert "--profile [" not in result.output
    assert "--vault PATH" in result.output


def test_ui_default_launches_ink_app(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_launch(*, mode, profile, no_model, vault):
        calls.append({"mode": mode, "profile": profile, "no_model": no_model, "vault": vault})
        return 0

    import soca.cli as cli

    monkeypatch.setattr(cli, "_launch_ink_ui", fake_launch)
    result = CliRunner().invoke(main, ["ui", "voice", "baseline", "--no-model"])

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "mode": "voice",
            "profile": "baseline",
            "no_model": True,
            "vault": default_vault_root(),
        }
    ]


def test_ui_bare_launches_ink_splash_without_mode(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_launch(*, mode, profile, no_model, vault):
        calls.append({"mode": mode, "profile": profile, "no_model": no_model, "vault": vault})
        return 0

    import soca.cli as cli

    monkeypatch.setattr(cli, "_launch_ink_ui", fake_launch)
    result = CliRunner().invoke(main, ["ui"])

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "mode": None,
            "profile": None,
            "no_model": False,
            "vault": default_vault_root(),
        }
    ]


def test_ui_passes_explicit_vault_to_ink_app(monkeypatch, tmp_path) -> None:
    calls: list[dict[str, object]] = []

    def fake_launch(*, mode, profile, no_model, vault):
        calls.append({"mode": mode, "profile": profile, "no_model": no_model, "vault": vault})
        return 0

    import soca.cli as cli

    monkeypatch.setattr(cli, "_launch_ink_ui", fake_launch)
    result = CliRunner().invoke(main, ["ui", "chat", "--vault", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "mode": "chat",
            "profile": None,
            "no_model": False,
            "vault": tmp_path,
        }
    ]


def test_ui_uses_soca_vault_when_no_flag_is_given(monkeypatch, tmp_path) -> None:
    calls: list[dict[str, object]] = []

    def fake_launch(*, mode, profile, no_model, vault):
        calls.append({"mode": mode, "profile": profile, "no_model": no_model, "vault": vault})
        return 0

    import soca.cli as cli

    monkeypatch.setattr(cli, "_launch_ink_ui", fake_launch)
    monkeypatch.setenv("SOCA_VAULT", str(tmp_path))
    result = CliRunner().invoke(main, ["ui", "chat"])

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "mode": "chat",
            "profile": None,
            "no_model": False,
            "vault": tmp_path.resolve(),
        }
    ]
