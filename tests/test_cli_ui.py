from __future__ import annotations

from click.testing import CliRunner

from soca.cli import main


def test_ui_help_shows_quick_examples_not_hidden_compat_options() -> None:
    result = CliRunner().invoke(main, ["ui", "--help"])

    assert result.exit_code == 0, result.output
    assert "soca ui chat" in result.output
    assert "--mode [" not in result.output
    assert "--profile [" not in result.output


def test_ui_default_launches_ink_app(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_launch(*, mode, profile, no_model):
        calls.append({"mode": mode, "profile": profile, "no_model": no_model})
        return 0

    import soca.cli as cli

    monkeypatch.setattr(cli, "_launch_ink_ui", fake_launch)
    result = CliRunner().invoke(main, ["ui", "voice", "baseline", "--no-model"])

    assert result.exit_code == 0, result.output
    assert calls == [{"mode": "voice", "profile": "baseline", "no_model": True}]


def test_ui_bare_launches_ink_splash_without_mode(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_launch(*, mode, profile, no_model):
        calls.append({"mode": mode, "profile": profile, "no_model": no_model})
        return 0

    import soca.cli as cli

    monkeypatch.setattr(cli, "_launch_ink_ui", fake_launch)
    result = CliRunner().invoke(main, ["ui"])

    assert result.exit_code == 0, result.output
    assert calls == [{"mode": None, "profile": None, "no_model": False}]
