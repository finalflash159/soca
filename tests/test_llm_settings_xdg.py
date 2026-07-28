from __future__ import annotations

from pathlib import Path

from soca.config.llm_settings import default_settings_path, load_settings


def test_default_settings_path_respects_xdg_at_call_time(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert default_settings_path() == tmp_path / "soca" / "llm.json"
    assert load_settings().backend == "local"
