from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_xdg_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Never let a developer's persisted remote provider affect unit tests."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
