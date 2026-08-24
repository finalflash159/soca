from __future__ import annotations

import pytest

from soca.model_paths import default_model_root


def test_model_root_prefers_explicit_absolute_runtime_location(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SOCA_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "other-data"))

    assert default_model_root() == (tmp_path / "models").resolve()


def test_model_root_rejects_relative_explicit_runtime_location(monkeypatch) -> None:
    monkeypatch.setenv("SOCA_MODEL_ROOT", "models")

    with pytest.raises(ValueError, match="SOCA_MODEL_ROOT must be absolute"):
        default_model_root()


def test_model_root_uses_xdg_data_home_when_no_explicit_location(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("SOCA_MODEL_ROOT", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    assert default_model_root() == tmp_path / "soca" / "models"
