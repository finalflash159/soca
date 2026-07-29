from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_xdg_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Never let a developer's persisted remote provider affect unit tests."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))


@pytest.fixture
def sparse_knowledge_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CLI behavior tests independent from provisioned production weights."""
    from soca.app import text_runtime

    get_profile = text_runtime.get_voice_runtime_profile

    def resolve(profile_key: str):
        return replace(
            get_profile(profile_key),
            knowledge_retrieval_mode="cached_sparse",
        )

    monkeypatch.setattr(text_runtime, "get_voice_runtime_profile", resolve)
