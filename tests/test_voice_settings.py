from __future__ import annotations

import json

import pytest

from soca.config.voice_settings import (
    DEFAULT_VOICE_PROFILE,
    load_voice_profile,
    save_voice_profile,
)


def test_voice_profile_defaults_without_a_file(tmp_path):
    assert load_voice_profile(tmp_path / "voice.json") == DEFAULT_VOICE_PROFILE


def test_voice_profile_round_trips_with_private_permissions(tmp_path):
    path = tmp_path / "voice.json"

    save_voice_profile("qwen-release", path)

    assert load_voice_profile(path) == "qwen-release"
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text(encoding="utf-8")) == {"profile": "qwen-release"}


def test_voice_profile_save_wraps_config_directory_errors(tmp_path, monkeypatch):
    path = tmp_path / "voice.json"

    def fail_mkdir(*_args, **_kwargs):
        raise PermissionError("read-only config")

    monkeypatch.setattr(path.parent.__class__, "mkdir", fail_mkdir)

    with pytest.raises(ValueError, match="Không thể lưu voice settings"):
        save_voice_profile("qwen-release", path)


@pytest.mark.parametrize(
    "payload",
    [
        "{}",
        '{"profile": ""}',
        '{"profile": 42}',
        '{"profile": "baseline", "extra": true}',
    ],
)
def test_invalid_voice_profile_file_is_not_silently_replaced(tmp_path, payload):
    path = tmp_path / "voice.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError):
        load_voice_profile(path)
