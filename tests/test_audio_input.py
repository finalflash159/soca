from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from soca.config.audio_settings import load_audio_input_device, save_audio_input_device
from soca.core.audio_input import (
    AudioInputDeviceError,
    audio_input_status,
    resolve_audio_input_device,
)


def _devices():
    return [
        {"name": "Built-in Microphone", "max_input_channels": 1},
        {"name": "USB Headset", "max_input_channels": 2},
        {"name": "Built-in Output", "max_input_channels": 0},
    ]


def test_audio_input_settings_round_trip_with_private_permissions(tmp_path):
    path = tmp_path / "audio.json"

    save_audio_input_device("USB Headset", path)

    assert load_audio_input_device(path) == "USB Headset"
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text(encoding="utf-8")) == {"input_device": "USB Headset"}


def test_audio_input_settings_default_to_system_choice(tmp_path):
    assert load_audio_input_device(tmp_path / "audio.json") is None


def test_audio_input_status_marks_system_default(monkeypatch):
    import soca.core.audio_input as audio_input

    monkeypatch.setattr(
        audio_input,
        "sd",
        SimpleNamespace(query_devices=_devices, default=SimpleNamespace(device=np.array([1, 2]))),
    )

    assert audio_input_status(None) == {
        "selected_id": None,
        "selected_label": "USB Headset",
        "uses_system_default": True,
        "devices": [
            {"id": "Built-in Microphone", "label": "Built-in Microphone", "is_system_default": False},
            {"id": "USB Headset", "label": "USB Headset", "is_system_default": True},
        ],
    }


def test_named_audio_input_is_never_replaced_when_missing(monkeypatch):
    import soca.core.audio_input as audio_input

    monkeypatch.setattr(
        audio_input,
        "sd",
        SimpleNamespace(query_devices=_devices, default=SimpleNamespace(device=[0, 2])),
    )

    assert resolve_audio_input_device("USB Headset") == "USB Headset"
    with pytest.raises(AudioInputDeviceError, match="không còn khả dụng"):
        resolve_audio_input_device("Disconnected microphone")
