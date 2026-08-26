"""Truthful audio-input enumeration and validation for Voice.

This module never substitutes a different microphone. A saved device that is
gone is an explicit error; only ``None`` deliberately means "use the current
system default".
"""

from __future__ import annotations

from dataclasses import dataclass

import sounddevice as sd


class AudioInputDeviceError(RuntimeError):
    """The requested recording device cannot be used as selected."""


@dataclass(frozen=True)
class AudioInputDevice:
    key: str
    label: str
    is_system_default: bool

    def as_protocol(self) -> dict[str, object]:
        return {
            "id": self.key,
            "label": self.label,
            "is_system_default": self.is_system_default,
        }


def list_audio_input_devices() -> tuple[AudioInputDevice, ...]:
    """List presently available capture endpoints without opening one."""
    try:
        devices = sd.query_devices()
        defaults = sd.default.device
    except Exception as exc:  # PortAudio/CoreAudio boundary
        raise AudioInputDeviceError(f"Không thể liệt kê microphone: {exc}") from exc

    try:
        default_index = int(defaults[0])
    except (IndexError, TypeError, ValueError):
        default_index = None
    available: list[AudioInputDevice] = []
    for index, raw in enumerate(devices):
        try:
            channels = int(raw["max_input_channels"])
            name = str(raw["name"])
        except (KeyError, TypeError, ValueError):
            continue
        if channels <= 0 or not name:
            continue
        available.append(
            AudioInputDevice(
                key=name,
                label=name,
                is_system_default=index == default_index,
            )
        )
    if not available:
        raise AudioInputDeviceError("Không có microphone nào khả dụng.")
    return tuple(available)


def resolve_audio_input_device(selected: str | None) -> str | None:
    """Validate an explicit device immediately before capture opens it."""
    devices = list_audio_input_devices()
    if selected is None:
        return None
    if any(device.key == selected for device in devices):
        return selected
    raise AudioInputDeviceError(
        f"Microphone đã chọn không còn khả dụng: {selected}. Chọn một microphone khác trong Voice settings."
    )


def audio_input_status(selected: str | None) -> dict[str, object]:
    """Return selection plus current enumeration for the desktop protocol."""
    devices = list_audio_input_devices()
    active = next((item for item in devices if item.key == selected), None)
    if selected is not None and active is None:
        raise AudioInputDeviceError(
            f"Microphone đã chọn không còn khả dụng: {selected}. Chọn một microphone khác trong Voice settings."
        )
    default = next((item for item in devices if item.is_system_default), None)
    return {
        "selected_id": selected,
        "selected_label": active.label if active is not None else (default.label if default else None),
        "uses_system_default": selected is None,
        "devices": [item.as_protocol() for item in devices],
    }


__all__ = [
    "AudioInputDevice",
    "AudioInputDeviceError",
    "audio_input_status",
    "list_audio_input_devices",
    "resolve_audio_input_device",
]
