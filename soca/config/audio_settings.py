"""Persistent selection of the microphone that Voice captures from.

The value is intentionally a device name rather than a PortAudio index: macOS
reorders those indexes when a USB or Bluetooth device is connected. ``None``
means the user explicitly chose the operating-system default device.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def default_audio_settings_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config_home) if config_home else Path.home() / ".config"
    return base / "soca" / "audio.json"


def load_audio_input_device(path: Path | None = None) -> str | None:
    """Return the explicitly selected device, or ``None`` for system default."""
    path = path or default_audio_settings_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f"Không thể đọc audio settings tại {path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Audio settings tại {path} không phải JSON hợp lệ.") from exc
    if not isinstance(payload, dict) or set(payload) != {"input_device"}:
        raise ValueError(f"Audio settings tại {path} phải chứa đúng trường input_device.")
    device = payload["input_device"]
    if device is not None and (not isinstance(device, str) or not device.strip()):
        raise ValueError(f"Audio settings tại {path} có input_device không hợp lệ.")
    return device


def save_audio_input_device(device: str | None, path: Path | None = None) -> None:
    """Atomically persist the explicit device choice with owner-only access."""
    if device is not None and (not isinstance(device, str) or not device.strip()):
        raise ValueError("audio input device phải là tên thiết bị không rỗng hoặc null.")
    path = path or default_audio_settings_path()
    contents = json.dumps({"input_device": device}, ensure_ascii=False, indent=2) + "\n"
    descriptor: int | None = None
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            descriptor = None
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    except OSError as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise ValueError(f"Không thể lưu audio settings tại {path}: {exc}") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


__all__ = [
    "default_audio_settings_path",
    "load_audio_input_device",
    "save_audio_input_device",
]
