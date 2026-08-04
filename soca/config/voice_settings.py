"""Persistent, non-secret voice runtime selection."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

DEFAULT_VOICE_PROFILE = "baseline"


def default_voice_settings_path() -> Path:
    """Resolve the voice selection path using the same XDG config root as LLM settings."""
    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config_home) if config_home else Path.home() / ".config"
    return base / "soca" / "voice.json"


def load_voice_profile(path: Path | None = None) -> str:
    """Load the last selected voice profile, or the initial baseline selection."""
    path = path or default_voice_settings_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return DEFAULT_VOICE_PROFILE
    except OSError as exc:
        raise ValueError(f"Không thể đọc voice settings tại {path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Voice settings tại {path} không phải JSON hợp lệ.") from exc
    if not isinstance(payload, dict) or set(payload) != {"profile"}:
        raise ValueError(f"Voice settings tại {path} phải chứa đúng trường profile.")
    profile = payload["profile"]
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError(f"Voice settings tại {path} có profile không hợp lệ.")
    return profile


def save_voice_profile(profile: str, path: Path | None = None) -> None:
    """Atomically persist the selected profile with owner-only permissions."""
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError("voice profile phải là chuỗi không rỗng.")
    path = path or default_voice_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = json.dumps({"profile": profile}, ensure_ascii=False, indent=2) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
        os.chmod(path, 0o600)
    except OSError as exc:
        raise ValueError(f"Không thể lưu voice settings tại {path}: {exc}") from exc
    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


__all__ = [
    "DEFAULT_VOICE_PROFILE",
    "default_voice_settings_path",
    "load_voice_profile",
    "save_voice_profile",
]
