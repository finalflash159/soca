"""Keyring-first API-key lookup that never writes project ``.env`` files."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from soca.llm.providers import get_provider

KEYRING_SERVICE = "soca-llm"
DEFAULT_CONFIG_DIR = Path.home() / ".config" / "soca"
DEFAULT_KEYS_PATH = DEFAULT_CONFIG_DIR / "keys.json"
READ_DOTENV_ENV = "SOCA_READ_DOTENV"


class _Unset:
    pass


_UNSET = _Unset()


class KeyringBackend(Protocol):
    def get_password(self, service: str, username: str) -> str | None: ...

    def set_password(self, service: str, username: str, value: str) -> None: ...


class SecretStore:
    """Resolve provider keys from keyring, env, opt-in dotenv, then JSON."""

    def __init__(
        self,
        *,
        keyring_module: KeyringBackend | None | _Unset = _UNSET,
        env: Mapping[str, str] | None = None,
        dotenv_path: Path | None = None,
        json_path: Path = DEFAULT_KEYS_PATH,
    ) -> None:
        self._keyring = _load_keyring() if keyring_module is _UNSET else keyring_module
        self._env = dict(os.environ if env is None else env)
        self._dotenv_path = (
            dotenv_path
            if dotenv_path is not None
            else Path.cwd() / ".env"
            if self._env.get(READ_DOTENV_ENV) == "1"
            else None
        )
        self._json_path = json_path

    def get_key(self, provider_key: str) -> str | None:
        provider = get_provider(provider_key)
        keyring_value = self._read_keyring(provider.key)
        if keyring_value:
            return keyring_value

        environment_value = self._env.get(provider.api_key_env, "").strip()
        if environment_value:
            return environment_value

        dotenv_value = _read_dotenv(self._dotenv_path).get(provider.api_key_env, "").strip()
        if dotenv_value:
            return dotenv_value

        return _read_json_keys(self._json_path).get(provider.key)

    def set_key(self, provider_key: str, value: str) -> None:
        provider = get_provider(provider_key)
        secret = value.strip()
        if not secret:
            raise ValueError("API key must not be empty")

        if self._write_keyring(provider.key, secret):
            return
        _write_json_key(self._json_path, provider.key, secret)

    def has_key(self, provider_key: str) -> bool:
        return self.get_key(provider_key) is not None

    @staticmethod
    def mask(value: str) -> str:
        if not value:
            return ""
        if len(value) <= 8:
            return "••••"
        prefix = value[:3] if value.startswith("sk-") else value[:2]
        return f"{prefix}...{value[-4:]}"

    def _read_keyring(self, provider_key: str) -> str | None:
        if self._keyring is None:
            return None
        try:
            value = self._keyring.get_password(KEYRING_SERVICE, provider_key)
        except Exception:  # unavailable keyring must not block env/fallback lookup
            return None
        return value.strip() if value else None

    def _write_keyring(self, provider_key: str, value: str) -> bool:
        if self._keyring is None:
            return False
        try:
            self._keyring.set_password(KEYRING_SERVICE, provider_key, value)
        except Exception:  # unavailable OS backend uses the documented JSON fallback
            return False
        return True


def _load_keyring() -> KeyringBackend | None:
    try:
        import keyring
    except ImportError:
        return None
    return keyring


def _read_dotenv(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {}
    except OSError:
        return {}

    values: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", maxsplit=1)
        normalized_name = name.strip()
        if normalized_name.startswith("export "):
            normalized_name = normalized_name.removeprefix("export ").strip()
        if normalized_name:
            values[normalized_name] = value.strip().strip("\"'")
    return values


def _read_json_keys(path: Path) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        key: value.strip()
        for key, value in payload.items()
        if isinstance(key, str) and isinstance(value, str) and value.strip()
    }


def _write_json_key(path: Path, provider_key: str, value: str) -> None:
    existing = _read_json_keys(path)
    updated = {**existing, provider_key: value}
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
            output.write(json.dumps(updated, indent=2, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
        os.chmod(path, 0o600)
    except OSError as exc:
        raise ValueError(f"Không thể lưu API key dự phòng tại {path}: {exc}") from exc
    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


__all__ = ["DEFAULT_KEYS_PATH", "KEYRING_SERVICE", "READ_DOTENV_ENV", "SecretStore"]
