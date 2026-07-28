"""Validated, non-secret settings for selecting the text LLM backend."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from soca.llm.providers import get_provider

Backend = Literal["local", "remote"]
DEFAULT_CONFIG_DIR = Path.home() / ".config" / "soca"
DEFAULT_SETTINGS_PATH = DEFAULT_CONFIG_DIR / "llm.json"
DEFAULT_LOCAL_MODEL_ID = "arcee_vylinh_3b_q4_k_m"
DEFAULT_MAX_TOKENS = 4_096
_LEGACY_DEFAULT_MAX_TOKENS = 160


@dataclass(frozen=True)
class LlmSettings:
    """The selected LLM and generation controls, excluding API keys."""

    backend: Backend = "local"
    provider_key: str = "openrouter"
    model_id: str = DEFAULT_LOCAL_MODEL_ID
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.2
    top_p: float = 0.95

    def __post_init__(self) -> None:
        if self.backend not in ("local", "remote"):
            raise ValueError("backend must be either 'local' or 'remote'")
        get_provider(self.provider_key)
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or self.temperature < 0
        ):
            raise ValueError("temperature must be non-negative")
        if (
            isinstance(self.top_p, bool)
            or not isinstance(self.top_p, (int, float))
            or not 0 < self.top_p <= 1
        ):
            raise ValueError("top_p must be in the range (0, 1]")

    def with_backend(self, backend: Backend) -> LlmSettings:
        return replace(self, backend=backend)

    def with_provider(self, provider_key: str) -> LlmSettings:
        return replace(self, provider_key=provider_key)

    def with_model(self, model_id: str) -> LlmSettings:
        return replace(self, model_id=model_id)

    def with_generation(
        self,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> LlmSettings:
        return replace(
            self,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )


DEFAULT_SETTINGS = LlmSettings()
_SETTINGS_FIELDS = frozenset(LlmSettings.__dataclass_fields__)


def default_settings_path() -> Path:
    """Resolve settings at call time so tests and XDG users stay isolated."""
    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config_home) if config_home else Path.home() / ".config"
    return base / "soca" / "llm.json"


def load_settings(path: Path | None = None) -> LlmSettings:
    """Load settings or return the safe local default when none were saved."""
    path = path or default_settings_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return DEFAULT_SETTINGS
    except OSError as exc:
        raise ValueError(f"Không thể đọc LLM settings tại {path}: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM settings tại {path} không phải JSON hợp lệ.") from exc
    return _settings_from_payload(payload, path)


def save_settings(settings: LlmSettings, path: Path | None = None) -> None:
    """Atomically persist non-secret settings with owner-only permissions."""
    path = path or default_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = json.dumps(asdict(settings), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
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
        raise ValueError(f"Không thể lưu LLM settings tại {path}: {exc}") from exc
    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def _settings_from_payload(payload: object, path: Path) -> LlmSettings:
    if not isinstance(payload, Mapping):
        raise ValueError(f"LLM settings tại {path} phải là một JSON object.")
    if set(payload) != _SETTINGS_FIELDS:
        expected = ", ".join(sorted(_SETTINGS_FIELDS))
        raise ValueError(f"LLM settings tại {path} phải có đúng các trường: {expected}.")
    # Settings files created before the UI exposed generation controls contain
    # the old implicit 160-token default. Migrate that implicit value so a
    # long grounded answer or a reasoning model cannot consume the entire
    # budget before its answer.
    normalized = dict(payload)
    if normalized.get("max_tokens") == _LEGACY_DEFAULT_MAX_TOKENS:
        normalized["max_tokens"] = DEFAULT_MAX_TOKENS
    try:
        return LlmSettings(**normalized)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"LLM settings tại {path} không hợp lệ: {exc}") from exc


__all__ = [
    "DEFAULT_CONFIG_DIR",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_SETTINGS",
    "DEFAULT_SETTINGS_PATH",
    "default_settings_path",
    "LlmSettings",
    "load_settings",
    "save_settings",
]
