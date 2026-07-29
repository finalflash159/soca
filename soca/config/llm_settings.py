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
ReasoningParameter = Literal["reasoning", "reasoning_effort"]
DEFAULT_CONFIG_DIR = Path.home() / ".config" / "soca"
DEFAULT_SETTINGS_PATH = DEFAULT_CONFIG_DIR / "llm.json"
DEFAULT_LOCAL_MODEL_ID = "arcee_vylinh_3b_q4_k_m"
DEFAULT_MAX_TOKENS = 4_096
MIN_MAX_TOKENS = 2_048
MAX_MAX_TOKENS = 500_000
_LEGACY_DEFAULT_MAX_TOKENS = 160


@dataclass(frozen=True)
class LlmSettings:
    """The selected LLM and generation controls, excluding API keys."""

    backend: Backend = "local"
    provider_key: str = "openrouter"
    model_id: str = DEFAULT_LOCAL_MODEL_ID
    max_tokens: int = DEFAULT_MAX_TOKENS
    reasoning_enabled: bool = False
    temperature: float = 0.2
    top_p: float = 0.95
    model_context_window: int | None = None
    model_max_output_tokens: int | None = None
    model_reasoning_supported: bool | None = None
    model_reasoning_mandatory: bool = False
    model_reasoning_parameter: ReasoningParameter | None = None

    def __post_init__(self) -> None:
        if self.backend not in ("local", "remote"):
            raise ValueError("backend must be either 'local' or 'remote'")
        get_provider(self.provider_key)
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or not MIN_MAX_TOKENS <= self.max_tokens <= MAX_MAX_TOKENS
        ):
            raise ValueError(
                f"max_tokens must be an integer from {MIN_MAX_TOKENS} to {MAX_MAX_TOKENS}"
            )
        if not isinstance(self.reasoning_enabled, bool):
            raise ValueError("reasoning_enabled must be a boolean")
        if self.model_max_output_tokens is not None and (
            isinstance(self.model_max_output_tokens, bool)
            or not isinstance(self.model_max_output_tokens, int)
            or self.model_max_output_tokens <= 0
        ):
            raise ValueError("model_max_output_tokens must be a positive integer or null")
        if self.model_context_window is not None and (
            isinstance(self.model_context_window, bool)
            or not isinstance(self.model_context_window, int)
            or self.model_context_window <= 0
        ):
            raise ValueError("model_context_window must be a positive integer or null")
        if self.model_reasoning_supported is not None and not isinstance(
            self.model_reasoning_supported, bool
        ):
            raise ValueError("model_reasoning_supported must be a boolean or null")
        if not isinstance(self.model_reasoning_mandatory, bool):
            raise ValueError("model_reasoning_mandatory must be a boolean")
        if self.model_reasoning_parameter not in ("reasoning", "reasoning_effort", None):
            raise ValueError("model_reasoning_parameter is invalid")
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
        reasoning_enabled: bool | None = None,
        temperature: float,
        top_p: float,
    ) -> LlmSettings:
        return replace(
            self,
            max_tokens=max_tokens,
            reasoning_enabled=(
                self.reasoning_enabled if reasoning_enabled is None else reasoning_enabled
            ),
            temperature=temperature,
            top_p=top_p,
        )

    def with_model_capabilities(
        self,
        *,
        context_window: int | None = None,
        max_output_tokens: int | None,
        reasoning_supported: bool | None,
        reasoning_mandatory: bool,
        reasoning_parameter: ReasoningParameter | None,
    ) -> LlmSettings:
        return replace(
            self,
            model_context_window=context_window,
            model_max_output_tokens=max_output_tokens,
            model_reasoning_supported=reasoning_supported,
            model_reasoning_mandatory=reasoning_mandatory,
            model_reasoning_parameter=reasoning_parameter,
        )

    @property
    def effective_max_tokens(self) -> int:
        if self.model_max_output_tokens is None:
            return self.max_tokens
        return min(self.max_tokens, self.model_max_output_tokens)

    @property
    def effective_reasoning_enabled(self) -> bool | None:
        if self.model_reasoning_mandatory:
            return True
        if self.model_reasoning_supported is True:
            return self.reasoning_enabled
        return None


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
    unknown = set(payload) - _SETTINGS_FIELDS
    if unknown:
        fields = ", ".join(sorted(unknown))
        raise ValueError(f"LLM settings tại {path} có trường không hỗ trợ: {fields}.")
    # Older releases accepted every positive integer.  Keep the selected
    # provider/model on upgrade by making legacy values satisfy the current UI
    # contract instead of rejecting the whole persisted settings file.
    #
    # 160 was the old implicit default, so upgrade it to the new default.  Any
    # other explicit positive value below the new minimum is clamped to that
    # minimum; malformed, zero, negative, and boolean values still fail
    # validation normally.
    normalized = dict(payload)
    if normalized.get("max_tokens") == _LEGACY_DEFAULT_MAX_TOKENS:
        normalized["max_tokens"] = DEFAULT_MAX_TOKENS
    elif (
        isinstance(normalized.get("max_tokens"), int)
        and not isinstance(normalized.get("max_tokens"), bool)
        and 0 < normalized["max_tokens"] < MIN_MAX_TOKENS
    ):
        normalized["max_tokens"] = MIN_MAX_TOKENS
    for field_name in _SETTINGS_FIELDS:
        if field_name not in normalized:
            normalized[field_name] = getattr(DEFAULT_SETTINGS, field_name)
    try:
        return LlmSettings(**normalized)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"LLM settings tại {path} không hợp lệ: {exc}") from exc


__all__ = [
    "DEFAULT_CONFIG_DIR",
    "MAX_MAX_TOKENS",
    "MIN_MAX_TOKENS",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_SETTINGS",
    "DEFAULT_SETTINGS_PATH",
    "default_settings_path",
    "LlmSettings",
    "load_settings",
    "save_settings",
]
