"""Resolve the explicit Qwen worker runtime selected by the host."""

from __future__ import annotations

import os
from pathlib import Path

QWEN_RUNTIME_ROOT_ENV = "SOCA_QWEN_RUNTIME_ROOT"


def _absolute(path: Path, *, variable: str) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute() or ".." in expanded.parts:
        raise ValueError(f"{variable} must be an absolute path without traversal")
    return expanded.resolve()


def default_qwen_runtime_root() -> Path:
    """Return the selected immutable Qwen worker runtime root.

    Desktop supplies an explicit app-owned selection. Source workflows retain
    the checked-in runtime convention; neither path falls back to a different
    ASR implementation.
    """
    configured = os.environ.get(QWEN_RUNTIME_ROOT_ENV, "").strip()
    if configured:
        return _absolute(Path(configured), variable=QWEN_RUNTIME_ROOT_ENV)
    return Path(__file__).resolve().parents[2] / "runtime" / "qwen-asr"


def default_qwen_venv_python() -> Path:
    return default_qwen_runtime_root() / ".venv" / "bin" / "python"


__all__ = [
    "QWEN_RUNTIME_ROOT_ENV",
    "default_qwen_runtime_root",
    "default_qwen_venv_python",
]
