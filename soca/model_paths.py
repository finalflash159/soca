"""One explicit model-root contract for source and bundled runtimes.

The frozen sidecar never treats PyInstaller's temporary extraction directory as
durable model storage.  A host may set ``SOCA_MODEL_ROOT`` to an absolute
directory; otherwise an XDG data home is honoured, and a source checkout keeps
its repository-local ``models/`` convention for developer workflows.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MODEL_ROOT_ENV = "SOCA_MODEL_ROOT"


def _absolute(path: Path, *, variable: str) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        raise ValueError(f"{variable} must be absolute")
    return expanded.resolve()


def default_model_root() -> Path:
    configured = os.environ.get(MODEL_ROOT_ENV, "").strip()
    if configured:
        return _absolute(Path(configured), variable=MODEL_ROOT_ENV)

    data_home = os.environ.get("XDG_DATA_HOME", "").strip()
    if data_home:
        return _absolute(Path(data_home), variable="XDG_DATA_HOME") / "soca" / "models"

    if getattr(sys, "frozen", False):
        return Path.home() / ".local" / "share" / "soca" / "models"

    return Path(__file__).resolve().parents[1] / "models"


__all__ = ["MODEL_ROOT_ENV", "default_model_root"]
