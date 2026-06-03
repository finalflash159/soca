from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TuiStageEvent:
    """One pipeline stage marker rendered on the stage rail."""

    stage: str
    status: str
    latency_ms: float | None = None
    detail: str = ""


__all__ = ["TuiStageEvent"]
