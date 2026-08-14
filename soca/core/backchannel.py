"""Typed semantic gate between sustained speech and a playback interruption."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

import numpy as np


class BargeInIntent(StrEnum):
    BACKCHANNEL = "backchannel"
    INTERRUPTION = "interruption"


class BackchannelClassificationError(RuntimeError):
    def __init__(self, code: str) -> None:
        normalized = code.strip()
        if not normalized:
            raise ValueError("backchannel error code is required")
        super().__init__(normalized)
        self.code = normalized


@dataclass(frozen=True)
class BackchannelDecision:
    intent: BargeInIntent
    confidence: float
    model_id: str
    model_revision: str
    latency_ms: float
    provider_trace: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        numeric = (self.confidence, self.latency_ms)
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("backchannel decision metrics must be finite")
        if not 0.0 <= self.confidence <= 1.0 or self.latency_ms < 0.0:
            raise ValueError("invalid backchannel confidence or latency")
        if not self.model_id.strip() or not self.model_revision.strip():
            raise ValueError("backchannel model identity is required")
        object.__setattr__(self, "provider_trace", dict(self.provider_trace))

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "latency_ms": self.latency_ms,
            "provider_trace": dict(self.provider_trace),
        }


class BackchannelClassifier(Protocol):
    """Classify one bounded, post-AEC audio window without a text heuristic."""

    def classify(self, audio: np.ndarray, sample_rate: int) -> BackchannelDecision: ...


def classify_barge_in_window(
    classifier: BackchannelClassifier,
    audio: np.ndarray,
    sample_rate: int,
) -> BackchannelDecision:
    array = np.asarray(audio, dtype=np.float32).reshape(-1)
    if array.size == 0 or sample_rate <= 0:
        raise BackchannelClassificationError("invalid_audio_window")
    try:
        decision = classifier.classify(array, sample_rate)
    except BackchannelClassificationError:
        raise
    except Exception as exc:  # noqa: BLE001 - model boundary must fail closed
        raise BackchannelClassificationError("classifier_unavailable") from exc
    if not isinstance(decision, BackchannelDecision):
        raise BackchannelClassificationError("invalid_classifier_output")
    return decision


__all__ = [
    "BackchannelClassificationError",
    "BackchannelClassifier",
    "BackchannelDecision",
    "BargeInIntent",
    "classify_barge_in_window",
]
