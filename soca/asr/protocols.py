"""Typed ASR backend contracts."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from .result import ASRResult


@runtime_checkable
class CalibratableASR(Protocol):
    """Minimum surface needed to calibrate the confidence guard for a backend."""

    model_key: str

    def transcribe(
        self,
        audio: np.ndarray,
        max_new_tokens: int = 128,
        *,
        context: str | None = None,
    ) -> ASRResult:
        """Accepts float32 mono 16kHz 1-D audio. `avg_logprob` must be a real
        number from the model, not a placeholder."""
        ...

    def runtime_metadata(self, max_new_tokens: int = 128) -> dict[str, Any]: ...


@runtime_checkable
class VoiceASRBackend(Protocol):
    model_key: str
    supports_avg_logprob: bool

    def transcribe(
        self,
        audio: np.ndarray,
        max_new_tokens: int = 128,
        *,
        context: str,
    ) -> ASRResult: ...

    def runtime_metadata(self, max_new_tokens: int = 128) -> dict[str, Any]: ...

    def close(self) -> None: ...
