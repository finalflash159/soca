"""Contract between tooling (calibrator, eval scripts) and any ASR backend.

Making this explicit is what lets `local/calibrate_asr_confidence.py`
calibrate a non-Whisper backend without depending on `VietnameseASR`
directly.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from .whisper_onnx import ASRResult


@runtime_checkable
class CalibratableASR(Protocol):
    """Minimum surface needed to calibrate the confidence guard for a backend."""

    model_key: str

    def transcribe(self, audio: np.ndarray, max_new_tokens: int = 128) -> ASRResult:
        """Accepts float32 mono 16kHz 1-D audio. `avg_logprob` must be a real
        number from the model, not a placeholder."""
        ...

    def runtime_metadata(self, max_new_tokens: int = 128) -> dict[str, Any]:
        """Runtime identity recorded into the calibration file.

        Required, not decorative: a threshold is only valid for the exact
        runtime that produced it. Changing decoder/provider/max_new_tokens
        (or, for LLM-decoder backends, context) requires recalibrating.
        """
        ...
