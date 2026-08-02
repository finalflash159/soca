from __future__ import annotations

from typing import Any

import numpy as np

from .result import ASRResult
from .whisper_onnx import VietnameseASR


class PhoWhisperVoiceBackend:
    def __init__(self, model_key: str) -> None:
        self._backend = VietnameseASR(model_key=model_key)
        self.model_key = model_key
        self.supports_avg_logprob = self._backend.supports_avg_logprob

    def transcribe(
        self,
        audio: np.ndarray,
        max_new_tokens: int = 128,
        *,
        context: str,
    ) -> ASRResult:
        if context:
            raise ValueError("PhoWhisper does not accept ASR context")
        return self._backend.transcribe(audio, max_new_tokens=max_new_tokens)

    def runtime_metadata(self, max_new_tokens: int = 128) -> dict[str, Any]:
        return self._backend.runtime_metadata(max_new_tokens=max_new_tokens)

    def close(self) -> None:
        return None
