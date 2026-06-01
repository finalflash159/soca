from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class TTSResult:
    text: str
    audio: np.ndarray
    sample_rate: int
    latency_ms: float
    audio_duration_ms: float
    rtf: float
    voice: str
    engine: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["audio"] = f"<{len(self.audio)} samples>"
        return data


class TTSEngine(Protocol):
    def synthesize(self, text: str, voice: str | None = None) -> TTSResult: ...

    def list_voices(self) -> list[str]: ...
