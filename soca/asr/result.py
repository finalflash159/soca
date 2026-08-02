from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ASRResult:
    text: str
    latency_ms: float
    audio_duration_ms: float
    rtf: float
    avg_logprob: float = 0.0
    avg_logprob_reliable: bool = True
    alternatives: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)
