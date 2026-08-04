from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ValtecModelInputs:
    phone_ids: tuple[int, ...]
    tone_ids: tuple[int, ...]
    language_ids: tuple[int, ...]
    backend: str
    unknown_phoneme_count: int = 0
    # Phones produced by English IPA or letter spelling; the runtime stretches
    # exactly these phones so foreign words stay intelligible without slowing
    # the surrounding Vietnamese.
    foreign_phone_count: int = 0
    # 0/1 per position in phone_ids (empty tuple = no foreign phones).
    foreign_flags: tuple[int, ...] = ()
    # Per-phone duration multipliers for speech-only technical-term pacing.
    # Empty means every phone uses the model/application scale.
    technical_duration_scales: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        lengths = {len(self.phone_ids), len(self.tone_ids), len(self.language_ids)}
        if lengths == {0}:
            raise ValueError("Valtec frontend returned an empty sequence")
        if len(lengths) != 1:
            raise ValueError("Valtec phone/tone/language sequences must have equal length")
        if self.unknown_phoneme_count < 0:
            raise ValueError("unknown_phoneme_count must not be negative")
        if self.foreign_phone_count < 0:
            raise ValueError("foreign_phone_count must not be negative")
        if self.foreign_flags and len(self.foreign_flags) != len(self.phone_ids):
            raise ValueError("foreign_flags must align with phone_ids")
        if self.technical_duration_scales:
            if len(self.technical_duration_scales) != len(self.phone_ids):
                raise ValueError("technical_duration_scales must align with phone_ids")
            if any(scale < 1.0 for scale in self.technical_duration_scales):
                raise ValueError("technical duration scales must be at least 1.0")


class ValtecFrontend(Protocol):
    def prepare(self, text: str) -> ValtecModelInputs:
        """Normalize Vietnamese text and produce aligned model IDs."""
        raise NotImplementedError
