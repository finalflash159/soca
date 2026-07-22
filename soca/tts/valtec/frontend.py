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

    def __post_init__(self) -> None:
        lengths = {len(self.phone_ids), len(self.tone_ids), len(self.language_ids)}
        if lengths == {0}:
            raise ValueError("Valtec frontend returned an empty sequence")
        if len(lengths) != 1:
            raise ValueError("Valtec phone/tone/language sequences must have equal length")
        if self.unknown_phoneme_count < 0:
            raise ValueError("unknown_phoneme_count must not be negative")


class ValtecFrontend(Protocol):
    def prepare(self, text: str) -> ValtecModelInputs:
        """Normalize Vietnamese text and produce aligned model IDs."""
        raise NotImplementedError
