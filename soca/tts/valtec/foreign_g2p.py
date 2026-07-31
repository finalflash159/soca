from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class ForeignG2P(Protocol):
    """Transcribe a single OOV token into the eng_to_ipa IPA dialect, or None.

    Contract:
      - Input: one alphabetic token, non-empty, mixed case.
      - Output: an IPA string using only characters in TRAINED_ENGLISH_IPA,
        or None when the token cannot be transcribed or falls outside the
        trained inventory.
      - Offline, deterministic, never raises.
    """

    def to_ipa(self, token: str) -> str | None: ...


class ChainedForeignG2P:
    """Try each backend in order and return the first transcription.

    Order encodes precision: a curated lexicon entry is known-correct, whereas
    a statistical prediction is a guess, so the lexicon must be consulted
    first. Returning None from every backend leaves the caller on its existing
    letter-spelling fallback.
    """

    def __init__(self, backends: Sequence[ForeignG2P]) -> None:
        self._backends = tuple(backends)

    def to_ipa(self, token: str) -> str | None:
        for backend in self._backends:
            ipa = backend.to_ipa(token)
            if ipa:
                return ipa
        return None
