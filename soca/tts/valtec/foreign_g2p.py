from __future__ import annotations

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
