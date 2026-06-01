from __future__ import annotations


class TTSRuntimeUnavailableError(RuntimeError):
    """Raised when a registered TTS backend is valid but not installed or not running."""

