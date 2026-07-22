from __future__ import annotations

from .base import TTSEngine
from .config import VALTEC_TTS_CONFIG
from .valtec_runner import VietnameseTTS


def create_tts_engine(*, voice: str | None = None) -> TTSEngine:
    return VietnameseTTS(voice=voice or VALTEC_TTS_CONFIG.default_voice)
