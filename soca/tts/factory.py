from __future__ import annotations

from .base import TTSEngine
from .external_command_runner import ExternalCommandTTS
from .f5_runner import F5VietnameseTTS
from .kani_runner import KaniTTSRunner
from .mms_runner import MMSTTS
from .omnivoice_runner import OmniVoiceTTS
from .piper_runner import PiperTTS
from .registry import get_tts_model_config
from .valtec_runner import VietnameseTTS
from .vieneu_runner import VieneuTTS
from .viettts_server_runner import VietTTSServerTTS


def create_tts_engine(model_key: str, *, voice: str | None = None, lazy: bool = True) -> TTSEngine:
    config = get_tts_model_config(model_key)
    selected_voice = voice or config.default_voice

    if config.runner == "valtec":
        return VietnameseTTS(voice=selected_voice, lazy=lazy)
    if config.runner == "mms_transformers":
        return MMSTTS(lazy=lazy)
    if config.runner == "piper":
        return PiperTTS(config=config, lazy=lazy)
    if config.runner == "vieneu":
        return VieneuTTS(config=config, voice=selected_voice, lazy=lazy)
    if config.runner == "kani":
        return KaniTTSRunner(config=config, voice=selected_voice, lazy=lazy)
    if config.runner == "f5":
        return F5VietnameseTTS(config=config, voice=selected_voice, lazy=lazy)
    if config.runner == "omnivoice":
        return OmniVoiceTTS(config=config, voice=selected_voice, lazy=lazy)
    if config.runner == "viettts_server":
        return VietTTSServerTTS(config=config, voice=selected_voice, lazy=lazy)
    if config.runner == "external_command":
        return ExternalCommandTTS(config=config, voice=selected_voice, lazy=lazy)

    raise NotImplementedError(f"Unsupported TTS runner {config.runner!r} for {model_key}.")
