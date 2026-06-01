from __future__ import annotations

import time
from typing import Any

import numpy as np

from .audio_utils import build_result, empty_result
from .base import TTSResult
from .errors import TTSRuntimeUnavailableError
from .registry import TTSModelConfig


class VieneuTTS:
    ENGINE_NAME = "vieneu"
    SAMPLE_RATE = 24000

    def __init__(
        self,
        config: TTSModelConfig,
        *,
        voice: str | None = None,
    ) -> None:
        self.config = config
        self.voice = voice or config.default_voice
        self._model: Any | None = None
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            from vieneu import Vieneu
        except ImportError as exc:
            raise TTSRuntimeUnavailableError(
                "VieNeu TTS requires the vieneu SDK. Install with: "
                "uv sync --extra tts-vieneu"
            ) from exc

        kwargs: dict[str, Any] = {}
        if self.config.sdk_mode:
            kwargs["mode"] = self.config.sdk_mode
        if self.config.hf_repo:
            kwargs["backbone_repo"] = self.config.hf_repo
        if self.config.sdk_mode == "turbo" and self.config.hf_model_file:
            kwargs["backbone_filename"] = self.config.hf_model_file
        if self.config.sdk_mode == "standard" and self.config.hf_model_file:
            kwargs["gguf_filename"] = self.config.hf_model_file

        try:
            self._model = Vieneu(**kwargs)
        except Exception as exc:
            raise TTSRuntimeUnavailableError(
                f"VieNeu runtime could not load {self.config.key}: {exc}"
            ) from exc

    def list_voices(self) -> list[str]:
        self._ensure_loaded()
        voices = self._model.list_preset_voices()
        return [str(voice_id) for _, voice_id in voices] or [self.config.default_voice]

    def _voice_data(self, selected_voice: str) -> tuple[str, Any | None]:
        voices = self.list_voices()
        if selected_voice not in voices:
            if selected_voice == self.config.default_voice and voices:
                selected_voice = voices[0]
            else:
                raise ValueError(f"Unknown VieNeu voice {selected_voice!r}. Available voices: {voices}")
        return selected_voice, self._model.get_preset_voice(selected_voice)

    def synthesize(self, text: str, voice: str | None = None) -> TTSResult:
        text_clean = text.strip()
        selected_voice = voice or self.voice
        if not text_clean:
            return empty_result(text_clean, selected_voice, self.ENGINE_NAME, self.SAMPLE_RATE)

        self._ensure_loaded()
        selected_voice, voice_data = self._voice_data(selected_voice)

        t0 = time.perf_counter()
        audio = self._model.infer(text=text_clean, voice=voice_data, show_progress=False)
        sample_rate = int(getattr(self._model, "sample_rate", self.SAMPLE_RATE))
        return build_result(
            text=text_clean,
            audio=np.asarray(audio),
            sample_rate=sample_rate,
            started_at=t0,
            voice=selected_voice,
            engine=self.ENGINE_NAME,
        )
