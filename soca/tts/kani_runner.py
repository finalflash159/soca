from __future__ import annotations

import time
from typing import Any

from .audio_utils import build_result, empty_result
from .base import TTSResult
from .errors import TTSRuntimeUnavailableError
from .registry import TTSModelConfig


class KaniTTSRunner:
    ENGINE_NAME = "kani-tts-2"

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
            from kani_tts import KaniTTS
        except ImportError as exc:
            raise TTSRuntimeUnavailableError(
                "Kani TTS requires kani-tts-2. This currently conflicts with SoCa's "
                "main Hugging Face stack; test it in a separate env or install manually with: "
                "uv pip install kani-tts-2"
            ) from exc

        if not self.config.hf_repo:
            raise RuntimeError(f"Missing Kani model repo for {self.config.key}")

        try:
            self._model = KaniTTS(self.config.hf_repo, show_info=False)
        except Exception as exc:
            raise TTSRuntimeUnavailableError(
                f"Kani runtime could not load {self.config.key}: {exc}"
            ) from exc

    def list_voices(self) -> list[str]:
        self._ensure_loaded()
        status = getattr(self._model, "status", "")
        tags = list(getattr(self._model, "language_tags_list", []) or [])
        if status == "available_language_tags" and tags:
            return [str(tag) for tag in tags]
        return [self.config.default_voice]

    def synthesize(self, text: str, voice: str | None = None) -> TTSResult:
        text_clean = text.strip()
        selected_voice = voice or self.voice
        if not text_clean:
            return empty_result(text_clean, selected_voice, self.ENGINE_NAME, 22050)

        self._ensure_loaded()
        voices = self.list_voices()
        if selected_voice not in voices:
            raise ValueError(f"Unknown Kani voice/tag {selected_voice!r}. Available voices: {voices}")

        language_tag = None if selected_voice == self.config.default_voice else selected_voice
        t0 = time.perf_counter()
        audio, _ = self._model.generate(text_clean, language_tag=language_tag)
        sample_rate = int(getattr(self._model, "sample_rate", 22050))
        return build_result(
            text=text_clean,
            audio=audio,
            sample_rate=sample_rate,
            started_at=t0,
            voice=selected_voice,
            engine=self.ENGINE_NAME,
        )
