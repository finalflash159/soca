from __future__ import annotations

import time
from typing import Any

import numpy as np

from .audio_utils import to_float32_mono
from .base import TTSResult


class MMSTTS:
    DEFAULT_MODEL_ID = "facebook/mms-tts-vie"
    ENGINE_NAME = "mms-tts-vie"
    DEFAULT_VOICE = "default"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, lazy: bool = True) -> None:
        self.model_id = model_id
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        if not lazy:
            self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoTokenizer, VitsModel
        except ImportError as exc:
            raise RuntimeError(
                "MMS TTS requires torch and transformers. Run: uv sync --extra tts"
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = VitsModel.from_pretrained(self.model_id)
        self._model.eval()

    def list_voices(self) -> list[str]:
        return [self.DEFAULT_VOICE]

    def synthesize(self, text: str, voice: str | None = None) -> TTSResult:
        text_clean = text.strip()
        selected_voice = voice or self.DEFAULT_VOICE
        if selected_voice != self.DEFAULT_VOICE:
            raise ValueError("MMS Vietnamese TTS exposes a single default voice.")

        if not text_clean:
            return TTSResult(
                text="",
                audio=np.array([], dtype=np.float32),
                sample_rate=16000,
                latency_ms=0.0,
                audio_duration_ms=0.0,
                rtf=0.0,
                voice=selected_voice,
                engine=self.ENGINE_NAME,
            )

        self._ensure_loaded()
        t0 = time.perf_counter()
        inputs = self._tokenizer(text_clean, return_tensors="pt")
        with self._torch.no_grad():
            waveform = self._model(**inputs).waveform
        latency_ms = (time.perf_counter() - t0) * 1000

        audio_np = to_float32_mono(waveform.detach().cpu().numpy())
        sample_rate = int(getattr(self._model.config, "sampling_rate", 16000))
        audio_duration_ms = (len(audio_np) / sample_rate) * 1000 if sample_rate > 0 else 0.0
        rtf = latency_ms / audio_duration_ms if audio_duration_ms > 0 else 0.0

        return TTSResult(
            text=text_clean,
            audio=audio_np,
            sample_rate=sample_rate,
            latency_ms=latency_ms,
            audio_duration_ms=audio_duration_ms,
            rtf=rtf,
            voice=selected_voice,
            engine=self.ENGINE_NAME,
        )
