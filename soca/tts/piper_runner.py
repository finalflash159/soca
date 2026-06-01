from __future__ import annotations

import io
import time
import wave
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download

from .base import TTSResult
from .registry import TTSModelConfig


class PiperTTS:
    ENGINE_NAME = "piper-tts"

    def __init__(self, config: TTSModelConfig, lazy: bool = True) -> None:
        self.config = config
        self._voice: Any | None = None
        if not lazy:
            self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._voice is not None:
            return

        try:
            from piper import PiperVoice
        except ImportError as exc:
            raise RuntimeError("Piper requires: uv sync --extra tts-piper") from exc

        if not self.config.hf_repo or not self.config.hf_model_file or not self.config.hf_config_file:
            raise RuntimeError(f"Missing Piper artifact metadata for {self.config.key}")

        model_path = hf_hub_download(
            repo_id=self.config.hf_repo,
            filename=self.config.hf_model_file,
            local_dir=self.config.local_dir,
        )
        config_path = hf_hub_download(
            repo_id=self.config.hf_repo,
            filename=self.config.hf_config_file,
            local_dir=self.config.local_dir,
        )
        self._voice = PiperVoice.load(model_path, config_path=config_path)

    def list_voices(self) -> list[str]:
        self._ensure_loaded()
        speaker_map = getattr(self._voice.config, "speaker_id_map", None) or {}
        return list(speaker_map) or ["default"]

    def _speaker_id(self, voice: str | None) -> int | None:
        selected = voice or self.config.default_voice
        speaker_map = getattr(self._voice.config, "speaker_id_map", None) or {}
        if not speaker_map:
            return None
        if selected not in speaker_map:
            raise ValueError(f"Unknown Piper voice {selected!r}. Available voices: {sorted(speaker_map)}")
        return int(speaker_map[selected])

    @staticmethod
    def _wav_bytes_to_float32_mono(payload: bytes) -> tuple[np.ndarray, int]:
        with wave.open(io.BytesIO(payload), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())

        if width != 2:
            raise RuntimeError(f"Unsupported Piper sample width: {width}")

        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        return np.ascontiguousarray(audio, dtype=np.float32), sample_rate

    def synthesize(self, text: str, voice: str | None = None) -> TTSResult:
        text_clean = text.strip()
        selected_voice = voice or self.config.default_voice

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

        from piper import SynthesisConfig

        syn_config = SynthesisConfig(speaker_id=self._speaker_id(selected_voice))
        buffer = io.BytesIO()

        t0 = time.perf_counter()
        with wave.open(buffer, "wb") as wav_file:
            self._voice.synthesize_wav(text_clean, wav_file, syn_config=syn_config)
        latency_ms = (time.perf_counter() - t0) * 1000

        audio, sample_rate = self._wav_bytes_to_float32_mono(buffer.getvalue())
        duration_ms = (len(audio) / sample_rate) * 1000 if sample_rate > 0 else 0.0

        return TTSResult(
            text=text_clean,
            audio=audio,
            sample_rate=sample_rate,
            latency_ms=latency_ms,
            audio_duration_ms=duration_ms,
            rtf=latency_ms / duration_ms if duration_ms > 0 else 0.0,
            voice=selected_voice,
            engine=self.ENGINE_NAME,
        )
