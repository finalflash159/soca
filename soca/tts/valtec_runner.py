from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import TTSResult


class VietnameseTTS:
    DEFAULT_SAMPLE_RATE = 24000
    ENGINE_NAME = "valtec-tts"
    ENV_SOURCE_DIR = "VALTEC_TTS_SOURCE_DIR"
    DEFAULT_SOURCE_DIR = Path(__file__).resolve().parents[2] / "external" / "valtec-tts"

    def __init__(
        self,
        voice: str = "NF",
        lazy: bool = True,
        source_dir: str | Path | None = None,
    ) -> None:
        self.voice = voice
        self._model: Any | None = None
        self.source_dir = self._resolve_source_dir(source_dir)
        if not lazy:
            self._ensure_loaded()

    @classmethod
    def _resolve_source_dir(cls, source_dir: str | Path | None) -> Path | None:
        candidate = source_dir or os.environ.get(cls.ENV_SOURCE_DIR)
        if candidate is None and cls.DEFAULT_SOURCE_DIR.exists():
            candidate = cls.DEFAULT_SOURCE_DIR
        if candidate is None:
            return None

        path = Path(candidate).expanduser().resolve()
        required_files = [path / "infer.py", path / "src"]
        if not all(p.exists() for p in required_files):
            raise FileNotFoundError(
                f"Invalid valtec source directory: {path}\n"
                "Expected it to contain infer.py and src/. Clone with:\n"
                "  git clone https://github.com/tronghieuit/valtec-tts.git external/valtec-tts"
            )
        return path

    def _add_source_dir_to_path(self) -> None:
        if self.source_dir is None:
            return
        source_path = str(self.source_dir)
        if source_path not in sys.path:
            sys.path.insert(0, source_path)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        self._add_source_dir_to_path()
        try:
            from valtec_tts import TTS
        except ImportError as exc:
            raise RuntimeError(
                "Could not import valtec_tts from the vendored source checkout.\n"
                f"Expected source directory: {self.source_dir or self.DEFAULT_SOURCE_DIR}\n"
                "Make sure external/valtec-tts exists and run: uv sync --extra tts"
            ) from exc
        try:
            self._model = TTS()
        except ImportError as exc:
            message = str(exc)
            if "No module named 'infer'" in message:
                raise RuntimeError(
                    "valtec_tts imported, but it could not find its top-level infer.py module. "
                    "This usually means the source checkout was not added to sys.path correctly.\n"
                    f"Expected source directory: {self.source_dir or self.DEFAULT_SOURCE_DIR}"
                ) from exc
            raise RuntimeError(
                "Valtec TTS failed while importing its runtime modules. "
                "This usually means a TTS optional dependency is missing.\n"
                "Run: uv sync --extra tts\n"
                f"Original error: {message}"
            ) from exc

    @staticmethod
    def _to_float32_mono(audio: Any) -> np.ndarray:
        arr = np.asarray(audio)

        if arr.ndim == 0:
            return np.array([], dtype=np.float32)

        arr = np.squeeze(arr)

        if arr.ndim == 2:
            if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1]:
                arr = arr.mean(axis=0)
            else:
                arr = arr.mean(axis=1)

        arr = arr.astype(np.float32, copy=False)

        if arr.size:
            peak = float(np.max(np.abs(arr)))
            if peak > 1.5:
                arr = arr / peak

        return np.ascontiguousarray(arr, dtype=np.float32)

    def list_voices(self) -> list[str]:
        self._ensure_loaded()
        voices = self._model.list_speakers()
        return [str(v) for v in voices]

    def synthesize(self, text: str, voice: str | None = None) -> TTSResult:
        text_clean = text.strip()
        selected_voice = voice or self.voice

        if not text_clean:
            return TTSResult(
                text="",
                audio=np.array([], dtype=np.float32),
                sample_rate=self.DEFAULT_SAMPLE_RATE,
                latency_ms=0.0,
                audio_duration_ms=0.0,
                rtf=0.0,
                voice=selected_voice,
                engine=self.ENGINE_NAME,
            )

        self._ensure_loaded()

        voices = self.list_voices()
        if selected_voice not in voices:
            raise ValueError(f"Unknown voice '{selected_voice!r}'. Available voices: {voices}")
        t0 = time.perf_counter()
        audio, sample_rate = self._model.synthesize(text_clean, speaker=selected_voice)
        latency_ms = (time.perf_counter() - t0) * 1000

        audio_np = self._to_float32_mono(audio)
        sample_rate = int(sample_rate)
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
