# soca/tts/valtec/onnx_runner.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from soca.tts.base import TTSResult
from soca.tts.config import VALTEC_TTS_CONFIG

from .artifacts import ValtecOnnxArtifacts, resolve_valtec_onnx_artifacts
from .frontend import ValtecFrontend


class ValtecOnnxTTS:
    ENGINE_NAME = "valtec-onnx"

    def __init__(
        self,
        *,
        voice: str | None = None,
        providers: list[str] | None = None,
        artifact_root: Path,
        artifact_variant: str | None = None,
        allow_reference: bool = False,
        frontend: ValtecFrontend,
        seed: int | None = None,
        max_audio_seconds: float = 60.0,
        noise_scale: float | None = None,
        length_scale: float | None = None,
    ) -> None:
        self.voice = voice or VALTEC_TTS_CONFIG.default_voice
        self.artifact_root = artifact_root
        self.artifact_variant = artifact_variant
        self.allow_reference = allow_reference
        self.providers = providers or ["CPUExecutionProvider"]
        self.frontend = frontend
        self._rng = np.random.default_rng(seed)
        if max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be positive")
        if noise_scale is not None and noise_scale < 0:
            raise ValueError("noise_scale must not be negative")
        if length_scale is not None and length_scale <= 0:
            raise ValueError("length_scale must be positive")
        self._max_audio_seconds = max_audio_seconds
        self._noise_scale_override = noise_scale
        self._length_scale_override = length_scale

        self._sessions: dict[str, Any] = {}
        self._sample_rate = 24000
        self._hop_length = 256
        self._noise_scale = 0.667
        self._length_scale = 1.0
        self._speaker_map: dict[str, int] = {}
        self._artifacts: ValtecOnnxArtifacts | None = None
        self._last_frontend_metadata: dict[str, Any] = {}
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._sessions:
            return

        artifacts = resolve_valtec_onnx_artifacts(
            self.artifact_root,
            variant=self.artifact_variant,
            allow_reference=self.allow_reference,
        )
        self._artifacts = artifacts
        self._sample_rate = artifacts.sample_rate
        self._hop_length = artifacts.hop_length
        self._noise_scale = (
            artifacts.noise_scale
            if self._noise_scale_override is None
            else self._noise_scale_override
        )
        self._length_scale = (
            artifacts.length_scale
            if self._length_scale_override is None
            else self._length_scale_override
        )
        self._noise_scale = artifacts.noise_scale
        self._length_scale = artifacts.length_scale
        self._speaker_map = artifacts.voice_map

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._sessions = {
            "text_encoder": ort.InferenceSession(
                str(artifacts.text_encoder), sess_options=opts, providers=self.providers
            ),
            "duration_predictor": ort.InferenceSession(
                str(artifacts.duration_predictor), sess_options=opts, providers=self.providers
            ),
            "flow": ort.InferenceSession(
                str(artifacts.flow), sess_options=opts, providers=self.providers
            ),
            "decoder": ort.InferenceSession(
                str(artifacts.decoder), sess_options=opts, providers=self.providers
            ),
        }

    def list_voices(self) -> list[str]:
        self._ensure_loaded()
        return list(self._speaker_map)

    @property
    def frontend_metadata(self) -> dict[str, Any]:
        return dict(self._last_frontend_metadata)

    def _speaker_id(self, selected_voice: str) -> int:
        if selected_voice not in self._speaker_map:
            raise ValueError(
                f"Unknown Valtec voice {selected_voice!r}. Available voices: {self.list_voices()}"
            )
        return int(self._speaker_map[selected_voice])

    def synthesize(self, text: str, voice: str | None = None) -> TTSResult:
        text_clean = text.strip()
        selected_voice = voice or self.voice
        if not text_clean:
            return TTSResult(
                text=text_clean,
                audio=np.array([], dtype=np.float32),
                sample_rate=self._sample_rate,
                latency_ms=0.0,
                audio_duration_ms=0.0,
                rtf=0.0,
                voice=selected_voice,
                engine=self.ENGINE_NAME,
            )

        started_at = time.perf_counter()
        self._ensure_loaded()
        model_inputs = self.frontend.prepare(text_clean)
        self._last_frontend_metadata = {
            "backend": model_inputs.backend,
            "unknown_phoneme_count": model_inputs.unknown_phoneme_count,
        }
        seq_len = len(model_inputs.phone_ids)
        sid = np.array([self._speaker_id(selected_voice)], dtype=np.int64)

        inputs = {
            "phone_ids": np.asarray([model_inputs.phone_ids], dtype=np.int64),
            "phone_lengths": np.asarray([seq_len], dtype=np.int64),
            "tone_ids": np.asarray([model_inputs.tone_ids], dtype=np.int64),
            "language_ids": np.asarray([model_inputs.language_ids], dtype=np.int64),
            "bert": np.zeros((1, 1024, seq_len), dtype=np.float32),
            "ja_bert": np.zeros((1, 768, seq_len), dtype=np.float32),
            "speaker_id": sid,
        }

        x_encoded, m_p, logs_p, x_mask, g = self._sessions["text_encoder"].run(None, inputs)
        (logw,) = self._sessions["duration_predictor"].run(
            None,
            {"x": x_encoded, "x_mask": x_mask, "g": g},
        )

        z_p, y_mask = self._expand_latents(
            m_p,
            logs_p,
            logw,
            x_mask,
            length_scale=self._length_scale,
            noise_scale=self._noise_scale,
        )
        (z,) = self._sessions["flow"].run(None, {"z_p": z_p, "y_mask": y_mask, "g": g})
        (audio,) = self._sessions["decoder"].run(None, {"z": z, "g": g})

        audio_np = np.ascontiguousarray(np.asarray(audio).squeeze(), dtype=np.float32)
        latency_ms = (time.perf_counter() - started_at) * 1000
        audio_duration_ms = len(audio_np) / self._sample_rate * 1000
        return TTSResult(
            text=text_clean,
            audio=audio_np,
            sample_rate=self._sample_rate,
            latency_ms=latency_ms,
            audio_duration_ms=audio_duration_ms,
            rtf=latency_ms / audio_duration_ms if audio_duration_ms > 0 else 0.0,
            voice=selected_voice,
            engine=self.ENGINE_NAME,
        )

    def _expand_latents(
        self,
        m_p: np.ndarray,
        logs_p: np.ndarray,
        logw: np.ndarray,
        x_mask: np.ndarray,
        *,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
    ) -> tuple[np.ndarray, np.ndarray]:
        durations = np.ceil(np.exp(logw) * x_mask * length_scale).astype(np.int64).reshape(-1)
        durations = np.maximum(durations, 0)
        total_frames = int(durations.sum())
        if total_frames <= 0:
            total_frames = 1
        max_frames = int(self._sample_rate * self._max_audio_seconds / self._hop_length)
        if total_frames > max_frames:
            raise ValueError(
                f"Valtec predicted {total_frames} frames; limit is {max_frames}. "
                "Split the text into smaller chunks or inspect duration predictor output."
            )

        # m_p/logs_p: (1, channels, seq_len). np.repeat replaces the nested Python loop
        # in upstream edge inference and matters for short assistant chunks.
        expanded_mp = np.repeat(m_p[0], durations, axis=1)
        expanded_logs = np.repeat(logs_p[0], durations, axis=1)
        if expanded_mp.shape[1] == 0:
            expanded_mp = np.zeros((m_p.shape[1], 1), dtype=np.float32)
            expanded_logs = np.zeros((logs_p.shape[1], 1), dtype=np.float32)

        noise = self._rng.standard_normal(expanded_mp.shape, dtype=np.float32) * noise_scale
        z_p = expanded_mp + np.exp(expanded_logs) * noise
        return z_p[None].astype(np.float32, copy=False), np.ones((1, 1, z_p.shape[1]), dtype=np.float32)
