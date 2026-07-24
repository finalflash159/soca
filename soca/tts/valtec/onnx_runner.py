# soca/tts/valtec/onnx_runner.py
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from soca.tts.base import TTSResult
from soca.tts.config import VALTEC_TTS_CONFIG

from .artifacts import ValtecOnnxArtifacts, resolve_valtec_onnx_artifacts
from .frontend import ValtecFrontend
from .normalizer import split_sentences

# VITS checkpoints are trained on single sentences; synthesizing per sentence
# with a short rest keeps prosody stable on long inputs. Sentences beyond
# LONG_SENTENCE_CHARS additionally split at commas (packed back up to that
# limit so chunks stay mid-sized) to stop mid-sentence duration collapse
# observed on number-heavy clauses.
INTER_SENTENCE_SILENCE_SECONDS = 0.25
CLAUSE_SILENCE_SECONDS = 0.15
LONG_SENTENCE_CHARS = 100
MIN_CLAUSE_CHARS = 20
EDGE_FADE_SECONDS = 0.01
CLAUSE_BOUNDARY = re.compile(r"(?<=,)\s+")
# Chunks measured faster than this read as slurred; stretch them back down.
MAX_PHONES_PER_SECOND = 11.5
# English/spelled phones get extra time to stay intelligible; applied per
# phone so the surrounding Vietnamese keeps its natural pace.
FOREIGN_SLOWDOWN = 1.25
MAX_ADAPTIVE_SCALE = 1.4
# Chunk edges keep at most this much model-generated silence; the planned
# inter-chunk gaps then produce uniform pauses instead of 0.4-0.6s drifts.
EDGE_SILENCE_KEEP_SECONDS = 0.05
EDGE_SILENCE_THRESHOLD = 0.01
# Loudness matching bounds: chunks measured up to ~8 dB apart on one paragraph.
LOUDNESS_GAIN_RANGE = (0.6, 1.6)
PEAK_LIMIT = 0.98


def _trim_edge_silence(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Trim model-generated edge silence down to a small fixed padding."""
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= 0.0:
        return audio
    loud = np.where(np.abs(audio) > EDGE_SILENCE_THRESHOLD * peak)[0]
    if loud.size == 0:
        return audio
    keep = int(EDGE_SILENCE_KEEP_SECONDS * sample_rate)
    start = max(int(loud[0]) - keep, 0)
    stop = min(int(loud[-1]) + 1 + keep, audio.size)
    return audio[start:stop]


def _fade_edges(audio: np.ndarray, fade_samples: int) -> np.ndarray:
    """Return a copy with linear fade-in/out so chunk joins never click."""
    if fade_samples <= 0 or audio.size < 2 * fade_samples:
        return audio
    ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    faded = audio.copy()
    faded[:fade_samples] *= ramp
    faded[-fade_samples:] *= ramp[::-1]
    return faded


def _match_loudness(chunks: list[np.ndarray]) -> list[np.ndarray]:
    """Scale chunks toward their median RMS so no clause drops in volume."""
    rms = [float(np.sqrt(np.mean(chunk**2))) if chunk.size else 0.0 for chunk in chunks]
    audible = [value for value in rms if value > 1e-4]
    if len(audible) < 2:
        return chunks
    target = float(np.median(audible))
    low, high = LOUDNESS_GAIN_RANGE
    leveled = [
        chunk * float(np.clip(target / value, low, high)) if value > 1e-4 else chunk
        for chunk, value in zip(chunks, rms, strict=True)
    ]
    peak = max((float(np.max(np.abs(chunk))) for chunk in leveled if chunk.size), default=0.0)
    if peak > PEAK_LIMIT:
        leveled = [chunk * (PEAK_LIMIT / peak) for chunk in leveled]
    return leveled


def _pack_clauses(clauses: list[str], minimum: int) -> list[str]:
    """Merge only clauses shorter than `minimum` into their neighbour.

    Larger clauses stay separate on purpose: chunks isolate mumbling clauses
    (e.g. hard number phrases) so they cannot contaminate clean neighbours.
    """
    packed: list[str] = []
    current = ""
    for clause in clauses:
        current = f"{current} {clause}".strip()
        if len(current) >= minimum:
            packed.append(current)
            current = ""
    if current:
        if packed and len(current) < minimum:
            packed[-1] = f"{packed[-1]} {current}"
        else:
            packed.append(current)
    return packed


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
        allow_candidate: bool = False,
        frontend: ValtecFrontend,
        seed: int | None = None,
        max_audio_seconds: float = 60.0,
        noise_scale: float | None = None,
        length_scale: float | None = None,
        sentence_chunking: bool = True,
    ) -> None:
        self.voice = voice or VALTEC_TTS_CONFIG.default_voice
        self.artifact_root = artifact_root
        self.artifact_variant = artifact_variant
        self.allow_reference = allow_reference
        self.allow_candidate = allow_candidate
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
        self._sentence_chunking = sentence_chunking

        self._sessions: dict[str, Any] = {}
        self._sample_rate = 24000
        self._hop_length = 512
        self._noise_scale = 0.667
        self._length_scale = 1.0
        self._speaker_map: dict[str, int] = {}
        self._artifacts: ValtecOnnxArtifacts | None = None
        self._last_frontend_metadata: dict[str, Any] = {}
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._sessions:
            return

        import onnxruntime as ort

        artifacts = resolve_valtec_onnx_artifacts(
            self.artifact_root,
            variant=self.artifact_variant,
            allow_reference=self.allow_reference,
            allow_candidate=self.allow_candidate,
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
        sid = np.array([self._speaker_id(selected_voice)], dtype=np.int64)
        plan = (
            self._plan_chunks(text_clean)
            if self._sentence_chunking
            else [(text_clean, 0.0)]
        )
        fade_samples = int(EDGE_FADE_SECONDS * self._sample_rate)

        chunk_audios: list[np.ndarray] = []
        unknown_total = 0
        backend = ""
        for chunk_text, _gap in plan:
            chunk_audio, chunk_meta = self._render_chunk(chunk_text, sid)
            chunk_audios.append(chunk_audio)
            backend = chunk_meta["backend"]
            unknown_total += chunk_meta["unknown_phoneme_count"]

        pieces: list[np.ndarray] = []
        if len(plan) > 1:
            trimmed = [
                _trim_edge_silence(chunk_audio, self._sample_rate)
                for chunk_audio in chunk_audios
            ]
            for chunk_audio, (_text, gap_seconds) in zip(
                _match_loudness(trimmed), plan, strict=True
            ):
                if pieces:
                    pieces.append(
                        np.zeros(int(gap_seconds * self._sample_rate), dtype=np.float32)
                    )
                pieces.append(_fade_edges(chunk_audio, fade_samples))
        else:
            pieces = chunk_audios

        self._last_frontend_metadata = {
            "backend": backend,
            "unknown_phoneme_count": unknown_total,
        }
        audio_np = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
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

    def _plan_chunks(self, text: str) -> list[tuple[str, float]]:
        """Return (chunk_text, silence_before_seconds) pairs; first gap is unused."""
        plan: list[tuple[str, float]] = []
        for sentence in split_sentences(text):
            if len(sentence) <= LONG_SENTENCE_CHARS:
                plan.append((sentence, INTER_SENTENCE_SILENCE_SECONDS))
                continue
            clauses = [
                clause.strip()
                for clause in CLAUSE_BOUNDARY.split(sentence)
                if clause.strip()
            ]
            plan.extend(
                (
                    chunk,
                    INTER_SENTENCE_SILENCE_SECONDS if index == 0 else CLAUSE_SILENCE_SECONDS,
                )
                for index, chunk in enumerate(_pack_clauses(clauses, MIN_CLAUSE_CHARS))
            )
        return plan

    def _render_chunk(
        self, text: str, sid: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        model_inputs = self.frontend.prepare(text)
        seq_len = len(model_inputs.phone_ids)
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

        length_scale = self._length_scale
        duration_scales = None
        if self._sentence_chunking:
            length_scale = self._paced_length_scale(model_inputs, logw, x_mask)
            flags = getattr(model_inputs, "foreign_flags", ())
            if flags:
                duration_scales = np.where(
                    np.asarray(flags, dtype=np.float32) > 0, FOREIGN_SLOWDOWN, 1.0
                ).astype(np.float32)

        z_p, y_mask = self._expand_latents(
            m_p,
            logs_p,
            logw,
            x_mask,
            length_scale=length_scale,
            noise_scale=self._noise_scale,
            duration_scales=duration_scales,
        )
        (z,) = self._sessions["flow"].run(None, {"z_p": z_p, "y_mask": y_mask, "g": g})
        (audio,) = self._sessions["decoder"].run(None, {"z": z, "g": g})

        audio_np = np.ascontiguousarray(np.asarray(audio).squeeze(), dtype=np.float32)
        metadata = {
            "backend": model_inputs.backend,
            "unknown_phoneme_count": model_inputs.unknown_phoneme_count,
        }
        return audio_np, metadata

    def _paced_length_scale(
        self, model_inputs: Any, logw: np.ndarray, x_mask: np.ndarray
    ) -> float:
        """Cap chunks whose predicted speaking rate would sound slurred."""
        length_scale = self._length_scale
        spoken_phones = sum(1 for phone in model_inputs.phone_ids if phone != 0)
        frames = float(np.ceil(np.exp(logw) * x_mask * length_scale).sum())
        if spoken_phones and frames > 0:
            seconds = frames * self._hop_length / self._sample_rate
            rate = spoken_phones / seconds
            if rate > MAX_PHONES_PER_SECOND:
                length_scale *= rate / MAX_PHONES_PER_SECOND
        return min(length_scale, self._length_scale * MAX_ADAPTIVE_SCALE)

    def _expand_latents(
        self,
        m_p: np.ndarray,
        logs_p: np.ndarray,
        logw: np.ndarray,
        x_mask: np.ndarray,
        *,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        duration_scales: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        weights = np.exp(logw) * x_mask * length_scale
        if duration_scales is not None:
            weights = weights * duration_scales.reshape(weights.shape)
        durations = np.ceil(weights).astype(np.int64).reshape(-1)
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
