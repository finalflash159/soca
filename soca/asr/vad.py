"""Silero VAD wrapper for speech extraction before Whisper transcription.

Whisper hallucinates on silence/noise. Pre-filtering with VAD reduces
hallucinations by 40-60% (Bain et al. 2023, faster-whisper).

References:
- https://github.com/snakers4/silero-vad
- WhisperX paper: https://arxiv.org/abs/2303.00747
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class VADResult:
    """VAD output: original audio split into speech-only chunks."""

    has_speech: bool
    speech_audio: np.ndarray  # concatenated speech regions
    speech_duration_ms: float
    original_duration_ms: float
    speech_ratio: float  # speech_duration / original_duration
    vad_latency_ms: float
    n_speech_segments: int


class SpeechDetector:
    """Silero VAD wrapper. Extracts speech regions from raw audio."""

    SAMPLE_RATE = 16000

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 500,
        speech_pad_ms: int = 200,
    ):
        """
        Args:
            threshold: VAD confidence threshold (0-1). Lower = more sensitive.
                0.5 = default. Lower (0.3) catches whispered speech but more FPs.
            min_speech_ms: discard speech segments shorter than this
            min_silence_ms: bridge silence gaps shorter than this between speech.
                Default 500 (Whisper-tuned, voice-command). Silero's own default
                is 100, faster-whisper uses 2000 for long-form audio.
            speech_pad_ms: pad each speech segment with N ms on both sides.
                Default 200 (Whisper-tuned). Silero's own default is 30 —
                too tight for Whisper, which expects natural sentence boundaries.
        """
        from silero_vad import get_speech_timestamps, load_silero_vad

        self.model = load_silero_vad()
        self.get_speech_timestamps = get_speech_timestamps
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.speech_pad_ms = speech_pad_ms

    def detect(self, audio: np.ndarray) -> VADResult:
        """Run VAD on float32 16kHz mono audio. Returns speech-only audio."""
        t0 = time.perf_counter()
        original_duration_ms = len(audio) / self.SAMPLE_RATE * 1000

        # get_speech_timestamps returns list of {"start": idx, "end": idx}
        speech_timestamps = self.speech_timestamps(audio)

        if not speech_timestamps:
            return VADResult(
                has_speech=False,
                speech_audio=np.array([], dtype=np.float32),
                speech_duration_ms=0.0,
                original_duration_ms=original_duration_ms,
                speech_ratio=0.0,
                vad_latency_ms=(time.perf_counter() - t0) * 1000,
                n_speech_segments=0,
            )

        # Concatenate speech segments
        speech_chunks = []
        speech_samples = 0
        for ts in speech_timestamps:
            chunk = audio[ts["start"] : ts["end"]]
            speech_chunks.append(chunk)
            speech_samples += len(chunk)

        speech_audio = np.concatenate(speech_chunks).astype(np.float32)
        speech_duration_ms = speech_samples / self.SAMPLE_RATE * 1000
        vad_latency_ms = (time.perf_counter() - t0) * 1000

        return VADResult(
            has_speech=True,
            speech_audio=speech_audio,
            speech_duration_ms=speech_duration_ms,
            original_duration_ms=original_duration_ms,
            speech_ratio=speech_duration_ms / max(original_duration_ms, 1.0),
            vad_latency_ms=vad_latency_ms,
            n_speech_segments=len(speech_timestamps),
        )

    def speech_timestamps(self, audio: np.ndarray) -> list[dict[str, int]]:
        import torch

        audio_tensor = torch.from_numpy(audio).float()
        return self.get_speech_timestamps(
            audio_tensor,
            self.model,
            sampling_rate=self.SAMPLE_RATE,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
            speech_pad_ms=self.speech_pad_ms,
        )
