"""Production-grade ASR pipeline: VAD → ASR → de-loop → BoH → heuristics.

This is the entry point used by the voice assistant runtime (D4 orchestrator).
Replicates the SOTA approach from Barański et al. (ICASSP 2025) for Vietnamese
ASR. Current BoH/confidence artifacts are model-specific and should be
recalibrated when switching ASR checkpoints or decoder settings.

Behavior on rejected input: returns RobustASRResult with empty `text` and a
`rejection_reason`. Caller can branch on `.text` for happy path or surface
the reason for graceful "could you repeat?" UX.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .boh import VietnameseBoH
from .deloop import remove_consecutive_repeats
from .hallucination_heuristics import (
    HeuristicCheck,
    check_heuristics,
    compression_ratio,
)
from .registry import DEFAULT_ASR_MODEL_KEY
from .vad import SpeechDetector, VADResult
from .whisper_onnx import ASRResult, VietnameseASR


@dataclass(frozen=True)
class RobustASRResult:
    """Full pipeline result with per-stage diagnostics.

    The intermediate text fields (`raw_text`, `text_after_deloop`,
    `text_after_boh`) are useful for debugging: they let you trace exactly
    which stage modified the output without re-running the pipeline.
    """

    text: str  # Final cleaned text. Empty if any stage rejected.
    raw_text: str  # Whisper output before any post-processing.
    text_after_deloop: str
    text_after_boh: str
    has_speech: bool
    was_looping: bool
    boh_matches: tuple[str, ...] = field(default_factory=tuple)
    heuristic: HeuristicCheck | None = None
    rejection_reason: str = (
        ""  # "" if accepted, else "no_speech" | "heuristic:..." | "empty_after_boh"
    )
    vad: VADResult | None = None
    asr: ASRResult | None = None
    total_latency_ms: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    boh_status: str = ""
    confidence_guard_status: str = ""


class RobustASR:
    """5-stage hallucination-mitigation pipeline.

    Stages:
        1. Silero VAD (Whisper-tuned params) — skip ASR if no speech.
        2. PhoWhisper transcription on speech-only audio.
        3. De-loop: collapse consecutively repeated fragments.
        4. Model-specific Aho-Corasick BoH: remove empirical hallucination phrases.
        5. Heuristics: filler / n-gram / chars-per-100ms safety net.

    If a matching BoH artifact is missing or built for another ASR model,
    stage 4 is skipped and `boh_status` explains why. This prevents a
    PhoWhisper-tiny artifact from being silently applied to base/small/medium.
    """

    def __init__(
        self,
        asr: VietnameseASR | None = None,
        vad: SpeechDetector | None = None,
        boh: VietnameseBoH | None = None,
        # Calibrated by `uv run python -m local.calibrate_asr_confidence`
        # on 200 FLEURS vi speech samples + 50 non-speech samples:
        # speech p01=-0.250, noise max=-1.200, midpoint=-0.725.
        # Pass confidence_profile_model_key="<model_key>" after recalibrating
        # thresholds for another ASR model. Pass None only for explicit custom
        # thresholds whose model identity is tracked outside this class.
        min_avg_logprob: float = -0.725,
        max_compression_ratio: float = 2.4,
        confidence_profile_model_key: str | None = DEFAULT_ASR_MODEL_KEY,
    ):
        self.asr = asr if asr is not None else VietnameseASR(num_threads=4)
        self.vad = vad if vad is not None else SpeechDetector()
        runtime_model_key = getattr(self.asr, "model_key", None)
        self.asr_model_key = str(runtime_model_key or DEFAULT_ASR_MODEL_KEY)
        self.min_avg_logprob = min_avg_logprob
        self.max_compression_ratio = max_compression_ratio
        self.confidence_profile_model_key = confidence_profile_model_key
        (
            self.use_confidence_guard,
            self.confidence_guard_status,
        ) = self._resolve_confidence_guard_status(confidence_profile_model_key)
        self.boh, self.boh_status = self._resolve_boh(boh)

    def _resolve_confidence_guard_status(
        self, profile_model_key: str | None
    ) -> tuple[bool, str]:
        """Enable confidence thresholds only when their model identity matches."""
        if profile_model_key is None:
            return True, "enabled:unversioned_profile"
        if profile_model_key == self.asr_model_key:
            return True, f"enabled:{profile_model_key}"
        return (
            False,
            f"skipped:model_mismatch:profile={profile_model_key},runtime={self.asr_model_key}",
        )

    def _resolve_boh(self, boh: VietnameseBoH | None) -> tuple[VietnameseBoH | None, str]:
        """Load a model-specific BoH artifact or skip with a diagnostic reason."""
        if boh is None:
            model_path = VietnameseBoH.path_for_model(self.asr_model_key)
            if model_path.exists():
                loaded = VietnameseBoH(model_path)
                if loaded.model_key and loaded.model_key != self.asr_model_key:
                    return (
                        None,
                        f"skipped:model_mismatch:artifact={loaded.model_key},runtime={self.asr_model_key}",
                    )
                return loaded, f"loaded:{self.asr_model_key}"

            # Backward-compatible fallback for older tiny-only local setups.
            if self.asr_model_key == DEFAULT_ASR_MODEL_KEY and VietnameseBoH.DEFAULT_PATH.exists():
                loaded = VietnameseBoH()
                if loaded.model_key and loaded.model_key != self.asr_model_key:
                    return (
                        None,
                        f"skipped:model_mismatch:artifact={loaded.model_key},runtime={self.asr_model_key}",
                    )
                return loaded, f"loaded:legacy_default:{self.asr_model_key}"

            return None, f"skipped:missing_for_model:{self.asr_model_key}"

        artifact_model_key = getattr(boh, "model_key", None)
        if artifact_model_key and artifact_model_key != self.asr_model_key:
            return (
                None,
                f"skipped:model_mismatch:artifact={artifact_model_key},runtime={self.asr_model_key}",
            )
        if artifact_model_key:
            return boh, f"loaded:provided:{artifact_model_key}"
        return boh, "loaded:provided_unversioned"

    def transcribe(self, audio: np.ndarray) -> RobustASRResult:
        """Run the full 5-stage pipeline on a 1D float32 16kHz mono array."""
        t0 = time.perf_counter()

        # Stage 1: VAD
        vad_result = self.vad.detect(audio)
        if not vad_result.has_speech:
            return RobustASRResult(
                text="",
                raw_text="",
                text_after_deloop="",
                text_after_boh="",
                has_speech=False,
                was_looping=False,
                heuristic=HeuristicCheck(False, "no_speech_vad_skip", 0.0),
                rejection_reason="no_speech",
                vad=vad_result,
                asr=None,
                total_latency_ms=(time.perf_counter() - t0) * 1000,
                boh_status=self.boh_status,
                confidence_guard_status=self.confidence_guard_status,
            )

        # Stage 2: ASR on speech-only audio (saves compute when VAD trimmed silence)
        asr_result = self.asr.transcribe(vad_result.speech_audio)
        raw_text = asr_result.text
        avg_logprob = asr_result.avg_logprob
        raw_compression_ratio = compression_ratio(raw_text)

        # Stage 2b: model-confidence guard on raw ASR output.
        # This must run before text-cleaning stages so diagnostics reflect
        # exactly what the model produced.
        if not raw_text.strip():
            return RobustASRResult(
                text="",
                raw_text=raw_text,
                text_after_deloop=raw_text,
                text_after_boh=raw_text,
                has_speech=True,
                was_looping=False,
                rejection_reason="empty_asr",
                vad=vad_result,
                asr=asr_result,
                total_latency_ms=(time.perf_counter() - t0) * 1000,
                avg_logprob=avg_logprob,
                compression_ratio=raw_compression_ratio,
                boh_status=self.boh_status,
                confidence_guard_status=self.confidence_guard_status,
            )

        if self.use_confidence_guard:
            if avg_logprob < self.min_avg_logprob:
                return RobustASRResult(
                    text="",
                    raw_text=raw_text,
                    text_after_deloop=raw_text,
                    text_after_boh=raw_text,
                    has_speech=True,
                    was_looping=False,
                    rejection_reason=f"low_confidence:{avg_logprob:.2f}",
                    vad=vad_result,
                    asr=asr_result,
                    total_latency_ms=(time.perf_counter() - t0) * 1000,
                    avg_logprob=avg_logprob,
                    compression_ratio=raw_compression_ratio,
                    boh_status=self.boh_status,
                    confidence_guard_status=self.confidence_guard_status,
                )

            if raw_compression_ratio > self.max_compression_ratio:
                return RobustASRResult(
                    text="",
                    raw_text=raw_text,
                    text_after_deloop=raw_text,
                    text_after_boh=raw_text,
                    has_speech=True,
                    was_looping=False,
                    rejection_reason=f"high_compression:{raw_compression_ratio:.2f}",
                    vad=vad_result,
                    asr=asr_result,
                    total_latency_ms=(time.perf_counter() - t0) * 1000,
                    avg_logprob=avg_logprob,
                    compression_ratio=raw_compression_ratio,
                    boh_status=self.boh_status,
                    confidence_guard_status=self.confidence_guard_status,
                )

        # Stage 3: De-loop
        text_deloop, was_looping = remove_consecutive_repeats(raw_text)

        # Stage 4: BoH match (skipped if BoH artifact is missing)
        if self.boh is not None:
            boh_match = self.boh.match_and_clean(text_deloop)
            text_boh = boh_match.cleaned_text
            boh_matches = boh_match.matched_phrases
        else:
            text_boh = text_deloop
            boh_matches = ()

        # Stage 5: Heuristics safety net
        heuristic = check_heuristics(text_boh, vad_result.speech_duration_ms)

        if heuristic.is_hallucination:
            final_text = ""
            reason = f"heuristic:{heuristic.reason}"
        elif not text_boh.strip():
            final_text = ""
            reason = "empty_after_boh"
        else:
            final_text = text_boh
            reason = ""

        return RobustASRResult(
            text=final_text,
            raw_text=raw_text,
            text_after_deloop=text_deloop,
            text_after_boh=text_boh,
            has_speech=True,
            was_looping=was_looping,
            boh_matches=boh_matches,
            heuristic=heuristic,
            rejection_reason=reason,
            vad=vad_result,
            asr=asr_result,
            avg_logprob=avg_logprob,
            compression_ratio=raw_compression_ratio,
            total_latency_ms=(time.perf_counter() - t0) * 1000,
            boh_status=self.boh_status,
            confidence_guard_status=self.confidence_guard_status,
        )

    def transcribe_file(self, path: str | Path) -> RobustASRResult:
        import librosa

        audio, _ = librosa.load(str(path), sr=16000, mono=True)
        return self.transcribe(audio.astype(np.float32))
