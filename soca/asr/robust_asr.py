from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .context import ASRContextBuilder, ASRContextSnapshot
from .deloop import remove_consecutive_repeats
from .hallucination_heuristics import (
    HeuristicCheck,
    check_heuristics,
    compression_ratio,
    looks_like_context_echo,
)
from .protocols import VoiceASRBackend
from .registry import DEFAULT_ASR_MODEL_KEY
from .vad import SpeechDetector, VADResult
from .voice_backend import PhoWhisperVoiceBackend
from .whisper_onnx import ASRResult

LOGGER = logging.getLogger(__name__)

ASR_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "asr"
CONFIDENCE_CALIBRATION_PATH = ASR_DATA_DIR / "threshold_calibration.json"
DEFAULT_MIN_AVG_LOGPROB = -0.725
DEFAULT_MAX_COMPRESSION_RATIO = 2.4


@dataclass(frozen=True)
class ConfidenceGuardCalibration:
    """Model-specific confidence guard thresholds loaded from calibration data."""

    model_key: str
    min_avg_logprob: float
    max_compression_ratio: float
    source_path: Path
    created_at_utc: str = ""


def _payload_model_key(payload: dict[str, Any]) -> str | None:
    value = payload.get("model_key")
    if value:
        return str(value)

    # Backward compatibility for older single-model payloads that only stored
    # runtime metadata. This keeps the loader conservative: it only infers a
    # model when the known model key appears in the model_dir path.
    model_dir = payload.get("runtime_identity", {}).get("asr", {}).get("model_dir", "")
    model_dir = str(model_dir)
    for model_key in (
        "phowhisper_tiny",
        "phowhisper_base",
        "phowhisper_small",
        "phowhisper_medium",
    ):
        if model_key.replace("_", "-") in model_dir or model_key in model_dir:
            return model_key
    return None


def load_confidence_guard_calibration(
    model_key: str,
    path: Path = CONFIDENCE_CALIBRATION_PATH,
) -> ConfidenceGuardCalibration | None:
    """Load calibrated confidence thresholds for one ASR model.

    The canonical format is:

        {
          "asr_confidence_by_model": {
            "phowhisper_base": {
              "model_key": "phowhisper_base",
              "recommended_thresholds": {...}
            }
          }
        }

    The older `asr_confidence` singleton is only accepted when its runtime
    identity matches the requested model.
    """
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    by_model = data.get("asr_confidence_by_model", {})
    candidates: list[dict[str, Any]] = []
    if isinstance(by_model, dict) and isinstance(by_model.get(model_key), dict):
        candidates.append(by_model[model_key])

    singleton = data.get("asr_confidence")
    if isinstance(singleton, dict):
        candidates.append(singleton)

    for payload in candidates:
        if _payload_model_key(payload) != model_key:
            continue

        thresholds = payload.get("recommended_thresholds", {})
        try:
            return ConfidenceGuardCalibration(
                model_key=model_key,
                min_avg_logprob=float(thresholds["min_avg_logprob"]),
                max_compression_ratio=float(thresholds["max_compression_ratio"]),
                source_path=path,
                created_at_utc=str(payload.get("created_at_utc", "")),
            )
        except (KeyError, TypeError, ValueError):
            continue

    return None


@dataclass(frozen=True)
class RobustASRResult:
    """Full pipeline result with per-stage diagnostics.

    The intermediate text fields are useful for debugging without rerunning
    the model.
    """

    text: str  # Final cleaned text. Empty if any stage rejected.
    raw_text: str  # Whisper output before any post-processing.
    text_after_deloop: str
    has_speech: bool
    was_looping: bool
    heuristic: HeuristicCheck | None = None
    rejection_reason: str = ""
    vad: VADResult | None = None
    asr: ASRResult | None = None
    total_latency_ms: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    # Describes the logprob guard specifically (model-dependent). See
    # `compression_guard_status` for the independent text-only guard — a
    # rejection can come from either one regardless of this field's value.
    confidence_guard_status: str = ""
    compression_guard_status: str = ""
    alternatives: tuple[str, ...] = ()
    context_digest: str = ""
    context_provenance: tuple[str, ...] = ()


class RobustASR:
    """VAD, ASR and observable transcript quality checks."""

    def __init__(
        self,
        asr: VoiceASRBackend | None = None,
        vad: SpeechDetector | None = None,
        # Calibrated by `uv run python -m local.calibrate_asr_confidence`
        # on 200 FLEURS vi speech samples + 50 non-speech samples:
        # speech p01=-0.250, noise max=-1.200, midpoint=-0.725.
        # Pass confidence_profile_model_key="<model_key>" after recalibrating
        # thresholds for another ASR model. Pass None only for explicit custom
        # thresholds whose model identity is tracked outside this class.
        min_avg_logprob: float = DEFAULT_MIN_AVG_LOGPROB,
        max_compression_ratio: float = DEFAULT_MAX_COMPRESSION_RATIO,
        context_echo_min_contiguous_tokens: int = 4,
        partial_max_new_tokens: int | None = None,
        confidence_profile_model_key: str | None = DEFAULT_ASR_MODEL_KEY,
        confidence_guard_skip_reason: str | None = None,
        context_provider: Callable[[], ASRContextSnapshot] | None = None,
    ):
        self.asr = asr if asr is not None else PhoWhisperVoiceBackend(DEFAULT_ASR_MODEL_KEY)
        self.vad = vad if vad is not None else SpeechDetector()
        self._context_provider = context_provider
        self.last_context = ASRContextBuilder().build(())
        self._close_lock = threading.Lock()
        self._closed = False
        runtime_model_key = getattr(self.asr, "model_key", None)
        self.asr_model_key = str(runtime_model_key or DEFAULT_ASR_MODEL_KEY)
        self.min_avg_logprob = min_avg_logprob
        self.max_compression_ratio = max_compression_ratio
        if (
            isinstance(context_echo_min_contiguous_tokens, bool)
            or not isinstance(context_echo_min_contiguous_tokens, int)
            or context_echo_min_contiguous_tokens < 1
        ):
            raise ValueError("context echo span threshold must be a positive integer")
        self.context_echo_min_contiguous_tokens = context_echo_min_contiguous_tokens
        if partial_max_new_tokens is not None and (
            isinstance(partial_max_new_tokens, bool)
            or not isinstance(partial_max_new_tokens, int)
            or partial_max_new_tokens < 1
        ):
            raise ValueError("partial max_new_tokens must be a positive integer or None")
        self.partial_max_new_tokens = partial_max_new_tokens
        self.confidence_profile_model_key = confidence_profile_model_key

        # Backend capability. Backends should declare `supports_avg_logprob`
        # explicitly (VietnameseASR does); missing entirely is treated as
        # capable for backward compatibility with pre-existing callers that
        # predate this attribute, but that fallback is logged so a backend
        # that genuinely can't produce a logprob and forgot to say so is
        # observable instead of silently getting a meaningless threshold
        # check applied to it.
        if hasattr(self.asr, "supports_avg_logprob"):
            backend_has_logprob = bool(self.asr.supports_avg_logprob)
        else:
            backend_has_logprob = True
            LOGGER.warning(
                "ASR backend %s does not declare supports_avg_logprob; "
                "assuming logprob-capable for backward compatibility. "
                "New backends should set this attribute explicitly.",
                type(self.asr).__name__,
            )

        if confidence_guard_skip_reason:
            self.use_logprob_guard = False
            self.confidence_guard_status = confidence_guard_skip_reason
        elif not backend_has_logprob:
            self.use_logprob_guard = False
            reported_model_key = str(runtime_model_key) if runtime_model_key else "unknown_backend"
            self.confidence_guard_status = f"skipped:backend_has_no_logprob:{reported_model_key}"
        else:
            (
                self.use_logprob_guard,
                self.confidence_guard_status,
            ) = self._resolve_confidence_guard_status(confidence_profile_model_key)

        # Compression is a text-only signal independent of model identity, so
        # it must never be disabled just because the logprob guard is off.
        # confidence_guard_status above only ever describes the logprob
        # guard; expose this separately so a compression rejection is never
        # misread as "all confidence checks were skipped".
        self.use_compression_guard = True
        self.compression_guard_status = "enabled"

    def _resolve_confidence_guard_status(self, profile_model_key: str | None) -> tuple[bool, str]:
        """Enable confidence thresholds only when their model identity matches."""
        if profile_model_key is None:
            return True, "enabled:unversioned_profile"
        if profile_model_key == self.asr_model_key:
            return True, f"enabled:{profile_model_key}"
        return (
            False,
            f"skipped:model_mismatch:profile={profile_model_key},runtime={self.asr_model_key}",
        )

    def transcribe(self, audio: np.ndarray) -> RobustASRResult:
        """Run the production pipeline on a 1D float32 16kHz mono array."""
        t0 = time.perf_counter()
        self.last_context = ASRContextBuilder().build(())

        # Stage 1: VAD
        vad_result = self.vad.detect(audio)
        if not vad_result.has_speech:
            return RobustASRResult(
                text="",
                raw_text="",
                text_after_deloop="",
                has_speech=False,
                was_looping=False,
                heuristic=HeuristicCheck(False, "no_speech_vad_skip", 0.0),
                rejection_reason="no_speech",
                vad=vad_result,
                asr=None,
                total_latency_ms=(time.perf_counter() - t0) * 1000,
                confidence_guard_status=self.confidence_guard_status,
                compression_guard_status=self.compression_guard_status,
            )

        # Stage 2: ASR on speech-only audio (saves compute when VAD trimmed silence)
        context = (
            self._context_provider() if self._context_provider is not None else self.last_context
        )
        self.last_context = context
        asr_result = self.asr.transcribe(vad_result.speech_audio, context=context.text)
        raw_text = asr_result.text
        alternatives = _normalise_alternatives(asr_result, primary=raw_text)
        avg_logprob = asr_result.avg_logprob
        raw_compression_ratio = compression_ratio(raw_text)

        # Stage 2b: model-confidence guard on raw ASR output.
        # This must run before text-cleaning stages so diagnostics reflect
        # exactly what the model produced.
        if asr_result.hit_max_new_tokens is True:
            return RobustASRResult(
                text="",
                raw_text=raw_text,
                text_after_deloop=raw_text,
                has_speech=True,
                was_looping=False,
                rejection_reason="decode_limit_reached",
                vad=vad_result,
                asr=asr_result,
                total_latency_ms=(time.perf_counter() - t0) * 1000,
                avg_logprob=avg_logprob,
                compression_ratio=raw_compression_ratio,
                confidence_guard_status=self.confidence_guard_status,
                compression_guard_status=self.compression_guard_status,
                alternatives=alternatives,
                context_digest=context.digest,
                context_provenance=context.provenances,
            )

        if not raw_text.strip():
            return RobustASRResult(
                text="",
                raw_text=raw_text,
                text_after_deloop=raw_text,
                has_speech=True,
                was_looping=False,
                rejection_reason="empty_asr",
                vad=vad_result,
                asr=asr_result,
                total_latency_ms=(time.perf_counter() - t0) * 1000,
                avg_logprob=avg_logprob,
                compression_ratio=raw_compression_ratio,
                confidence_guard_status=self.confidence_guard_status,
                compression_guard_status=self.compression_guard_status,
                alternatives=alternatives,
                context_digest=context.digest,
                context_provenance=context.provenances,
            )

        if self.use_logprob_guard and avg_logprob < self.min_avg_logprob:
            return RobustASRResult(
                text="",
                raw_text=raw_text,
                text_after_deloop=raw_text,
                has_speech=True,
                was_looping=False,
                rejection_reason=f"low_confidence:{avg_logprob:.2f}",
                vad=vad_result,
                asr=asr_result,
                total_latency_ms=(time.perf_counter() - t0) * 1000,
                avg_logprob=avg_logprob,
                compression_ratio=raw_compression_ratio,
                confidence_guard_status=self.confidence_guard_status,
                compression_guard_status=self.compression_guard_status,
                alternatives=alternatives,
                context_digest=context.digest,
                context_provenance=context.provenances,
            )

        if self.use_compression_guard and raw_compression_ratio > self.max_compression_ratio:
            return RobustASRResult(
                text="",
                raw_text=raw_text,
                text_after_deloop=raw_text,
                has_speech=True,
                was_looping=False,
                rejection_reason=f"high_compression:{raw_compression_ratio:.2f}",
                vad=vad_result,
                asr=asr_result,
                total_latency_ms=(time.perf_counter() - t0) * 1000,
                avg_logprob=avg_logprob,
                compression_ratio=raw_compression_ratio,
                confidence_guard_status=self.confidence_guard_status,
                compression_guard_status=self.compression_guard_status,
                alternatives=alternatives,
                context_digest=context.digest,
                context_provenance=context.provenances,
            )

        # Context-echo guard: a context-aware backend can occasionally
        # return its system prompt verbatim instead of transcribing the
        # audio. That output passes every check above (clean text,
        # normal compression), so it needs its own dedicated check, run
        # before de-loop/heuristics touch the text.
        if context.text and looks_like_context_echo(
            raw_text,
            context.text,
            minimum_contiguous_tokens=self.context_echo_min_contiguous_tokens,
        ):
            return RobustASRResult(
                text="",
                raw_text=raw_text,
                text_after_deloop=raw_text,
                has_speech=True,
                was_looping=False,
                rejection_reason="context_echo",
                vad=vad_result,
                asr=asr_result,
                total_latency_ms=(time.perf_counter() - t0) * 1000,
                avg_logprob=avg_logprob,
                compression_ratio=raw_compression_ratio,
                confidence_guard_status=self.confidence_guard_status,
                compression_guard_status=self.compression_guard_status,
                alternatives=alternatives,
                context_digest=context.digest,
                context_provenance=context.provenances,
            )

        # Stage 3: De-loop
        text_deloop, was_looping = remove_consecutive_repeats(raw_text)

        # Stage 4: Heuristics safety net
        heuristic = check_heuristics(text_deloop, vad_result.speech_duration_ms)

        if heuristic.is_hallucination:
            final_text = ""
            reason = f"heuristic:{heuristic.reason}"
        elif not text_deloop.strip():
            final_text = ""
            reason = "empty_after_cleanup"
        else:
            final_text = text_deloop
            reason = ""

        return RobustASRResult(
            text=final_text,
            raw_text=raw_text,
            text_after_deloop=text_deloop,
            has_speech=True,
            was_looping=was_looping,
            heuristic=heuristic,
            rejection_reason=reason,
            vad=vad_result,
            asr=asr_result,
            avg_logprob=avg_logprob,
            compression_ratio=raw_compression_ratio,
            total_latency_ms=(time.perf_counter() - t0) * 1000,
            confidence_guard_status=self.confidence_guard_status,
            compression_guard_status=self.compression_guard_status,
            alternatives=alternatives,
            context_digest=context.digest,
            context_provenance=context.provenances,
        )

    def transcribe_partial(self, audio: np.ndarray) -> ASRResult:
        if self.partial_max_new_tokens is None:
            return self.asr.transcribe(audio, context="")
        return self.asr.transcribe(
            audio,
            max_new_tokens=self.partial_max_new_tokens,
            context="",
        )

    def snapshot_context(self) -> ASRContextSnapshot:
        context = (
            self._context_provider() if self._context_provider is not None else self.last_context
        )
        self.last_context = context
        return context

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self.asr.close()

    def transcribe_file(self, path: str | Path) -> RobustASRResult:
        import librosa

        audio, _ = librosa.load(str(path), sr=16000, mono=True)
        return self.transcribe(audio.astype(np.float32))


def _normalise_alternatives(asr_result: Any, *, primary: str) -> tuple[str, ...]:
    """Preserve backend-provided N-best text without inventing candidates."""
    raw = getattr(asr_result, "alternatives", ())
    if not isinstance(raw, (list, tuple)):
        return ()
    primary_normalized = " ".join(primary.split()).casefold()
    values: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        normalized = " ".join(item.split())
        if not normalized or normalized.casefold() == primary_normalized:
            continue
        if normalized not in values:
            values.append(normalized)
    return tuple(values)
