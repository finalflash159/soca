from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, Protocol

from .context import ASRContextLimits
from .qwen_artifacts import QwenASRArtifactSpec

QWEN_CONFIDENCE_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "asr" / "qwen_confidence_calibration.json"
)
QWEN_ASR_MAX_NEW_TOKENS = 128


def _digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


CONTEXT_ECHO_POLICY_DIGEST = _digest(
    {
        "minimum_unique_tokens": 4,
        "mode": "token_set_overlap",
        "normalization": "lower_word_regex_v1",
        "threshold": 0.6,
    }
)
DEFAULT_VAD_POLICY_DIGEST = _digest(
    {
        "min_silence_ms": 500,
        "min_speech_ms": 250,
        "sample_rate": 16_000,
        "speech_pad_ms": 200,
        "threshold": 0.5,
    }
)


class VADPolicySource(Protocol):
    SAMPLE_RATE: ClassVar[int]
    threshold: float
    min_speech_ms: int
    min_silence_ms: int
    speech_pad_ms: int


class ASRCalibrationError(RuntimeError):
    pass


class ASRCalibrationNotReady(ASRCalibrationError):
    pass


@dataclass(frozen=True, slots=True)
class ASRCalibrationIdentity:
    engine: str
    model_key: str
    artifact_digest: str
    runtime_lock_digest: str
    context_policy_digest: str
    context_echo_policy_digest: str
    vad_policy_digest: str
    decode_strategy: str
    language: str
    device: str
    dtype: str
    max_new_tokens: int

    def __post_init__(self) -> None:
        strings = (
            self.engine,
            self.model_key,
            self.artifact_digest,
            self.runtime_lock_digest,
            self.context_policy_digest,
            self.context_echo_policy_digest,
            self.vad_policy_digest,
            self.decode_strategy,
            self.language,
            self.device,
            self.dtype,
        )
        if any(not value.strip() for value in strings):
            raise ValueError("calibration identity fields must be non-empty")
        if isinstance(self.max_new_tokens, bool) or self.max_new_tokens < 1:
            raise ValueError("calibration max_new_tokens must be positive")

    @property
    def payload(self) -> dict[str, object]:
        return {
            "artifact_digest": self.artifact_digest,
            "context_policy_digest": self.context_policy_digest,
            "context_echo_policy_digest": self.context_echo_policy_digest,
            "decode_strategy": self.decode_strategy,
            "device": self.device,
            "dtype": self.dtype,
            "engine": self.engine,
            "language": self.language,
            "max_new_tokens": self.max_new_tokens,
            "model_key": self.model_key,
            "runtime_lock_digest": self.runtime_lock_digest,
            "vad_policy_digest": self.vad_policy_digest,
        }

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ASRConfidenceCalibration:
    identity: ASRCalibrationIdentity
    min_avg_logprob: float
    max_compression_ratio: float
    source_path: Path
    created_at_utc: str


def compute_vad_policy_digest(detector: VADPolicySource) -> str:
    values = {
        "min_silence_ms": detector.min_silence_ms,
        "min_speech_ms": detector.min_speech_ms,
        "sample_rate": detector.SAMPLE_RATE,
        "speech_pad_ms": detector.speech_pad_ms,
        "threshold": detector.threshold,
    }
    if (
        isinstance(detector.SAMPLE_RATE, bool)
        or detector.SAMPLE_RATE < 1
        or isinstance(detector.threshold, bool)
        or not math.isfinite(detector.threshold)
        or not 0 <= detector.threshold <= 1
    ):
        raise ASRCalibrationError("VAD sample rate or threshold is invalid")
    for name in ("min_silence_ms", "min_speech_ms", "speech_pad_ms"):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ASRCalibrationError(f"VAD {name} must be a non-negative integer")
    return _digest(values)


def qwen_calibration_identity(
    spec: QwenASRArtifactSpec,
    *,
    context_limits: ASRContextLimits | None = None,
    vad_policy_digest: str = DEFAULT_VAD_POLICY_DIGEST,
) -> ASRCalibrationIdentity:
    if spec.runtime_lock_digest is None:
        raise ASRCalibrationError(f"Qwen artifact {spec.key} has no runtime lock identity")
    limits = context_limits or ASRContextLimits()
    if spec.context_policy_digest is not None and spec.context_policy_digest != limits.policy_digest:
        raise ASRCalibrationError(
            f"Qwen artifact {spec.key} context policy does not match the runtime policy"
        )
    return ASRCalibrationIdentity(
        engine="qwen_service",
        model_key=spec.key,
        artifact_digest=spec.digest,
        runtime_lock_digest=spec.runtime_lock_digest,
        context_policy_digest=limits.policy_digest,
        context_echo_policy_digest=CONTEXT_ECHO_POLICY_DIGEST,
        vad_policy_digest=vad_policy_digest,
        decode_strategy="llm_decoder",
        language="Vietnamese",
        device=spec.device,
        dtype=spec.dtype,
        max_new_tokens=QWEN_ASR_MAX_NEW_TOKENS,
    )


def load_strict_confidence_calibration(
    identity: ASRCalibrationIdentity,
    path: Path = QWEN_CONFIDENCE_CALIBRATION_PATH,
) -> ASRConfidenceCalibration:
    if not path.is_file():
        raise ASRCalibrationNotReady(
            f"confidence calibration is missing for {identity.model_key} ({identity.digest[:12]})"
        )
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ASRCalibrationError(f"cannot read confidence calibration: {path}") from exc
    records = root.get("calibrations") if isinstance(root, Mapping) else None
    payload = records.get(identity.digest) if isinstance(records, Mapping) else None
    if not isinstance(payload, Mapping):
        raise ASRCalibrationNotReady(
            f"confidence calibration identity is not qualified: {identity.digest}"
        )
    if payload.get("identity") != identity.payload:
        raise ASRCalibrationError("confidence calibration identity payload does not match its key")
    thresholds = payload.get("recommended_thresholds")
    if not isinstance(thresholds, Mapping):
        raise ASRCalibrationError("confidence calibration thresholds are malformed")
    try:
        min_avg_logprob = float(thresholds["min_avg_logprob"])
        max_compression_ratio = float(thresholds["max_compression_ratio"])
        created_at_utc = payload["created_at_utc"]
        if not math.isfinite(min_avg_logprob):
            raise ValueError("min_avg_logprob must be finite")
        if not math.isfinite(max_compression_ratio) or max_compression_ratio <= 0:
            raise ValueError("max_compression_ratio must be finite and positive")
        if not isinstance(created_at_utc, str) or not created_at_utc.strip():
            raise ValueError("created_at_utc must be a non-empty string")
        parsed_created_at = datetime.fromisoformat(created_at_utc.replace("Z", "+00:00"))
        if parsed_created_at.tzinfo is None or parsed_created_at.utcoffset() != UTC.utcoffset(None):
            raise ValueError("created_at_utc must be an ISO-8601 UTC timestamp")
        return ASRConfidenceCalibration(
            identity=identity,
            min_avg_logprob=min_avg_logprob,
            max_compression_ratio=max_compression_ratio,
            source_path=path,
            created_at_utc=created_at_utc,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ASRCalibrationError("confidence calibration record is malformed") from exc
