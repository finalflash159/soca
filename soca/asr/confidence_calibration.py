"""Confidence-guard calibration, kept free of the ASR model stack.

Loading these thresholds is pure JSON work. It used to live beside ``RobustASR``,
so ``soca status`` imported transformers, torchaudio and torch just to report
whether a calibration file existed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


