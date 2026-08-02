from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .deloop import has_excessive_repetition, remove_consecutive_repeats
    from .hallucination_heuristics import (
        HeuristicCheck,
        check_heuristics,
        is_filler_only,
        looks_like_context_echo,
        n_gram_repetition,
        repetition_ratio,
    )
    from .registry import (
        ASR_BAKEOFF_MODEL_KEYS,
        ASR_FULL_MODEL_KEYS,
        ASR_MODEL_REGISTRY,
        DEFAULT_ASR_MODEL_KEY,
        ASRModelConfig,
        get_asr_model_config,
        get_asr_profile_model_keys,
    )
    from .result import ASRResult
    from .robust_asr import RobustASR, RobustASRResult
    from .vad import SpeechDetector, VADResult
    from .whisper_onnx import VietnameseASR

_EXPORTS = {
    "ASRResult": (".result", "ASRResult"),
    "ASRModelConfig": (".registry", "ASRModelConfig"),
    "ASR_BAKEOFF_MODEL_KEYS": (".registry", "ASR_BAKEOFF_MODEL_KEYS"),
    "ASR_FULL_MODEL_KEYS": (".registry", "ASR_FULL_MODEL_KEYS"),
    "ASR_MODEL_REGISTRY": (".registry", "ASR_MODEL_REGISTRY"),
    "DEFAULT_ASR_MODEL_KEY": (".registry", "DEFAULT_ASR_MODEL_KEY"),
    "HeuristicCheck": (".hallucination_heuristics", "HeuristicCheck"),
    "RobustASR": (".robust_asr", "RobustASR"),
    "RobustASRResult": (".robust_asr", "RobustASRResult"),
    "SpeechDetector": (".vad", "SpeechDetector"),
    "VADResult": (".vad", "VADResult"),
    "VietnameseASR": (".whisper_onnx", "VietnameseASR"),
    "check_heuristics": (".hallucination_heuristics", "check_heuristics"),
    "get_asr_model_config": (".registry", "get_asr_model_config"),
    "get_asr_profile_model_keys": (".registry", "get_asr_profile_model_keys"),
    "has_excessive_repetition": (".deloop", "has_excessive_repetition"),
    "is_filler_only": (".hallucination_heuristics", "is_filler_only"),
    "looks_like_context_echo": (".hallucination_heuristics", "looks_like_context_echo"),
    "n_gram_repetition": (".hallucination_heuristics", "n_gram_repetition"),
    "remove_consecutive_repeats": (".deloop", "remove_consecutive_repeats"),
    "repetition_ratio": (".hallucination_heuristics", "repetition_ratio"),
}

__all__ = (
    "ASRResult",
    "ASRModelConfig",
    "ASR_BAKEOFF_MODEL_KEYS",
    "ASR_FULL_MODEL_KEYS",
    "ASR_MODEL_REGISTRY",
    "DEFAULT_ASR_MODEL_KEY",
    "HeuristicCheck",
    "RobustASR",
    "RobustASRResult",
    "SpeechDetector",
    "VADResult",
    "VietnameseASR",
    "check_heuristics",
    "get_asr_model_config",
    "get_asr_profile_model_keys",
    "has_excessive_repetition",
    "is_filler_only",
    "looks_like_context_echo",
    "n_gram_repetition",
    "remove_consecutive_repeats",
    "repetition_ratio",
)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
