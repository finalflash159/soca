from __future__ import annotations

import pytest

from soca.asr.registry import (
    ASR_BAKEOFF_MODEL_KEYS,
    ASR_FULL_MODEL_KEYS,
    ASR_MODEL_REGISTRY,
    DEFAULT_ASR_MODEL_KEY,
    MODELS_ROOT,
    get_asr_model_config,
    get_asr_profile_model_keys,
)
from soca.asr.whisper_onnx import VietnameseASR


def test_default_asr_model_key_exists() -> None:
    assert DEFAULT_ASR_MODEL_KEY in ASR_MODEL_REGISTRY


def test_asr_registry_keys_match_config_keys() -> None:
    for key, config in ASR_MODEL_REGISTRY.items():
        assert config.model_key == key


def test_asr_model_paths_are_under_models_root() -> None:
    models_root = MODELS_ROOT.resolve()
    for config in ASR_MODEL_REGISTRY.values():
        assert config.local_dir.resolve().is_relative_to(models_root)


def test_asr_bakeoff_and_full_keys_exist() -> None:
    for key in ASR_BAKEOFF_MODEL_KEYS + ASR_FULL_MODEL_KEYS:
        assert key in ASR_MODEL_REGISTRY


def test_phowhisper_mid_tier_models_are_registered() -> None:
    base = get_asr_model_config("phowhisper_base")
    small = get_asr_model_config("phowhisper_small")
    medium = get_asr_model_config("phowhisper_medium")

    assert base.params_m == 74
    assert small.params_m == 244
    assert medium.params_m == 769
    assert base.download_command.endswith("--model phowhisper_base")


def test_asr_profiles() -> None:
    assert get_asr_profile_model_keys("minimal") == [DEFAULT_ASR_MODEL_KEY]
    assert get_asr_profile_model_keys("bakeoff") == [
        "phowhisper_tiny",
        "phowhisper_base",
        "phowhisper_small",
    ]
    assert get_asr_profile_model_keys("full")[-1] == "phowhisper_medium"


def test_unknown_asr_model_has_helpful_error() -> None:
    with pytest.raises(KeyError, match="Unknown ASR model key"):
        get_asr_model_config("not_a_real_asr_model")


def test_runtime_metadata_includes_model_identity_without_loading() -> None:
    asr = object.__new__(VietnameseASR)
    asr.model_key = "phowhisper_base"
    asr.model_config = get_asr_model_config("phowhisper_base")
    asr.model_dir = asr.model_config.local_dir
    asr.onnx_dir = asr.model_config.onnx_dir
    asr.providers = ["CPUExecutionProvider"]
    asr.suppress_tokens = set()
    asr.begin_suppress_tokens = set()

    metadata = asr.runtime_metadata()

    assert metadata["model_key"] == "phowhisper_base"
    assert metadata["hf_repo"] == "huuquyet/PhoWhisper-base"
    assert metadata["params_m"] == 74
