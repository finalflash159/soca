from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from soca.asr.qwen_device import (
    QwenDeviceError,
    QwenDeviceMismatch,
    QwenDeviceUnavailable,
    assert_model_matches,
    torch_dtype,
    validate_execution_request,
)


def test_dtype_mapping_is_explicit() -> None:
    assert torch_dtype(torch, "float16") is torch.float16
    assert torch_dtype(torch, "float32") is torch.float32
    assert torch_dtype(torch, "bfloat16") is torch.bfloat16

    with pytest.raises(QwenDeviceError, match="unsupported Qwen dtype"):
        torch_dtype(torch, "float64")


def test_cpu_execution_is_always_available() -> None:
    assert validate_execution_request(torch, device="cpu", dtype="float32") == torch.device("cpu")


def test_mps_rejects_cpu_fallback_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable on this test host")

    with pytest.raises(QwenDeviceError, match="must disable CPU fallback"):
        validate_execution_request(torch, device="mps", dtype="float16")


def test_mps_unavailability_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_built=lambda: False, is_available=lambda: False)),
        device=torch.device,
    )
    with pytest.raises(QwenDeviceUnavailable, match="MPS is unavailable"):
        validate_execution_request(fake_torch, device="mps", dtype="float16")


def test_loaded_model_device_and_dtype_are_verified() -> None:
    model = SimpleNamespace(device="mps:0", dtype=torch.float16)
    assert assert_model_matches(model=model, device="mps", dtype="float16") == (
        "mps:0",
        "float16",
    )

    with pytest.raises(QwenDeviceMismatch, match="expected cpu"):
        assert_model_matches(model=model, device="cpu", dtype="float16")


def test_loaded_model_preserves_cuda_device_index() -> None:
    model = SimpleNamespace(device="cuda:1", dtype=torch.float16)

    assert assert_model_matches(model=model, device="cuda:1", dtype="float16") == (
        "cuda:1",
        "float16",
    )
    with pytest.raises(QwenDeviceMismatch, match="cuda:0"):
        assert_model_matches(model=model, device="cuda:0", dtype="float16")
