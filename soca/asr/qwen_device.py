from __future__ import annotations

import os
from typing import Any

MPS_DEVICE = "mps"
MPS_FALLBACK_ENV = "PYTORCH_ENABLE_MPS_FALLBACK"
SUPPORTED_QWEN_DTYPES = frozenset({"float16", "float32", "bfloat16"})


class QwenDeviceError(RuntimeError):
    """The requested Qwen execution device or dtype cannot be used."""


class QwenDeviceUnavailable(QwenDeviceError):
    """The requested accelerator is unavailable in this worker environment."""


class QwenDeviceMismatch(QwenDeviceError):
    """The loaded model is not executing on the requested device or dtype."""


def torch_dtype(torch: Any, dtype: str) -> Any:
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    if dtype == "bfloat16":
        return torch.bfloat16
    raise QwenDeviceError(f"unsupported Qwen dtype: {dtype}")


def validate_execution_request(torch: Any, *, device: str, dtype: str) -> Any:
    if dtype not in SUPPORTED_QWEN_DTYPES:
        raise QwenDeviceError(f"unsupported Qwen dtype: {dtype}")
    if device == "cpu":
        return torch.device("cpu")
    if device == MPS_DEVICE:
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_built() or not mps.is_available():
            raise QwenDeviceUnavailable("MPS is unavailable for this Qwen worker")
        if os.environ.get(MPS_FALLBACK_ENV, "0") not in {"", "0", "false", "False"}:
            raise QwenDeviceError(
                f"{MPS_FALLBACK_ENV} must disable CPU fallback for an MPS worker"
            )
        if dtype == "bfloat16":
            raise QwenDeviceError("Qwen MPS execution requires float16 or float32")
        return torch.device("mps")
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise QwenDeviceUnavailable("CUDA is unavailable for this Qwen worker")
        return torch.device(device)
    raise QwenDeviceError(f"unsupported Qwen device: {device}")


def _device_family(value: object) -> str:
    text = str(value)
    if text.startswith("mps"):
        return MPS_DEVICE
    if text.startswith("cuda"):
        return "cuda"
    if text.startswith("cpu"):
        return "cpu"
    return text


def _device_matches(expected: str, actual: object) -> bool:
    expected_family = _device_family(expected)
    actual_text = str(actual)
    if _device_family(actual_text) != expected_family:
        return False
    if expected_family == "cuda" and ":" in expected:
        return actual_text == expected
    return True


def assert_model_matches(*, model: Any, device: str, dtype: str) -> tuple[str, str]:
    actual_device = getattr(model, "device", None)
    if actual_device is None:
        try:
            actual_device = next(model.parameters()).device
        except (AttributeError, StopIteration) as exc:
            raise QwenDeviceMismatch("loaded Qwen model has no observable device") from exc
    actual_dtype = getattr(model, "dtype", None)
    if not _device_matches(device, actual_device):
        raise QwenDeviceMismatch(
            f"Qwen model loaded on {actual_device}, expected {device}"
        )
    expected_dtype = dtype
    observed_dtype = _dtype_name(actual_dtype)
    if observed_dtype != expected_dtype:
        raise QwenDeviceMismatch(
            f"Qwen model loaded with {observed_dtype}, expected {expected_dtype}"
        )
    return str(actual_device), observed_dtype


def _dtype_name(value: object) -> str:
    text = str(value)
    if text.endswith("float16"):
        return "float16"
    if text.endswith("float32"):
        return "float32"
    if text.endswith("bfloat16"):
        return "bfloat16"
    return text


__all__ = [
    "MPS_DEVICE",
    "MPS_FALLBACK_ENV",
    "QwenDeviceError",
    "QwenDeviceMismatch",
    "QwenDeviceUnavailable",
    "SUPPORTED_QWEN_DTYPES",
    "assert_model_matches",
    "torch_dtype",
    "validate_execution_request",
]
