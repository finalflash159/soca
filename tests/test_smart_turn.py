from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from soca.core import smart_turn
from soca.core.smart_turn import SmartTurnDetector, _truncate_or_pad


def test_truncate_or_pad_keeps_last_8s_and_pads_at_front() -> None:
    cap = smart_turn._SR * smart_turn._N_SECONDS

    short = np.arange(5, dtype=np.float32)
    padded = _truncate_or_pad(short)
    assert len(padded) == cap
    assert np.all(padded[:-5] == 0)
    assert np.array_equal(padded[-5:], short)

    long_audio = np.arange(cap + 3, dtype=np.float32)
    truncated = _truncate_or_pad(long_audio)
    assert len(truncated) == cap
    assert np.array_equal(truncated, long_audio[-cap:])


def test_missing_model_raises_clear_download_message(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="download_smart_turn.py"):
        SmartTurnDetector(tmp_path)


def test_detector_builds_reference_features_and_returns_still_speaking(
    tmp_path, monkeypatch
) -> None:
    model_path = tmp_path / smart_turn._MODEL_FILE
    model_path.write_bytes(b"fake")
    captured: dict[str, object] = {}

    class FakeSessionOptions:
        __slots__ = ("inter_op_num_threads", "graph_optimization_level")

        def __init__(self) -> None:
            self.inter_op_num_threads = 0
            self.graph_optimization_level = None

    class FakeSession:
        def __init__(self, path, *, sess_options, providers):
            captured["path"] = path
            captured["options"] = sess_options
            captured["providers"] = providers

        def run(self, _outputs, inputs):
            captured["input_features"] = inputs["input_features"]
            return [np.array([[0.25]], dtype=np.float32)]

    class FakeFeatureExtractor:
        def __init__(self, *, chunk_length):
            captured["chunk_length"] = chunk_length

        def __call__(self, audio, **kwargs):
            captured["audio"] = audio
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                input_features=np.zeros((1, 80, 800), dtype=np.float32)
            )

    fake_ort = SimpleNamespace(
        SessionOptions=FakeSessionOptions,
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        InferenceSession=FakeSession,
    )
    fake_transformers = SimpleNamespace(WhisperFeatureExtractor=FakeFeatureExtractor)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    detector = SmartTurnDetector(tmp_path, providers=["CPUExecutionProvider"])
    p_still = detector.p_still_speaking(np.ones(160, dtype=np.float32))

    assert p_still == pytest.approx(0.75)
    assert captured["path"] == str(model_path)
    assert captured["providers"] == ["CPUExecutionProvider"]
    assert captured["chunk_length"] == smart_turn._N_SECONDS
    assert captured["input_features"].shape == (1, 80, 800)
    assert captured["kwargs"] == {
        "sampling_rate": smart_turn._SR,
        "return_tensors": "np",
        "padding": "max_length",
        "max_length": smart_turn._N_SECONDS * smart_turn._SR,
        "truncation": True,
        "do_normalize": True,
    }
    assert captured["options"].inter_op_num_threads == 1
