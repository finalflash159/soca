from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from eval.eval_smart_turn_languages import (
    ArtifactIdentityError,
    ConfusionCounts,
    audio_array,
    evaluate_rows,
    verify_model_identity,
)


class _Detector:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = list(probabilities)
        self.batch_sizes: list[int] = []

    def p_complete_batch(self, audio_windows: list[np.ndarray]) -> np.ndarray:
        self.batch_sizes.append(len(audio_windows))
        batch = self.probabilities[: len(audio_windows)]
        del self.probabilities[: len(audio_windows)]
        return np.asarray(batch, dtype=np.float32)


def test_confusion_counts_match_upstream_total_denominators() -> None:
    counts = ConfusionCounts()
    for label, probability in ((True, 0.9), (True, 0.1), (False, 0.8), (False, 0.2)):
        counts.update(label=label, probability=probability)

    metrics = counts.metrics()

    assert metrics["sample_count"] == 4
    assert metrics["accuracy"] == pytest.approx(50.0)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["false_positive_rate"] == pytest.approx(25.0)
    assert metrics["false_negative_rate"] == pytest.approx(25.0)


def test_evaluate_rows_batches_and_reports_each_language() -> None:
    rows = [
        {
            "audio": {"array": np.full(160, index, dtype=np.float32), "sampling_rate": 16000},
            "endpoint_bool": label,
            "language": language,
            "dataset": "fixture",
        }
        for index, (language, label) in enumerate(
            (("vie", True), ("eng", False), ("vie", False), ("eng", True)), start=1
        )
    ]
    detector = _Detector([0.9, 0.2, 0.8, 0.1])

    report = evaluate_rows(rows, detector=detector, batch_size=2)

    assert detector.batch_sizes == [2, 2]
    assert report["overall"]["accuracy"] == pytest.approx(50.0)
    assert report["per_language"]["vie"]["sample_count"] == 2
    assert report["per_language"]["vie"]["accuracy"] == pytest.approx(50.0)
    assert report["per_language"]["eng"]["sample_count"] == 2
    assert report["per_dataset"]["fixture"]["sample_count"] == 4


def test_audio_array_supports_torchcodec_decoder_contract() -> None:
    decoded = SimpleNamespace(data=np.ones((1, 320), dtype=np.float32), sample_rate=16000)
    decoder = SimpleNamespace(get_all_samples=lambda: decoded)

    result = audio_array(decoder)

    assert result.shape == (320,)
    assert result.dtype == np.float32


def test_audio_array_rejects_wrong_sample_rate() -> None:
    with pytest.raises(ValueError, match="16 kHz"):
        audio_array({"array": np.zeros(80, dtype=np.float32), "sampling_rate": 8000})


def test_verify_model_identity_is_fail_closed(tmp_path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")

    with pytest.raises(ArtifactIdentityError, match="SHA-256"):
        verify_model_identity(model, expected_sha256="0" * 64)

    identity = verify_model_identity(
        model,
        expected_sha256="9372c470eeadd5ecd9c3c74c2b3cb633f8e2f2fad799250a0f70d652b6b825e4",
    )
    assert identity["bytes"] == 5
    assert identity["sha256"].startswith("9372c470")
