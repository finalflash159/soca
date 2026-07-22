import numpy as np

from eval.eval_valtec_parity import compare_audio


def test_exact_audio_has_perfect_parity():
    audio = np.linspace(-0.5, 0.5, 2400, dtype=np.float32)
    metrics = compare_audio(
        audio,
        audio.copy(),
        torch_sample_rate=24000,
        onnx_sample_rate=24000,
        same_checkpoint=True,
    )
    assert metrics.sample_rate_match
    assert metrics.duration_ratio == 1.0
    assert metrics.waveform_mae == 0.0
    assert metrics.spectral_cosine > 0.999999


def test_different_checkpoint_does_not_claim_waveform_mae():
    torch_audio = np.ones(1000, dtype=np.float32)
    onnx_audio = np.ones(1020, dtype=np.float32)
    metrics = compare_audio(
        torch_audio,
        onnx_audio,
        torch_sample_rate=24000,
        onnx_sample_rate=24000,
        same_checkpoint=False,
    )
    assert metrics.duration_ratio == 1.02
    assert metrics.waveform_mae is None


def test_sample_rate_mismatch_is_explicit():
    audio = np.ones(100, dtype=np.float32)
    metrics = compare_audio(
        audio,
        audio,
        torch_sample_rate=22050,
        onnx_sample_rate=24000,
        same_checkpoint=True,
    )
    assert not metrics.sample_rate_match