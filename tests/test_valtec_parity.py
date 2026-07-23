from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from eval.eval_valtec_parity import compare_audio
from eval.valtec_torch_reference import synthesize_torch_reference
from soca.tts.valtec.frontend import ValtecModelInputs


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


def test_torch_worker_receives_absolute_paths_before_cwd_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("G.pth").write_bytes(b"checkpoint")
    Path("config.json").write_text("{}", encoding="utf-8")
    (Path("external") / "src").mkdir(parents=True)

    def fake_run(command, *, cwd, env, check, capture_output, text):
        del env, check, capture_output, text
        assert Path(cwd).is_absolute()
        for flag in ("--checkpoint", "--config", "--inputs", "--output"):
            assert Path(command[command.index(flag) + 1]).is_absolute()
        output = Path(command[command.index("--output") + 1])
        sf.write(output, np.zeros(512, dtype=np.float32), 24000)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("eval.valtec_torch_reference.subprocess.run", fake_run)
    result = synthesize_torch_reference(
        ValtecModelInputs(
            phone_ids=(1, 2),
            tone_ids=(16, 16),
            language_ids=(7, 7),
            backend="test",
        ),
        speaker_id=0,
        checkpoint=Path("G.pth"),
        config=Path("config.json"),
        trust_checkpoint=True,
        source_root=Path("external"),
    )

    assert result.sample_rate == 24000
    assert result.audio.shape == (512,)
