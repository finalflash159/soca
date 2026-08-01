"""Tests for the backend-agnostic calibrator refactor. VAD and ASR are both
injected via factories so these run without any real model or Silero VAD —
matching the plan's "no model needed, CI-safe" requirement."""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from local import config as cfg
from local.calibrate_asr_confidence import calibrate_model, main
from soca.asr.protocols import CalibratableASR
from soca.asr.vad import VADResult
from soca.asr.whisper_onnx import ASRResult

SAMPLE_RATE = 16_000


class _AlwaysSpeechVAD:
    """Fake VAD: every item is treated as speech, so ASR always runs and the
    test doesn't depend on Silero's actual behavior on synthetic audio."""

    threshold = 0.5
    min_speech_ms = 250
    min_silence_ms = 500
    speech_pad_ms = 200

    def detect(self, audio: np.ndarray) -> VADResult:
        duration_ms = len(audio) / SAMPLE_RATE * 1000
        return VADResult(
            has_speech=True,
            speech_audio=audio,
            speech_duration_ms=duration_ms,
            original_duration_ms=duration_ms,
            speech_ratio=1.0,
            vad_latency_ms=0.0,
            n_speech_segments=1,
        )


class _ScriptedASR:
    """Returns avg_logprob values from a fixed sequence, in call order.
    `calibrate_model` processes speech items before noise items, so the
    first N calls correspond to speech and the rest to noise."""

    model_key = "fake_backend"

    def __init__(self, avg_logprobs: list[float]):
        self._avg_logprobs = iter(avg_logprobs)

    def transcribe(self, audio: np.ndarray, max_new_tokens: int = 128) -> ASRResult:
        return ASRResult(
            text="a b c d",
            latency_ms=1.0,
            audio_duration_ms=len(audio) / SAMPLE_RATE * 1000,
            rtf=0.1,
            avg_logprob=next(self._avg_logprobs),
        )

    def runtime_metadata(self, max_new_tokens: int = 128) -> dict:
        return {"backend": "fake", "model_key": self.model_key, "max_new_tokens": max_new_tokens}


def test_scripted_asr_satisfies_calibratable_asr_protocol():
    assert isinstance(_ScriptedASR([]), CalibratableASR)


@pytest.mark.real_model
def test_real_vietnamese_asr_satisfies_calibratable_asr_protocol():
    from soca.asr.whisper_onnx import VietnameseASR

    asr = VietnameseASR(num_threads=2)
    assert isinstance(asr, CalibratableASR)


def _write_wav(path: Path, seconds: float = 0.3) -> None:
    n = int(SAMPLE_RATE * seconds)
    audio = (0.01 * np.sin(2 * np.pi * 440 * np.arange(n) / SAMPLE_RATE)).astype(np.float32)
    sf.write(str(path), audio, SAMPLE_RATE)


def _write_manifests(tmp_path: Path, *, n_speech: int, n_noise: int) -> None:
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()

    speech_lines = []
    for i in range(n_speech):
        filename = f"s{i}.wav"
        _write_wav(wav_dir / filename)
        speech_lines.append(f'{{"filename": "{filename}", "ground_truth": "x", "speaker_id": "spk"}}')
    (tmp_path / "fleurs_manifest.jsonl").write_text("\n".join(speech_lines) + "\n", encoding="utf-8")

    noise_lines = []
    for i in range(n_noise):
        filename = f"n{i}.wav"
        _write_wav(wav_dir / filename)
        noise_lines.append(f'{{"path": "{filename}", "source": "esc50", "label": "dog"}}')
    (tmp_path / "noise_manifest.jsonl").write_text("\n".join(noise_lines) + "\n", encoding="utf-8")


def test_calibrate_model_with_injected_factories_matches_hand_computed_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _write_manifests(tmp_path, n_speech=3, n_noise=2)
    monkeypatch.setattr(cfg, "FLEURS_MANIFEST", tmp_path / "fleurs_manifest.jsonl")
    monkeypatch.setattr(cfg, "FLEURS_WAV_DIR", tmp_path / "wav")
    monkeypatch.setattr(cfg, "NOISE_MANIFEST", tmp_path / "noise_manifest.jsonl")
    monkeypatch.setattr(cfg, "NOISE_ROOT", tmp_path / "wav")
    # calibrate_model writes to these unconditionally (audit log + shared
    # threshold file) — must redirect both, or this test pollutes the real
    # production calibration file as a side effect.
    monkeypatch.setattr(cfg, "THRESHOLD_CALIBRATION_PATH", tmp_path / "threshold_calibration.json")
    monkeypatch.setattr(cfg, "EVAL_RESULTS_DIR", tmp_path / "eval_results")

    speech_logprobs = [-0.10, -0.05, -0.20]
    noise_logprobs = [-0.90, -1.10]

    payload = calibrate_model(
        model_key="fake_backend",
        n_speech=3,
        n_noise=2,
        provider_list=[],
        max_new_tokens=64,
        fallback_min_avg_logprob=-0.25,
        fallback_max_compression_ratio=2.4,
        asr_factory=lambda: _ScriptedASR(speech_logprobs + noise_logprobs),
        vad_factory=_AlwaysSpeechVAD,
    )

    speech_p01 = float(np.percentile(np.array(speech_logprobs), 1))
    noise_max = float(max(noise_logprobs))
    expected_min_avg_logprob = (
        (noise_max + speech_p01) / 2 if noise_max < speech_p01 else speech_p01
    )
    assert payload["recommended_thresholds"]["min_avg_logprob"] == pytest.approx(
        expected_min_avg_logprob
    )
    assert payload["model_key"] == "fake_backend"
    assert payload["runtime_identity"]["asr"]["backend"] == "fake"
    assert payload["dataset"]["n_speech_loaded"] == 3
    assert payload["dataset"]["n_noise_loaded"] == 2


def test_calibrate_model_sanitizes_slash_in_model_key_for_the_output_filename(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """model_key can be a HF repo id like 'Qwen/Qwen3-ASR-0.6B'; '/' is not a
    valid filename component but IS a valid JSON object key, so only the
    audit-log filename needs sanitizing."""
    _write_manifests(tmp_path, n_speech=1, n_noise=0)
    monkeypatch.setattr(cfg, "FLEURS_MANIFEST", tmp_path / "fleurs_manifest.jsonl")
    monkeypatch.setattr(cfg, "FLEURS_WAV_DIR", tmp_path / "wav")
    monkeypatch.setattr(cfg, "NOISE_MANIFEST", tmp_path / "noise_manifest.jsonl")
    monkeypatch.setattr(cfg, "NOISE_ROOT", tmp_path / "wav")
    monkeypatch.setattr(cfg, "THRESHOLD_CALIBRATION_PATH", tmp_path / "threshold_calibration.json")
    eval_results_dir = tmp_path / "eval_results"
    monkeypatch.setattr(cfg, "EVAL_RESULTS_DIR", eval_results_dir)

    model_key = "Qwen/Qwen3-ASR-0.6B"
    calibrate_model(
        model_key=model_key,
        n_speech=1,
        n_noise=0,
        provider_list=[],
        max_new_tokens=64,
        fallback_min_avg_logprob=-0.25,
        fallback_max_compression_ratio=2.4,
        asr_factory=lambda: _ScriptedASR([-0.1]),
        vad_factory=_AlwaysSpeechVAD,
    )

    assert (eval_results_dir / "asr_confidence_calibration_Qwen__Qwen3-ASR-0.6B.json").is_file()
    merged = json.loads((tmp_path / "threshold_calibration.json").read_text(encoding="utf-8"))
    assert model_key in merged["asr_confidence_by_model"]


def test_asr_factory_is_not_called_when_manifest_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Manifest validation must happen before loading any model."""
    monkeypatch.setattr(cfg, "FLEURS_MANIFEST", tmp_path / "does_not_exist.jsonl")
    calls: list[int] = []

    def factory() -> CalibratableASR:
        calls.append(1)
        return _ScriptedASR([])

    with pytest.raises(click.ClickException):
        calibrate_model(
            model_key="fake_backend",
            n_speech=1,
            n_noise=1,
            provider_list=[],
            max_new_tokens=64,
            fallback_min_avg_logprob=-0.25,
            fallback_max_compression_ratio=2.4,
            asr_factory=factory,
        )
    assert calls == []


def test_whisper_onnx_backend_rejects_a_model_key_outside_the_registry():
    runner = CliRunner()
    result = runner.invoke(
        main, ["--backend", "whisper-onnx", "--model", "Qwen/Qwen3-ASR-0.6B"]
    )
    assert result.exit_code != 0
    assert "not in registry" in result.output.lower()


def test_qwen_backend_requires_a_model_key():
    runner = CliRunner()
    result = runner.invoke(main, ["--backend", "qwen"])
    assert result.exit_code != 0
    assert "--model is required" in result.output
