import importlib
import sys
import warnings
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from soca.tts import TTSResult, VietnameseTTS


def test_tts_result_is_frozen():
    result = TTSResult(
        text="xin chào",
        audio=np.zeros(10, dtype=np.float32),
        sample_rate=24000,
        latency_ms=1.0,
        audio_duration_ms=10.0,
        rtf=0.1,
        voice="NF",
        engine="test",
    )

    with pytest.raises(FrozenInstanceError):
        result.text = "changed"


def test_empty_text_returns_empty_audio_after_eager_init(monkeypatch):
    monkeypatch.setattr(VietnameseTTS, "_ensure_loaded", lambda self: None)
    tts = VietnameseTTS()

    result = tts.synthesize("   ")

    assert result.text == ""
    assert result.audio.size == 0
    assert result.sample_rate == 24000
    assert result.rtf == 0.0


def test_vietnamese_tts_wraps_existing_model(monkeypatch):
    class FakeModel:
        def list_speakers(self):
            return ["NF", "SM"]

        def synthesize(self, text, speaker):
            assert text == "Xin chào"
            assert speaker == "NF"
            return np.ones(2400, dtype=np.float32) * 0.1, 24000

    monkeypatch.setattr(VietnameseTTS, "_ensure_loaded", lambda self: None)
    tts = VietnameseTTS()
    tts._model = FakeModel()

    result = tts.synthesize("Xin chào", voice="NF")

    assert result.text == "Xin chào"
    assert result.voice == "NF"
    assert result.engine == "valtec-tts"
    assert result.audio.dtype == np.float32
    assert result.audio.shape == (2400,)
    assert result.sample_rate == 24000
    assert result.audio_duration_ms == pytest.approx(100.0)


def test_vietnamese_tts_rejects_invalid_source_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="infer.py and src"):
        VietnameseTTS(source_dir=tmp_path)


def test_vietnamese_tts_accepts_valid_source_dir(tmp_path, monkeypatch):
    (tmp_path / "infer.py").write_text("", encoding="utf-8")
    (tmp_path / "src").mkdir()

    monkeypatch.setattr(VietnameseTTS, "_ensure_loaded", lambda self: None)
    tts = VietnameseTTS(source_dir=tmp_path)

    assert tts.source_dir == tmp_path.resolve()


def test_valtec_phonemizer_skips_viphoneme_when_vinorm_runtime_is_unsupported(
    monkeypatch, capsys
):
    source_dir = Path(__file__).resolve().parents[1] / "external" / "valtec-tts"
    monkeypatch.syspath_prepend(str(source_dir))
    phonemizer = importlib.import_module("src.vietnamese.phonemizer")

    def fail_if_called(text):
        pytest.fail(f"viphoneme should not run on unsupported vinorm runtime: {text}")

    monkeypatch.setattr(phonemizer, "_vinorm_runtime_supported", lambda: False)
    monkeypatch.setattr(phonemizer, "vi2IPA", fail_if_called)

    phones, tones, word2ph = phonemizer.text_to_phonemes_viphoneme("Xin chào")

    assert phones
    assert len(phones) == len(tones)
    assert sum(word2ph) == len(phones)
    assert "[WARN] Viphoneme failed" not in capsys.readouterr().out


def test_valtec_phonemizer_import_does_not_eagerly_import_vinorm(monkeypatch):
    source_dir = Path(__file__).resolve().parents[1] / "external" / "valtec-tts"
    monkeypatch.syspath_prepend(str(source_dir))
    sys.modules.pop("src.vietnamese.phonemizer", None)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        importlib.import_module("src.vietnamese.phonemizer")

    assert not any("imp module is deprecated" in str(w.message) for w in captured)
