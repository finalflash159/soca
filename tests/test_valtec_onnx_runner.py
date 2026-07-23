from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from soca.tts.valtec.frontend import ValtecModelInputs
from soca.tts.valtec.onnx_runner import ValtecOnnxTTS


class FakeFrontend:
    def prepare(self, text: str) -> ValtecModelInputs:
        assert text
        return ValtecModelInputs(
            phone_ids=(1, 2, 3),
            tone_ids=(16, 17, 18),
            language_ids=(7, 7, 7),
            backend="fake",
        )


def _artifact(root: Path, *, role: str = "reference") -> Path:
    root.mkdir()
    contents = {
        "text_encoder.onnx": b"encoder",
        "duration_predictor.onnx": b"duration",
        "flow.onnx": b"flow",
        "decoder.onnx": b"decoder",
        "tts_config.json": b"{}",
        "phoneme_dict.json": b"{}",
        "precomputed_latents.json": b"{}",
    }
    for relative, content in contents.items():
        (root / relative).write_bytes(content)
    manifest = {
        "schema_version": 1,
        "model_key": "valtec_multispeaker",
        "artifact_id": "upstream-reference",
        "role": role,
        "active_variant": "reference",
        "variants": {
            "reference": {
                "precision": "fp32",
                "runtime_graphs": {
                    "text_encoder": "text_encoder.onnx",
                    "duration_predictor": "duration_predictor.onnx",
                    "flow": "flow.onnx",
                    "decoder": "decoder.onnx",
                },
            }
        },
        "runtime_files": {
            "config": "tts_config.json",
        },
        "runtime_defaults": {
            "sample_rate": 24000,
            "hop_length": 512,
            "noise_scale": 0.0,
            "length_scale": 1.0,
            "tone_offset_vi": 16,
            "language_id_vi": 7,
            "add_blank": True,
        },
        "voices": {
            "map": {"NF": 0, "SF": 1, "NM1": 2, "SM": 3, "NM2": 4},
            "default": "NF",
        },
        "files": {
            relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in contents
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


class FakeSessionOptions:
    def __init__(self) -> None:
        self.inter_op_num_threads = 0
        self.graph_optimization_level = None


class FakeSession:
    created: list[FakeSession] = []

    def __init__(self, path, *, sess_options, providers) -> None:
        self.name = Path(path).stem
        self.providers = providers
        self.options = sess_options
        self.calls: list[dict[str, np.ndarray]] = []
        self.created.append(self)

    def run(self, _outputs, inputs):
        self.calls.append(inputs)
        if self.name == "text_encoder":
            assert inputs["phone_ids"].dtype == np.int64
            assert inputs["phone_ids"].shape == (1, 3)
            assert inputs["bert"].shape == (1, 1024, 3)
            assert inputs["ja_bert"].shape == (1, 768, 3)
            assert inputs["speaker_id"].tolist() == [0]
            x = np.zeros((1, 2, 3), dtype=np.float32)
            mask = np.ones((1, 1, 3), dtype=np.float32)
            g = np.zeros((1, 2, 1), dtype=np.float32)
            return x, x.copy(), x.copy(), mask, g
        if self.name == "duration_predictor":
            return [np.zeros((1, 1, 3), dtype=np.float32)]
        if self.name == "flow":
            return [inputs["z_p"]]
        if self.name == "decoder":
            frames = inputs["z"].shape[2]
            return [np.full((1, 1, frames * 512), 0.1, dtype=np.float32)]
        raise AssertionError(self.name)


@pytest.fixture
def fake_ort(monkeypatch):
    FakeSession.created.clear()
    module = types.SimpleNamespace(
        SessionOptions=FakeSessionOptions,
        GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL="all"),
        InferenceSession=FakeSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", module)
    return module


def _engine(tmp_path, fake_ort, **kwargs) -> ValtecOnnxTTS:
    return ValtecOnnxTTS(
        artifact_root=_artifact(tmp_path / "reference"),
        artifact_variant="reference",
        allow_reference=True,
        frontend=FakeFrontend(),
        seed=7,
        **kwargs,
    )


def test_synthesize_runs_four_graphs_with_expected_contract(tmp_path, fake_ort):
    engine = _engine(tmp_path, fake_ort)
    result = engine.synthesize("Xin chào")
    assert result.sample_rate == 24000
    assert result.voice == "NF"
    assert result.engine == "valtec-onnx"
    # Fake logw=0 predicts ~47 phones/s, so the pacing cap stretches
    # durations by MAX_ADAPTIVE_SCALE=1.4 -> ceil(1.4)=2 frames per phone.
    assert result.audio.shape == (6 * 512,)
    assert result.rtf >= 0.0
    assert engine.frontend_metadata == {"backend": "fake", "unknown_phoneme_count": 0}
    assert [session.name for session in FakeSession.created] == [
        "text_encoder",
        "duration_predictor",
        "flow",
        "decoder",
    ]


def test_sessions_are_loaded_once_and_unknown_voice_is_rejected(tmp_path, fake_ort):
    engine = _engine(tmp_path, fake_ort)
    engine.synthesize("Một")
    engine.synthesize("Hai")
    assert len(FakeSession.created) == 4
    with pytest.raises(ValueError, match="Unknown Valtec voice"):
        engine.synthesize("Ba", voice="not-a-voice")


def test_empty_text_returns_empty_result_without_frontend_call(tmp_path, fake_ort):
    engine = _engine(tmp_path, fake_ort)
    result = engine.synthesize("   ")
    assert result.audio.size == 0
    assert result.latency_ms == 0.0


def test_multi_sentence_text_is_chunked_with_inter_sentence_silence(tmp_path, fake_ort):
    engine = _engine(tmp_path, fake_ort)
    result = engine.synthesize("Câu một. Câu hai.")
    encoder_calls = [s for s in FakeSession.created if s.name == "text_encoder"]
    assert len(encoder_calls[0].calls) == 2
    silence = int(0.25 * 24000)
    assert result.audio.shape == (2 * 6 * 512 + silence,)


def test_sentence_chunking_can_be_disabled(tmp_path, fake_ort):
    engine = _engine(tmp_path, fake_ort, sentence_chunking=False)
    result = engine.synthesize("Câu một. Câu hai.")
    encoder_calls = [s for s in FakeSession.created if s.name == "text_encoder"]
    assert len(encoder_calls[0].calls) == 1
    assert result.audio.shape == (3 * 512,)


def test_long_sentence_is_subchunked_at_commas(tmp_path, fake_ort):
    engine = _engine(tmp_path, fake_ort)
    long_sentence = "Mô hình cần đọc " + "a " * 50 + "trước, rồi nghỉ ngắn ở vế sau này"
    assert len(long_sentence) > 100
    result = engine.synthesize(long_sentence)
    encoder_calls = [s for s in FakeSession.created if s.name == "text_encoder"]
    assert len(encoder_calls[0].calls) == 2
    clause_gap = int(0.15 * 24000)
    assert result.audio.shape == (2 * 6 * 512 + clause_gap,)


def test_chunk_edges_are_faded_to_avoid_clicks(tmp_path, fake_ort):
    engine = _engine(tmp_path, fake_ort)
    result = engine.synthesize("Câu một. Câu hai.")
    chunk = 6 * 512
    # FakeSession decoder emits constant 0.1; fades must ramp the edges down.
    assert abs(result.audio[0]) < 0.01
    assert abs(result.audio[chunk - 1]) < 0.01
    assert abs(result.audio[-1]) < 0.01


def test_pack_clauses_merges_only_short_clauses():
    from soca.tts.valtec.onnx_runner import _pack_clauses

    # Large clauses stay isolated; tiny ones merge forward, trailing tiny
    # clauses merge back into the previous chunk.
    packed = _pack_clauses(["a" * 60, "b" * 12, "c" * 12, "d" * 30, "e" * 5], 20)
    assert packed == ["a" * 60, "b" * 12 + " " + "c" * 12, "d" * 30 + " " + "e" * 5]


def test_match_loudness_levels_quiet_and_loud_chunks():
    from soca.tts.valtec.onnx_runner import _match_loudness

    quiet = np.full(1000, 0.05, dtype=np.float32)
    loud = np.full(1000, 0.2, dtype=np.float32)
    leveled_quiet, leveled_loud = _match_loudness([quiet, loud])
    assert leveled_quiet.max() > quiet.max()
    assert leveled_loud.max() < loud.max()


def test_duration_guard_prevents_unbounded_allocation(tmp_path, fake_ort):
    engine = _engine(tmp_path, fake_ort, max_audio_seconds=0.01)
    with pytest.raises(ValueError, match="predicted 6 frames"):
        engine.synthesize("Quá dài")


def test_candidate_requires_explicit_runtime_opt_in(tmp_path, fake_ort):
    root = _artifact(tmp_path / "candidate", role="candidate")

    with pytest.raises(ValueError, match="Candidate Valtec artifact is not allowed"):
        ValtecOnnxTTS(
            artifact_root=root,
            artifact_variant="reference",
            frontend=FakeFrontend(),
        )

    engine = ValtecOnnxTTS(
        artifact_root=root,
        artifact_variant="reference",
        allow_candidate=True,
        frontend=FakeFrontend(),
    )
    assert engine.synthesize("Xin chào").audio.size > 0


def test_import_does_not_load_torch_or_upstream_modules():
    code = (
        "import sys; import soca.tts.valtec.onnx_runner; "
        "assert 'torch' not in sys.modules; "
        "assert 'infer' not in sys.modules; "
        "assert 'viphoneme' not in sys.modules; "
        "assert 'vinorm' not in sys.modules"
    )
    completed = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
