"""Tests for the Qwen3-ASR backend. `qwen_asr` is stubbed via sys.modules so
these run in the main venv without the isolated .venv-qwen (torch itself is
already a main-venv dependency via the TTS stack)."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from soca.asr.qwen_backend import (
    QwenASRBackend,
    QwenLogprobUnavailable,
    _mean_selected_logprob,
)

# --- _mean_selected_logprob: pure function, no stubbing needed -------------


def test_mean_selected_logprob_averages_log_softmax_at_generated_tokens():
    step0 = torch.zeros((1, 5))
    step0[0, 2] = 10.0
    step1 = torch.zeros((1, 5))  # uniform -> log(1/5)
    scores = (step0, step1)
    token_ids = torch.tensor([2, 3])

    result, reliable = _mean_selected_logprob(scores, token_ids, skip_ids=frozenset())

    expected = (
        float(torch.log_softmax(step0[0], dim=-1)[2])
        + float(torch.log_softmax(step1[0], dim=-1)[3])
    ) / 2
    assert result == pytest.approx(expected, abs=1e-6)
    assert reliable is True


def test_mean_selected_logprob_excludes_skip_ids_from_the_average():
    step0 = torch.zeros((1, 5))
    step0[0, 1] = 10.0  # confident, kept
    step1 = torch.zeros((1, 5))  # uniform, would drag the average down if kept
    token_ids = torch.tensor([1, 99])  # 99 stands in for an EOS id

    result, reliable = _mean_selected_logprob((step0, step1), token_ids, skip_ids=frozenset({99}))

    only_step0 = float(torch.log_softmax(step0[0], dim=-1)[1])
    assert result == pytest.approx(only_step0, abs=1e-6)
    assert reliable is True


def test_mean_selected_logprob_flags_unreliable_when_every_token_is_skipped():
    """The dangerous edge case flagged in the plan: counted==0 would give
    0.0, the MOST confident value possible, if not flagged. `reliable=False`
    lets callers tell this apart from a real high-confidence score."""
    step0 = torch.zeros((1, 3))
    result, reliable = _mean_selected_logprob((step0,), torch.tensor([7]), skip_ids=frozenset({7}))
    assert result == 0.0
    assert reliable is False


# --- QwenASRBackend: stub qwen_asr via sys.modules --------------------------


class _FakeGenerateOutput:
    def __init__(self, sequences: torch.Tensor, scores):
        self.sequences = sequences
        self.scores = scores


class _FakeInputs(dict):
    def to(self, *_args, **_kwargs):
        return self


class _FakeProcessor:
    def __init__(self, decoded_text: str, prompt_len: int = 3):
        self._decoded_text = decoded_text
        self._prompt_len = prompt_len

    def __call__(self, *, text, audio, return_tensors, padding):
        input_ids = torch.zeros((1, self._prompt_len), dtype=torch.long)
        return _FakeInputs(input_ids=input_ids)

    def batch_decode(self, _sequences, **_kwargs):
        return [self._decoded_text]


class _FakeModel:
    def __init__(
        self,
        *,
        eos_token_id,
        scores_factory,
        generated_token_ids=None,
        return_plain_tensor=False,
    ):
        self.device = "cpu"
        self.dtype = torch.float32
        self.generation_config = SimpleNamespace(eos_token_id=eos_token_id)
        self._scores_factory = scores_factory
        self._generated_token_ids = generated_token_ids
        self._return_plain_tensor = return_plain_tensor
        self.last_generate_kwargs: dict | None = None

    def generate(self, **kwargs):
        self.last_generate_kwargs = kwargs
        prompt_len = int(kwargs["input_ids"].shape[1])
        scores = self._scores_factory()
        n_steps = len(scores) if scores is not None else 1
        tail = self._generated_token_ids or [0] * n_steps
        sequences = torch.zeros((1, prompt_len + n_steps), dtype=torch.long)
        sequences[0, prompt_len:] = torch.tensor(tail, dtype=torch.long)
        if self._return_plain_tensor:
            # Some runtimes return a bare tensor instead of a structured
            # object when return_dict_in_generate isn't honored — must not
            # crash with AttributeError, just fail the typed way.
            return sequences
        return _FakeGenerateOutput(sequences=sequences, scores=scores)


class _FakeEngine:
    def __init__(
        self,
        *,
        eos_token_id,
        scores_factory,
        decoded_text,
        fallback_text="",
        generated_token_ids=None,
        return_plain_tensor=False,
    ):
        self.model = _FakeModel(
            eos_token_id=eos_token_id,
            scores_factory=scores_factory,
            generated_token_ids=generated_token_ids,
            return_plain_tensor=return_plain_tensor,
        )
        self.processor = _FakeProcessor(decoded_text)
        self._fallback_text = fallback_text
        self.last_context: str | None = None

    def _build_text_prompt(self, *, context, force_language):
        self.last_context = context
        return f"[{context}][{force_language}]"

    def transcribe(self, *, audio, context, language):
        return [SimpleNamespace(text=self._fallback_text)]


def _install_fake_qwen_asr(monkeypatch, engine: _FakeEngine) -> None:
    fake_module = types.ModuleType("qwen_asr")

    class _FakeQwen3ASRModel:
        @staticmethod
        def from_pretrained(_model_id, **_kwargs):
            return engine

    fake_module.Qwen3ASRModel = _FakeQwen3ASRModel

    fake_inference = types.ModuleType("qwen_asr.inference")
    fake_utils = types.ModuleType("qwen_asr.inference.utils")
    fake_utils.parse_asr_output = lambda raw, *, user_language: (user_language, raw)
    fake_inference.utils = fake_utils

    monkeypatch.setitem(sys.modules, "qwen_asr", fake_module)
    monkeypatch.setitem(sys.modules, "qwen_asr.inference", fake_inference)
    monkeypatch.setitem(sys.modules, "qwen_asr.inference.utils", fake_utils)


def _one_confident_step_scores():
    step = torch.zeros((1, 10))
    step[0, 5] = 10.0
    return (step,)


def test_backend_uses_real_scores_when_generate_supports_output_scores(monkeypatch):
    engine = _FakeEngine(
        eos_token_id=[999],
        scores_factory=_one_confident_step_scores,
        decoded_text="xin chào",
        generated_token_ids=[5],
    )
    _install_fake_qwen_asr(monkeypatch, engine)

    backend = QwenASRBackend(context="tech context")

    assert backend.supports_avg_logprob is True

    result = backend.transcribe(np.zeros(1600, dtype=np.float32))

    assert result.text == "xin chào"
    expected_logprob = float(
        torch.log_softmax(_one_confident_step_scores()[0][0], dim=-1)[5]
    )
    assert result.avg_logprob == pytest.approx(expected_logprob, abs=1e-6)
    assert result.avg_logprob_reliable is True


def test_backend_uses_instance_context_by_default(monkeypatch):
    engine = _FakeEngine(
        eos_token_id=[999],
        scores_factory=_one_confident_step_scores,
        decoded_text="x",
        generated_token_ids=[5],
    )
    _install_fake_qwen_asr(monkeypatch, engine)

    backend = QwenASRBackend(context="tech context")
    backend.transcribe(np.zeros(1600, dtype=np.float32))

    assert engine.last_context == "tech context"


def test_backend_overrides_context_per_call_for_the_cheap_partial_path(monkeypatch):
    """context="" on a single call must not leak the instance's real context
    onto that call — this is what keeps the partial-caption path (§5.6.3)
    from ever showing the context-echo failure mode (§Q1b.3)."""
    engine = _FakeEngine(
        eos_token_id=[999],
        scores_factory=_one_confident_step_scores,
        decoded_text="x",
        generated_token_ids=[5],
    )
    _install_fake_qwen_asr(monkeypatch, engine)

    backend = QwenASRBackend(context="tech context")
    backend.transcribe(np.zeros(1600, dtype=np.float32), context="")

    assert engine.last_context == ""

    # The instance's own context is untouched for the next (final) call.
    backend.transcribe(np.zeros(1600, dtype=np.float32))
    assert engine.last_context == "tech context"


def test_backend_honors_per_call_max_new_tokens_not_just_the_constructor_default(monkeypatch):
    engine = _FakeEngine(
        eos_token_id=[999],
        scores_factory=_one_confident_step_scores,
        decoded_text="xin chào",
        generated_token_ids=[5],
    )
    _install_fake_qwen_asr(monkeypatch, engine)

    backend = QwenASRBackend(max_new_tokens=256)
    backend.transcribe(np.zeros(1600, dtype=np.float32), max_new_tokens=64)

    assert engine.model.last_generate_kwargs["max_new_tokens"] == 64


def test_backend_raises_at_init_when_scores_unavailable_and_logprob_required(monkeypatch):
    engine = _FakeEngine(eos_token_id=[999], scores_factory=lambda: None, decoded_text="x")
    _install_fake_qwen_asr(monkeypatch, engine)

    with pytest.raises(QwenLogprobUnavailable):
        QwenASRBackend(require_logprob=True)


def test_backend_raises_when_generate_returns_a_plain_tensor_instead_of_scores(monkeypatch):
    """Some runtimes silently ignore return_dict_in_generate and hand back a
    bare tensor. Must fail the typed way (QwenLogprobUnavailable), not crash
    with an unrelated AttributeError."""
    engine = _FakeEngine(
        eos_token_id=[999],
        scores_factory=_one_confident_step_scores,
        decoded_text="x",
        return_plain_tensor=True,
    )
    _install_fake_qwen_asr(monkeypatch, engine)

    with pytest.raises(QwenLogprobUnavailable):
        QwenASRBackend(require_logprob=True)


def test_backend_falls_back_when_scores_unavailable_and_logprob_not_required(monkeypatch):
    engine = _FakeEngine(
        eos_token_id=[999],
        scores_factory=lambda: None,
        decoded_text="x",
        fallback_text="fallback transcript",
    )
    _install_fake_qwen_asr(monkeypatch, engine)

    backend = QwenASRBackend(require_logprob=False)

    assert backend.supports_avg_logprob is False

    result = backend.transcribe(np.zeros(1600, dtype=np.float32))

    assert result.text == "fallback transcript"
    assert result.avg_logprob == 0.0
    assert result.avg_logprob_reliable is False


def test_runtime_metadata_records_context_and_language(monkeypatch):
    engine = _FakeEngine(
        eos_token_id=[999], scores_factory=_one_confident_step_scores, decoded_text="x"
    )
    _install_fake_qwen_asr(monkeypatch, engine)

    backend = QwenASRBackend(model_id="Qwen/Qwen3-ASR-1.7B", context="tech", language="Vietnamese")
    meta = backend.runtime_metadata()

    assert meta["backend"] == "qwen3_asr"
    assert meta["model_key"] == "Qwen/Qwen3-ASR-1.7B"
    assert meta["context"] == "tech"
    assert meta["language"] == "Vietnamese"
    assert meta["supports_avg_logprob"] is True


@pytest.mark.real_model
def test_real_qwen_backend_produces_a_strictly_negative_logprob():
    """The one test no stub can fake: catches output_scores=True silently
    ceasing to flow through after a qwen-asr upgrade. Run from .venv-qwen:
        .venv-qwen/bin/python -m pytest tests/test_qwen_backend.py -m real_model
    """
    pytest.importorskip("qwen_asr")
    backend = QwenASRBackend()
    audio = np.random.default_rng(0).uniform(-0.1, 0.1, size=16_000).astype(np.float32)
    result = backend.transcribe(audio)
    assert result.avg_logprob < 0.0
