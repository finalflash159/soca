"""Qwen3-ASR backend, speaking the same interface as VietnameseASR.

Optional backend: `qwen_asr` import is deferred to __init__ so the rest of
the repo still works without the extra installed. See
zplan/qwen3_asr_probe_plan.vi.md §Q0 for why this needs an isolated venv.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from .whisper_onnx import ASRResult

DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
SAMPLING_RATE = 16_000
DEFAULT_EOS_TOKEN_IDS = (151645, 151643)


class QwenLogprobUnavailable(RuntimeError):
    """generate() did not return scores; confidence cannot be derived."""


def _mean_selected_logprob(
    scores: Sequence[Any], token_ids: Any, skip_ids: frozenset[int]
) -> tuple[float, bool]:
    """Mean log P of the tokens the model actually generated, plus whether
    that mean is meaningful.

    Same quantity as VietnameseASR's selected-token logprob: log-softmax over
    each step's logits, read at the generated token. Special tokens
    (EOS/pad) are excluded since they carry no signal about transcription
    quality.

    NOT on the same scale as Whisper's avg_logprob (different tokenizer,
    vocab, and decoder architecture) — thresholds must be calibrated
    separately per backend.
    """
    import torch

    total = 0.0
    counted = 0
    for step, step_scores in enumerate(scores):
        if step >= int(token_ids.shape[0]):
            break
        token = int(token_ids[step])
        if token in skip_ids:
            continue
        logprobs = torch.log_softmax(step_scores[0].float(), dim=-1)
        total += float(logprobs[token])
        counted += 1
    if counted == 0:
        # Every generated token was a skip_id: there is nothing to average.
        # 0.0 would read as maximum confidence to a naive consumer, so this
        # is flagged unreliable rather than left to look like a real score.
        return 0.0, False
    return total / counted, True


class QwenASRBackend:
    """Qwen3-ASR speaking the same protocol as VietnameseASR.

    `avg_logprob` is a real number, obtained via `output_scores=True` on a
    hand-rolled generate() call (§5.3.2) — not a fake 0.0. RobustASR's
    confidence guard therefore works normally, provided this backend's own
    threshold has been calibrated (§5.5); Whisper's threshold does not
    transfer.
    """

    BACKEND = "qwen3_asr"
    DECODE_STRATEGY = "llm_decoder"

    def __init__(
        self,
        model_id: str = DEFAULT_QWEN_MODEL_ID,
        *,
        context: str = "",
        language: str | None = "Vietnamese",
        device: str = "cpu",
        dtype: str = "float32",
        max_new_tokens: int = 256,
        require_logprob: bool = True,
    ) -> None:
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The Qwen3-ASR backend requires the `qwen-asr` package. It "
                "pins transformers==4.57.6, so it must NOT be installed into "
                "the main venv; see zplan/qwen3_asr_probe_plan.vi.md §Q0."
            ) from exc

        self.model_key = model_id
        self.context = context
        self.language = language
        self._device = device
        self._dtype = dtype
        self._max_new_tokens = max_new_tokens
        self._engine = Qwen3ASRModel.from_pretrained(
            model_id,
            dtype=torch.float32 if dtype == "float32" else torch.bfloat16,
            device_map=device,
            max_new_tokens=max_new_tokens,
        )

        eos = getattr(self._engine.model.generation_config, "eos_token_id", None)
        if eos is None:
            eos = list(DEFAULT_EOS_TOKEN_IDS)
        self._skip_ids = frozenset(eos if isinstance(eos, (list, tuple)) else [eos])

        # Measure capability with 200ms of silence rather than trusting
        # documentation — this is the actual answer to "does THIS runtime
        # give me scores", not an assumption (§5.3.3).
        probe = np.zeros(int(0.2 * SAMPLING_RATE), dtype=np.float32)
        try:
            self._transcribe_with_scores(probe, max_new_tokens)
            self.supports_avg_logprob = True
        except QwenLogprobUnavailable:
            if require_logprob:
                raise
            self.supports_avg_logprob = False

    def transcribe(self, audio: np.ndarray, max_new_tokens: int = 128) -> ASRResult:
        if audio.ndim != 1:
            raise ValueError(f"Audio must be 1D mono, got shape {audio.shape}")

        audio = audio.astype(np.float32, copy=False)
        audio_duration_ms = len(audio) / SAMPLING_RATE * 1000
        start = time.perf_counter()

        if self.supports_avg_logprob:
            text, avg_logprob, avg_logprob_reliable = self._transcribe_with_scores(
                audio, max_new_tokens
            )
        else:
            # Only reachable when the caller explicitly accepted
            # require_logprob=False. RobustASR reads supports_avg_logprob to
            # disable just the logprob guard (§5.3.1); the compression guard
            # still runs.
            results = self._engine.transcribe(
                audio=(audio, SAMPLING_RATE),
                context=self.context,
                language=self.language,
            )
            text = results[0].text.strip() if results else ""
            avg_logprob = 0.0
            avg_logprob_reliable = False

        latency_ms = (time.perf_counter() - start) * 1000
        return ASRResult(
            text=text,
            latency_ms=latency_ms,
            audio_duration_ms=audio_duration_ms,
            rtf=latency_ms / max(audio_duration_ms, 1.0),
            avg_logprob=avg_logprob,
            avg_logprob_reliable=avg_logprob_reliable,
        )

    def runtime_metadata(self, max_new_tokens: int = 128) -> dict[str, Any]:
        """Runtime identity recorded into ASR evaluation artifacts.

        `context` and `language` are part of the identity: they change the
        logprob distribution, so a threshold calibrated for one context does
        not carry over to another (§5.5.3).
        """
        return {
            "backend": self.BACKEND,
            "asr_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "model_key": self.model_key,
            "decode_strategy": self.DECODE_STRATEGY,
            "max_new_tokens": max_new_tokens,
            "device": self._device,
            "dtype": self._dtype,
            "sampling_rate": SAMPLING_RATE,
            "context": self.context,
            "language": self.language,
            "supports_avg_logprob": self.supports_avg_logprob,
        }

    def _transcribe_with_scores(
        self, audio: np.ndarray, max_new_tokens: int
    ) -> tuple[str, float, bool]:
        """Inference path that keeps the generation scores.

        Mirrors what Qwen3ASRModel._infer_asr_transformers does for batch=1,
        plus output_scores=True. Has to be hand-rolled because the public
        transcribe() does not forward kwargs down to generate().
        """
        from qwen_asr.inference.utils import parse_asr_output

        engine = self._engine
        model = engine.model
        processor = engine.processor

        prompt = engine._build_text_prompt(context=self.context, force_language=self.language)
        inputs = processor(text=[prompt], audio=[audio], return_tensors="pt", padding=True)
        inputs = inputs.to(model.device).to(model.dtype)

        # NOT passing return_dict_in_generate=True here: verified against the
        # real package that Qwen3ASRForConditionalGeneration.generate()
        # already hardcodes it internally before forwarding to
        # self.thinker.generate() — passing it again raises "got multiple
        # values for keyword argument 'return_dict_in_generate'" (§5.3.2).
        # The `output.scores is None` check below is the actual defense if
        # a future qwen-asr version stops forcing this.
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
        )
        if getattr(output, "scores", None) is None:
            # A typed failure, not a fake 0.0 (§5.3.3).
            raise QwenLogprobUnavailable(
                "generate() did not return scores; the installed qwen-asr "
                "version may have changed how kwargs reach generate(). "
                "Check modeling_qwen3_asr.generate()."
            )

        prompt_len = int(inputs["input_ids"].shape[1])
        generated = output.sequences[0, prompt_len:]
        raw = processor.batch_decode(
            [generated], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        _language, text = parse_asr_output(raw, user_language=self.language)
        avg_logprob, avg_logprob_reliable = _mean_selected_logprob(
            output.scores, generated, self._skip_ids
        )
        return text.strip(), avg_logprob, avg_logprob_reliable
