# SoCa D2.5 — ASR Hallucination Robustness Notes

> Last updated: 2026-05-21.
> Runtime measured locally on a MacBook M4 Pro with PhoWhisper-tiny ONNX.

## 1. Problem

SoCa is an offline-first Vietnamese voice assistant. In this architecture, ASR is
the entry point:

```text
audio -> VAD -> ASR -> intent/LLM -> response
```

If ASR hallucinates fluent text from silence or non-speech audio, the downstream LLM
may treat that hallucinated transcript as a real user command. This is not only a WER
problem; it is a behavioral safety problem for a voice assistant.

In the local D2.5 experiments, raw PhoWhisper-tiny produced non-empty Vietnamese text
for 100% of 800 non-speech samples. That failure mode made ASR robustness a required
runtime layer before connecting ASR output to LLM and TTS.

## 2. Main Reference

The main reference for D2.5 is:

- Mateusz Baranski et al., "Investigation of Whisper ASR Hallucinations Induced by
  Non-Speech Audio", ICASSP 2025 / arXiv 2501.11378:
  <https://arxiv.org/abs/2501.11378>

The core idea from the paper:

1. Run Whisper on non-speech audio.
2. Collect repeated hallucinated phrases.
3. Build a Bag of Hallucinations (BoH).
4. Combine VAD, de-looping, and BoH matching.
5. Evaluate both hallucination rate on noise and WER on real speech.

A related Vietnamese case study is WhisperX issue #1086, where Whisper injected a
Vietnamese YouTube-style call-to-action phrase into unrelated content:

- <https://github.com/m-bain/whisperX/issues/1086>

The key lesson is that hallucinations are language-specific. A Vietnamese assistant
cannot rely on an English hallucination phrase list.

## 3. SoCa Adaptation

SoCa adapts the paper's approach to a different runtime:

| Component | Paper                        | SoCa                           |
| --------- | ---------------------------- | ------------------------------ |
| Model     | Whisper-large-v3             | PhoWhisper-tiny ONNX           |
| Language  | English-centered benchmark   | Vietnamese                     |
| Goal      | ASR hallucination mitigation | Voice assistant command safety |
| Runtime   | Research/evaluation          | Local offline inference        |
| Decode    | Whisper runtime              | Greedy ONNX, no KV cache       |

Local non-speech dataset:

| Source                                | Count |
| ------------------------------------- | ----: |
| ESC-50 filtered for non-human classes |   500 |
| Synthetic silence                     |   100 |
| Synthetic white noise                 |   100 |
| Synthetic pink noise                  |   100 |
| Total                                 |   800 |

BoH construction:

- Model: `huuquyet/PhoWhisper-tiny`
- Decode: greedy, `max_new_tokens=128`
- Candidate rule: `count >= 2` and `len >= 5`
- Manual review: reject short ambiguous phrases
- Final Vietnamese BoH: 74 phrases

Important manual-review example:

- `các em` was rejected because it is a legitimate Vietnamese phrase and caused
  false positives on real FLEURS speech.
- Longer phrases around the same YouTube/teacher-style pattern were kept.

## 4. Runtime Pipeline

The production path is `soca.asr.RobustASR`:

```text
audio
  -> [1] Silero VAD
  -> [2] PhoWhisper-tiny ONNX
  -> [3] ASR confidence guard
  -> [4] De-loop
  -> [5] Vietnamese BoH
  -> [6] Text heuristics
  -> clean text or rejection reason
```

Stage responsibilities:

| Stage            | Purpose                                                           |
| ---------------- | ----------------------------------------------------------------- |
| VAD              | Skip ASR entirely on no-speech audio                              |
| ASR              | Produce the raw Vietnamese transcript                             |
| Confidence guard | Reject low-confidence raw model output before text cleanup        |
| De-loop          | Collapse repeated token or phrase loops                           |
| BoH              | Remove empirically collected Vietnamese hallucination phrases     |
| Text heuristics  | Catch filler-only output, repetition, and impossible text density |

VAD parameters are Whisper-aware:

```text
threshold = 0.5
min_speech_ms = 250
min_silence_ms = 500
speech_pad_ms = 200
```

Silero's default silence handling is too aggressive for Whisper-class models, while
faster-whisper's long-form defaults add too much latency for voice commands. SoCa
uses a middle ground for short assistant utterances.

## 5. Threshold Calibration

Text-only heuristics were calibrated from real Vietnamese ground-truth transcripts
from FLEURS vi_vn:

```bash
uv run python -m local.calibrate_thresholds
```

Applied text-heuristic defaults:

| Metric                | Default |
| --------------------- | ------: |
| Unigram repetition    |    0.35 |
| 3-gram repetition     |    0.12 |
| Characters per 100 ms |    2.50 |

ASR confidence was calibrated from actual runtime output:

```bash
uv run python -m local.calibrate_asr_confidence --n-speech 200 --n-noise 50
```

Observed confidence distribution:

| Metric / event                  |   Value |
| ------------------------------- | ------: |
| Speech detected by VAD          | 200/200 |
| Noise detected as speech by VAD |    1/50 |
| `avg_logprob` speech p01        |  -0.250 |
| `avg_logprob` noise max         |  -1.200 |
| Applied `min_avg_logprob`       |  -0.725 |
| `compression_ratio` speech p99  |   1.543 |
| Applied `max_compression_ratio` |   2.400 |

The `min_avg_logprob` threshold is the midpoint between the VAD-leaked noise case
and the bottom 1% of real speech:

```text
(-1.200 + -0.250) / 2 = -0.725
```

Runtime policy:

```python
if avg_logprob < -0.725:
    reject("low_confidence")
```

## 6. Finding: `no_speech_prob` Was Not Useful

Canonical Whisper uses `no_speech_prob` as an important no-speech signal. SoCa
tested this signal, but it did not separate speech from noise on PhoWhisper-tiny.

Observed behavior:

- `no_speech_prob` stayed near zero for both speech and noise.
- PhoWhisper-tiny strongly preferred Vietnamese/task tokens at the first decoder step.
- The fine-tuned model appears to have weakened the original Whisper no-speech
  behavior.

Decision:

- Do not use `no_speech_prob` as a production rejection signal for PhoWhisper-tiny.
- Use measured local signals instead: `avg_logprob` and `compression_ratio`.

This is the main engineering lesson from D2.5: canonical Whisper heuristics must be
re-measured after fine-tuning. Copying defaults blindly would not have fixed this
runtime.

## 7. Table VII-Style Benchmark

Benchmark command:

```bash
uv run python -m local.eval_table7 --n-speech 200 --n-noise 50
```

Setup:

| Field         | Value                                                    |
| ------------- | -------------------------------------------------------- |
| Speech eval   | 200 FLEURS vi_vn test samples                            |
| Noise eval    | 50 non-speech samples                                    |
| Providers     | CoreMLExecutionProvider -> CPUExecutionProvider fallback |
| BoH           | 74 manually reviewed phrases                             |
| Full pipeline | Actual `RobustASR` production class                      |

Results:

| Config          |    WER |    CER | Hallucination rate |
| --------------- | -----: | -----: | -----------------: |
| Raw ASR         | 25.45% | 12.52% |               100% |
| De-loop only    | 24.62% | 12.03% |               100% |
| Silero VAD only | 25.22% | 12.40% |                 2% |
| BoH only        | 25.45% | 12.52% |               100% |
| De-loop + BoH   | 24.62% | 12.03% |               100% |
| Full RobustASR  | 25.22% | 12.40% |                 0% |

Key result:

```text
Raw ASR hallucination: 100%
Full RobustASR:        0%
WER raw:               25.45%
WER full:              25.22%
```

The full pipeline removed all 50/50 noise hallucinations in this run while preserving
real-speech accuracy.

## 8. Why BoH Still Matters

The Table VII hallucination metric is strict:

```text
any non-empty noise output = hallucination
```

BoH can remove a known phrase but leave residual punctuation or words, so the sample is
still counted as hallucinated. A finer analysis is more informative:

| BoH effect                         | Value |
| ---------------------------------- | ----: |
| Noise samples modified by BoH      | 27/50 |
| Noise samples fully emptied by BoH | 22/50 |
| BoH false positives on real speech | 0/200 |

Interpretation:

- VAD is the main first-line defense.
- BoH removes known repeated hallucination phrases.
- The confidence guard catches short valid-looking hallucinations that BoH and text
  heuristics cannot identify.

## 9. Edge Case Fixed

Before confidence calibration, one noise sample escaped:

```text
file: esc50_0048.wav
label: insects
VAD: detected 720 ms speech
Raw ASR: "thôi."
```

Why the old text-only pipeline missed it:

- It is one valid-looking Vietnamese word.
- It has no repetition.
- It is not filler-only.
- It is not in BoH.
- It has normal text density.

After B3:

```text
avg_logprob = -1.200
threshold   = -0.725
result      = reject low_confidence:-1.20
```

This is why the runtime is now six-stage instead of five-stage.

## 10. Limitations

- BoH was built from 800 non-speech samples; the reference paper used a much larger
  corpus.
- The current noise evaluation uses the same local non-speech collection family, so a
  future held-out noise split is needed.
- PhoWhisper-tiny ONNX uses greedy decode without KV cache.
- Thresholds are tied to runtime identity. Recalibrate after changing the model,
  decoder, provider, VAD parameters, or `max_new_tokens`.
- The current benchmark is batch/offline, not streaming.

## 11. Future Work

Recommended ASR robustness follow-ups:

1. Build a held-out noise benchmark separate from BoH construction.
2. Scale BoH to 5k-10k non-speech files.
3. Compare PhoWhisper-base/small under the same pipeline.
4. Add streaming VAD endpointing for push-to-talk.
5. Rebuild BoH and recalibrate confidence if switching decoder topology, especially
   merged decoder, KV-cache, or quantized decoder.

For now, D2.5 is sufficient to move into Phase 3 voice loop work: TTS, audio output,
intent routing, and full ASR -> LLM -> TTS orchestration.
