# SoCa — Benchmarks

> Last updated: 2026-07-28. All measurements are from real local runs;
> raw output JSON lives under `eval/results/` (gitignored).

## Common hardware

Unless noted otherwise:

- MacBook M4 Pro, macOS arm64
- Python 3.11.14 (`uv`-managed)
- ONNX Runtime providers available: `CoreMLExecutionProvider`, `CPUExecutionProvider`
- D2.5 runs use CoreML with CPU fallback (some ONNX nodes do not support CoreML — runtime auto-falls back per node, no manual partitioning)
- D1 baseline ran on `CPUExecutionProvider` only (`intra_op_num_threads=4`)
- D2 LLM uses Apple Metal via `llama-cpp-python` (CMake `-DGGML_METAL=on`)

---

## D1 — ASR Baseline (PhoWhisper-tiny ONNX)

**Dataset note:** script `eval/download_common_voice_vi.py` was renamed but
**actually downloads from `google/fleurs` config `vi_vn` split `test`** —
Common Voice 17 stopped exposing loadable files for the current `datasets`
loader, so D1 uses FLEURS Vietnamese instead. The WER number below is
on FLEURS, **not** Common Voice VI.

**Setup**

| Field              | Value                                                                       |
| ------------------ | --------------------------------------------------------------------------- |
| Model              | `huuquyet/PhoWhisper-tiny` (39M params)                                     |
| Encoder file       | `onnx/encoder_model.onnx`                                                   |
| Decoder file       | `onnx/decoder_model.onnx`                                                   |
| KV cache           | Disabled (recompute full decoder per step)                                  |
| Decode strategy    | Greedy (argmax), `max_new_tokens=128`                                       |
| Forced decoder IDs | `[<\|startoftranscript\|>, <\|vi\|>, <\|transcribe\|>, <\|notimestamps\|>]` |
| Token suppression  | From `generation_config.json` (`suppress_tokens` + `begin_suppress_tokens`) |
| Providers          | `CPUExecutionProvider` only, `intra_op_num_threads=4`                       |
| Eval dataset       | `google/fleurs` vi_vn test, first 100 streamed samples                      |
| WER normalization  | `lower().strip()` on both ref and hyp before `jiwer.wer`                    |
| Run date           | 2026-05-14                                                                  |
| Output path        | `eval/results/asr_d1_baseline.json` (gitignored)                            |

**Results (100 samples, computed from saved per-sample data)**

| Metric           | Value    | Notes                                       |
| ---------------- | -------- | ------------------------------------------- |
| WER              | 23.60%   | Paper PhoWhisper baseline: 19.05% (beam-5)  |
| CER              | 12.44%   |                                             |
| Avg latency      | 635.8 ms | Range 221 – 1463 ms (varies with audio len) |
| Avg audio length | 12.93 s  | FLEURS samples are longer than CMV          |
| Avg RTF          | 0.0492   | ~20× faster than realtime                   |

**Caveats**

- Greedy + no KV cache: each decoder step re-runs the full graph. Latency
  scales O(N²) in output length.
- Gap to paper WER ≈ greedy vs beam-5; the goal here is edge-deployable
  decoder topology, not paper-matching eval setup.

---

## D1.1 — ASR Model-Size Bake-Off (PhoWhisper ONNX)

**Purpose:** check whether moving beyond `PhoWhisper-tiny` is worth it before
changing the default ASR model.

**Setup**

| Field                | Value                                                                            |
| -------------------- | -------------------------------------------------------------------------------- |
| Runtime              | `soca.asr.VietnameseASR` with model registry selection                           |
| Decode               | Same greedy autoregressive decoder, no KV cache                                  |
| Providers            | `CoreMLExecutionProvider`, `CPUExecutionProvider` fallback                       |
| Eval data            | FLEURS Vietnamese local slice                                                    |
| Run date             | 2026-05-27                                                                       |
| Main output          | `eval/results/asr_bakeoff_profile_bakeoff_n20_20260527_014702.json` (gitignored) |
| Medium sanity output | `eval/results/asr_bakeoff_profile_full_n5_20260527_015434.json` (gitignored)     |

**Main result: tiny/base/small on 20 FLEURS samples**

| Model              | Params |    WER |   CER | Avg RTF | P95 RTF | P50 latency | P95 latency |
| ------------------ | -----: | -----: | ----: | ------: | ------: | ----------: | ----------: |
| `phowhisper_tiny`  |    39M | 15.58% | 6.96% |   0.118 |   0.161 |     1342 ms |     1967 ms |
| `phowhisper_base`  |    74M | 11.78% | 5.45% |   0.230 |   0.318 |     2709 ms |     3855 ms |
| `phowhisper_small` |   244M | 10.33% | 5.12% |   0.647 |   0.902 |     7695 ms |    10583 ms |

**Medium sanity check: 5 FLEURS samples**

| Model               | Params |   WER |   CER | Avg RTF | P95 RTF | P50 latency | P95 latency |
| ------------------- | -----: | ----: | ----: | ------: | ------: | ----------: | ----------: |
| `phowhisper_medium` |   769M | 5.88% | 1.52% |   1.744 |   1.967 |    15222 ms |    16734 ms |

**Decision**

- `phowhisper_base` is the best next ASR candidate to test in the voice loop:
  it improves WER meaningfully over tiny on the 20-sample slice while keeping
  RTF comfortably below real-time.
- `phowhisper_small` improves WER further, but current greedy/no-cache decode
  pushes p95 latency above 10 seconds on this slice. It is useful for quality
  comparison, not as the default real-time ASR path yet.
- `phowhisper_medium` loads and runs, but is already slower than real-time with
  the current decoder path. Treat it as a quality-ceiling probe until KV-cache,
  merged decoder, quantized decoder, or a different runtime is implemented.
- Confidence thresholds for `phowhisper_base` and `phowhisper_small` have now
  been recalibrated for the current voice-loop candidates.
- `phowhisper_base` and `phowhisper_small` now both have model-specific BoH
  artifacts, so the current voice-loop ASR candidates no longer depend on the
  tiny BoH profile.

**Caveats**

- These are small local slices, not final ASR claims.
- The first 5 FLEURS samples are not representative enough to rank medium
  quality; they only show runtime feasibility and approximate latency.
- Larger PhoWhisper models may change hallucination behavior, so RobustASR
  thresholds and BoH must be recalibrated before switching production ASR.

---

## D1.2 — ASR Confidence Guard Recalibration (PhoWhisper base/small)

**Purpose:** remove the old `confidence=model_mismatch:profile=phowhisper_tiny`
diagnostic in runtime profiles that now use `phowhisper_base` and
`phowhisper_small`.

**Setup**

| Field            | Value                                                                                                                                           |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Runtime          | `local.calibrate_asr_confidence` + `soca.asr.VietnameseASR`                                                                                     |
| Speech data      | 200 FLEURS Vietnamese samples (`google/fleurs`, `vi_vn`, `test`)                                                                                |
| Noise data       | first 50 rows from `data/noise_for_boh/manifest.jsonl`                                                                                          |
| Providers        | `CoreMLExecutionProvider`, `CPUExecutionProvider` fallback                                                                                      |
| Decode           | Greedy, no KV cache, `max_new_tokens=128`                                                                                                       |
| Command          | `uv run python -m local.calibrate_asr_confidence --model phowhisper_base --model phowhisper_small --n-speech 200 --n-noise 50 --providers auto` |
| Run date         | 2026-06-03                                                                                                                                      |
| Raw outputs      | `eval/results/asr_confidence_calibration_phowhisper_{base,small}.json` (gitignored)                                                             |
| Runtime artifact | `data/asr/threshold_calibration.json` (gitignored local artifact)                                                                               |

**Results**

| Model              | Speech ASR rows | Noise ASR rows | min avg_logprob | max compression | Rule                                      |
| ------------------ | --------------: | -------------: | --------------: | --------------: | ----------------------------------------- |
| `phowhisper_base`  |             200 |              1 |          -0.598 |             2.4 | midpoint between noise max and speech p01 |
| `phowhisper_small` |             200 |              1 |          -0.260 |             2.4 | midpoint between noise max and speech p01 |

**Runtime decision**

- `RobustASR` now loads confidence thresholds from
  `data/asr/threshold_calibration.json` by exact ASR `model_key`.
- If a matching calibration is missing, the confidence guard is skipped with
  `confidence=skipped:missing_for_model:<model>`. This is intentional and safer
  than applying a tiny calibration to base/small.
- Expected guard status after this run:
  - `phowhisper_base`: `confidence=enabled:phowhisper_base`
  - `phowhisper_small`: `confidence=enabled:phowhisper_small`
- BoH is separate from confidence calibration. After D1.3 and D1.4, expected
  runtime guard status is:
  - `phowhisper_base`: `BoH=loaded:phowhisper_base`
  - `phowhisper_small`: `BoH=loaded:phowhisper_small`

---

## D1.3 — ASR BoH Build (PhoWhisper base)

**Purpose:** build a model-specific Bag-of-Hallucinations artifact for
`phowhisper_base`. The tiny BoH cannot be safely reused because hallucination
phrases are model-specific.

**Setup**

| Field            | Value                                                                     |
| ---------------- | ------------------------------------------------------------------------- |
| Runtime          | `local.build_boh` + `soca.asr.VietnameseASR`                              |
| Model            | `huuquyet/PhoWhisper-base` (74M params)                                   |
| Noise data       | 800 rows from `data/noise_for_boh/manifest.jsonl`                         |
| Providers        | `CoreMLExecutionProvider`, `CPUExecutionProvider` fallback                |
| Decode           | Greedy, no KV cache, `max_new_tokens=128`                                 |
| Selection rule   | `count >= 2` and normalized phrase length `>= 5` chars                    |
| Run date         | 2026-06-03                                                                |
| Runtime artifact | `data/asr/boh/phowhisper_base_vi_boh_v1.json` (gitignored local artifact) |

**Results**

| Metric            | Value              |
| ----------------- | ------------------ |
| Noise samples     | 800                |
| Non-empty outputs | 800/800 (100.00%)  |
| BoH candidates    | 56                 |
| Errors            | 0                  |
| Elapsed           | 1124 s (18 m 44 s) |

**Top BoH candidates**

| Rank | Count | Phrase                                                                        |
| ---: | ----: | ----------------------------------------------------------------------------- |
|    1 |   100 | tuy nhiên khi thiên tai mục không thể thiên tai nạn được                      |
|    2 |    32 | đây là một trong những thứ mà trẻ em không thể thiếu                          |
|    3 |    22 | đây là một trong những thứ mà trẻ em không thể thiếu được                     |
|    4 |    13 | đây là một trong những thông tin trong việc thiết bị chung                    |
|    5 |    12 | đây là một trong những thứ trưởng quan trọng nhất trong trận đấu              |
|    6 |    11 | tuy nhiên không phải là điều mà trước mắt                                     |
|    7 |     9 | điều này đã tạo nên sự nghiệp rất quan trọng                                  |
|    8 |     9 | các thiết bị chất lượng từ thiện tại nhà thi đấu tranh tính trên thế giới     |
|    9 |     9 | đây là một trong những thứ trưởng quan trọng                                  |
|   10 |     8 | các thiết bị chất lượng tăng trưởng thành tăng trưởng thành tăng trưởng thành |

**Runtime decision**

- `phowhisper_base` now has both calibration layers available:
  - `confidence=enabled:phowhisper_base`
  - `BoH=loaded:phowhisper_base`
- The 100% non-empty rate on non-speech confirms that `phowhisper_base` still
  hallucinates aggressively without RobustASR guards.
- `phowhisper_small` has its own BoH artifact in D1.4. Do not reuse the base
  artifact for small, even if some repeated phrases look similar.

---

## D1.4 — ASR BoH Build (PhoWhisper small)

**Purpose:** build a model-specific Bag-of-Hallucinations artifact for
`phowhisper_small`, completing the RobustASR calibration pair for the current
base/quality ASR candidates.

**Setup**

| Field            | Value                                                                      |
| ---------------- | -------------------------------------------------------------------------- |
| Runtime          | `local.build_boh` + `soca.asr.VietnameseASR`                               |
| Model            | `huuquyet/PhoWhisper-small` (244M params)                                  |
| Noise data       | 800 rows from `data/noise_for_boh/manifest.jsonl`                          |
| Providers        | `CoreMLExecutionProvider`, `CPUExecutionProvider` fallback                 |
| Decode           | Greedy, no KV cache, `max_new_tokens=128`                                  |
| Selection rule   | `count >= 2` and normalized phrase length `>= 5` chars                     |
| Run date         | 2026-06-03                                                                 |
| Runtime artifact | `data/asr/boh/phowhisper_small_vi_boh_v1.json` (gitignored local artifact) |

**Results**

| Metric            | Value              |
| ----------------- | ------------------ |
| Noise samples     | 800                |
| Non-empty outputs | 800/800 (100.00%)  |
| BoH candidates    | 37                 |
| Errors            | 0                  |
| Elapsed           | 2831 s (47 m 10 s) |

**Top BoH candidates**

| Rank | Count | Phrase                                                                |
| ---: | ----: | --------------------------------------------------------------------- |
|    1 |   146 | tuy nhiên không phải ai cũng có thể thực hiện điều này                |
|    2 |     9 | đó là một cái nhìn rất mạnh lên từng phút                             |
|    3 |     8 | đó là tin nhắn của anh chị nhiều khi tưởng chừng đó                   |
|    4 |     6 | một trong những thay đổi lĩnh vực sân khấu của trẻ là giải đấu chung  |
|    5 |     5 | chúng                                                                 |
|    6 |     5 | tuy nhiên không phải ai cũng có thể tham gia các cuộc thi đấu thứ hai |
|    7 |     5 | điều này đã khiến nhiều người mơ ước đối với nghề đam mê              |
|    8 |     5 | tuy nhiên không phải lúc nào bạn cũng có thể thay tóa những vết nứt   |
|    9 |     4 | đó là một cái nhìn rất mạnh lên từng hồi sinh                         |
|   10 |     3 | điều này đã khiến nhiều người bình luận vì thế đế chất lượng thấp     |

**Runtime decision**

- `phowhisper_small` now has both calibration layers available:
  - `confidence=enabled:phowhisper_small`
  - `BoH=loaded:phowhisper_small`
- The 100% non-empty rate on non-speech confirms that the larger small model
  also hallucinates aggressively on silence/noise without RobustASR guards.
- Raw non-speech outputs may include one-character artifacts such as `n` or `a`.
  Those are excluded from the BoH artifact by the phrase-length selection rule;
  short garbage is handled by the existing text heuristics instead.

---

## D2 — LLM Baseline (PhoGPT-4B-Chat Q4_K_M via llama.cpp)

**Setup**

| Field             | Value                                                                    |
| ----------------- | ------------------------------------------------------------------------ |
| Model             | `vinai/PhoGPT-4B-Chat-gguf`, file `PhoGPT-4B-Chat-Q4_K_M.gguf` (~2.4 GB) |
| Architecture      | MPT-style (ALiBi positions, GeLU, LayerNorm) — not Llama                 |
| Quantization      | Q4_K_M (~95% of FP16 quality at 1/3 size)                                |
| Runtime           | `llama-cpp-python` with Metal acceleration (`-DGGML_METAL=on`)           |
| n_ctx             | 2048                                                                     |
| n_threads         | 4                                                                        |
| n_gpu_layers      | -1 (all layers on Metal GPU)                                             |
| Seed              | 42                                                                       |
| Decode            | Streaming sampling, `temperature=0.7`, `top_p=0.95`, `max_tokens=128`    |
| Prompt template   | `### Câu hỏi: {persona}\n\nCâu hỏi của tôi: {user}\n### Trả lời:`        |
| Persona injection | "Bạn là SoCa, trợ lý ảo tiếng Việt..." (first turn only)                 |
| Eval prompts      | 15 hand-crafted Vietnamese prompts (factual / reasoning / commands)      |
| Run date          | 2026-05-19                                                               |
| Device label      | "Mac 4 (Metal)" in JSON — actually MacBook M4 Pro                        |
| Output path       | `eval/results/llm_d2_baseline_mac_m4.json` (gitignored)                  |

**Results (15 prompts, after 1 warmup call)**

| Metric                       | Mean   | Median | P95    | Min   | Max    |
| ---------------------------- | ------ | ------ | ------ | ----- | ------ |
| TTFT (ms)                    | 61.3   | 61.7   | 62.2   | 59.8  | 62.2   |
| Total latency (ms, ≤128 tok) | 1004.8 | 754.3  | 2396.8 | 106.1 | 2396.8 |
| Throughput (tok/s)           | 62.8   | 63.6   | 67.0   | 55.4  | 67.0   |

**Resource usage**

| Metric          | Value    |
| --------------- | -------- |
| Peak memory     | 3.12 GB  |
| Model file size | ~2.36 GB |

**Notes**

- TTFT is extremely stable (stdev 0.77 ms) — Metal GPU prefill dominates
  cost; prompt token count variance has tiny effect.
- Total latency variance is dominated by `n_completion_tokens` (longer
  responses → more decode steps).
- 62.8 tok/s ≈ acceptable conversational speed; comfortably above Vietnamese
  speech rate (~3-4 syllables/sec for TTS pacing).

---

## D2.1 — LLM Bake-Off (Generic llama.cpp Registry)

**Purpose:** compare the practical local LLM candidates for the SoCa voice
assistant before changing the runtime default.

**Setup**

| Field           | Value                                                                                          |
| --------------- | ---------------------------------------------------------------------------------------------- |
| Runtime         | `soca.llm.LocalLlamaCppLLM` (`llama-cpp-python`, Metal)                                        |
| Model selection | `soca/llm/registry.py`, profile `bakeoff`                                                      |
| Prompt set      | `eval/prompts/llm_bakeoff_vi.jsonl`, 50 Vietnamese prompts                                     |
| Categories      | `assistant_command`, `local_utility`, `conversation`, `unknown_refusal`, `asr_noisy`, `coding` |
| Decode          | Streaming, `temperature=0.2`, `top_p=0.9`, `max_tokens=96`                                     |
| Run command     | `uv run python eval/eval_llm.py --profile bakeoff`                                             |
| Run date        | 2026-05-27                                                                                     |
| Output path     | `eval/results/llm_bakeoff_20260527_012847.{json,md}` (gitignored)                              |

**Results**

| Model                    | Role                               | TTFT p50 ms | TTFT p95 ms | tok/s mean | Total p95 ms | Too long | VI signal | CJK leak | EN leak | Cmd refuse | RT hallu | Privacy hallu | Peak MB |
| ------------------------ | ---------------------------------- | ----------: | ----------: | ---------: | -----------: | -------: | --------: | -------: | ------: | ---------: | -------: | ------------: | ------: |
| `phogpt_4b_q4_k_m`       | baseline                           |          62 |          63 |       62.1 |         2031 |    30.0% |    100.0% |     0.0% |    0.0% |      40.0% |    38.5% |         16.7% |    3096 |
| `arcee_vylinh_3b_q4_k_m` | primary candidate                  |          51 |          52 |       76.5 |          559 |     0.0% |     96.0% |     4.0% |    2.0% |      20.0% |    38.5% |         33.3% |    2161 |
| `qwen3_0_6b_q8_0`        | low-RAM fallback                   |          14 |          14 |      156.9 |          364 |     0.0% |     98.0% |     0.0% |    2.0% |      50.0% |     7.7% |         16.7% |    1277 |
| `vinallama_2_7b_q5_0`    | Vietnamese-trained small candidate |          47 |          47 |       54.2 |         1761 |    14.0% |     98.0% |     0.0% |    0.0% |      70.0% |    23.1% |         16.7% |    3278 |

**Decision**

- Keep PhoGPT as the historical LLM bake-off baseline only. The single current product profile,
  `baseline`, uses Arcee-VyLinh.
- Treat Arcee-VyLinh as the leading free-chat candidate after intent routing.
- Treat Qwen3-0.6B as the low-RAM fallback/smoke model.
- Do not prioritize VinaLLaMA-2.7B for the default path: clean Vietnamese output,
  but high command refusal, high memory, and slower full responses in this run.
- Do not send commands, real-time questions, or privacy-sensitive prompts straight
  into the LLM. The next Phase 3 task is a deterministic intent/tool/privacy
  router, then a post-router bake-off.

**Caveats**

- These numbers are Mac local measurements, not Raspberry Pi or ARM board claims.
- Behavioral rates are heuristic screeners, not human ratings; inspect the JSON
  responses before making public claims.

---

## D3 — TTS Bake-Off (Vietnamese local runtimes)

> **Status 2026-07-24 (historical):** this multi-engine bake-off is the record of
> _why_ Valtec was chosen; it predates the ONNX cutover. The registry and the
> per-model runners it references (`soca/tts/registry.py`, `mms_tts_vie`, `piper`,
> `vieneu`, `kani`, `f5`, `omnivoice`, …) no longer exist — SoCa collapsed to a
> single Valtec ONNX runtime. Current numbers are in **D3.0** below; the
> "Decision"/"Next benchmark" blocks in this section are superseded by that cutover.

**Purpose:** compare the practical Vietnamese TTS runtimes for the SoCa voice
loop before changing the default TTS engine.

**Setup**

| Field           | Value                                                                                                                                                                                                                            |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Runtime         | `soca.tts` registry + per-model runners                                                                                                                                                                                          |
| Model selection | `--all` registry candidates                                                                                                                                                                                                      |
| Prompt set      | `eval/prompts/tts_bakeoff_vi.jsonl`, 41 Vietnamese prompts                                                                                                                                                                       |
| Categories      | `short`, `assistant`, `coach`, `nutrition`, `fitness`, `safety`, `tracking`, `number`, `datetime`, `currency`, `measurement`, `name_place`, `abbreviation`, `punctuation`, `codeswitch`, `formal`, `casual`, `asr_noisy`, `long` |
| Voice policy    | Registry default voice only                                                                                                                                                                                                      |
| Process policy  | `--isolate-model-process`, one fresh subprocess per model                                                                                                                                                                        |
| Audio output    | Disabled for this timing run (`--no-write-audio`)                                                                                                                                                                                |
| Run date        | 2026-06-01                                                                                                                                                                                                                       |
| Output path     | `eval/results/tts_bakeoff/20260601_022154/{report.json,report.md,analysis.md}` (gitignored)                                                                                                                                      |

**Candidate pool and sources**

The candidate list is defined in `soca/tts/registry.py`. It intentionally
mixes production-usable local runners and heavier quality baselines:

| Model key              | Upstream/source                                                                                         | Why included                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `valtec_multispeaker`  | `https://github.com/tronghieuit/valtec-tts`                                                             | Current SoCa baseline; local Vietnamese multi-speaker runtime.                                   |
| `mms_tts_vie`          | `https://huggingface.co/facebook/mms-tts-vie`                                                           | Small VITS-style Vietnamese baseline through Transformers.                                       |
| `piper_vi_vivos_x_low` | `https://huggingface.co/speaches-ai/piper-vi_VN-vivos-x_low`, artifact path from `rhasspy/piper-voices` | ONNX/Piper edge fallback candidate.                                                              |
| `vieneu_v2_turbo`      | `https://huggingface.co/pnnbao-ump/VieNeu-TTS-v2-Turbo`                                                 | Main runtime challenger: Vietnamese-focused, CPU/GGUF-oriented path through the VieNeu SDK.      |
| `vieneu_v2_standard`   | `https://huggingface.co/pnnbao-ump/VieNeu-TTS-v2`                                                       | Higher-quality VieNeu path; included to measure quality/runtime trade-off.                       |
| `kani_370m_vie`        | `https://huggingface.co/pnnbao-ump/kani-tts-370m-vie`                                                   | Expressive Vietnamese candidate; kept in registry but tested separately due dependency conflict. |
| `viet_tts_onnx`        | `https://github.com/dangvansam/viet-tts`                                                                | ONNX/server-style Vietnamese TTS candidate.                                                      |
| `vixtts`               | `https://huggingface.co/capleaf/viXTTS`                                                                 | Quality/voice-clone baseline; exposed through an external command adapter.                       |
| `f5_vi_hynt`           | `https://huggingface.co/hynt/F5-TTS-Vietnamese-ViVoice`                                                 | F5-TTS Vietnamese quality baseline requiring reference audio.                                    |
| `f5_vi_zalopay`        | `https://huggingface.co/zalopay/vietnamese-tts`                                                         | Secondary F5 Vietnamese artifact requiring reference audio.                                      |
| `omnivoice`            | `https://huggingface.co/k2-fsa/OmniVoice`                                                               | Multilingual zero-shot/voice-clone quality path.                                                 |

**Prompt corpus design**

The benchmark uses a hand-authored text corpus, not a speech dataset. The goal
is to stress the text-to-audio layer that SoCa will actually speak:

- Short assistant turns: latency and clipping on very short outputs.
- Coach/nutrition/fitness/safety/tracking: project-specific assistant domain.
- Numbers, dates, currency, measurement: common Vietnamese TTS failure modes.
- Names/places and abbreviations: local assistant identity + technical terms.
- Punctuation and quote-like text: prosody and pause handling.
- Code-switching: English terms that appear in real SoCa responses.
- ASR-noisy text: no punctuation/casing, similar to raw ASR output.
- Long prompts: latency/RTF stability on longer assistant responses.

`--strict-prompts` validates that every required category is present before the
benchmark starts. In this run, all required categories were present; the
coverage summary is stored in `report.json` and printed in `report.md`.

**Measurement method**

For each selected model:

1. Create the TTS engine. TTS engines now load eagerly at construction time, so
   `load_ms` measures model/runtime readiness instead of a deferred first-call
   load.
2. Query available voices and select only the registry default voice. This keeps
   the benchmark about model/runtime comparison, not voice-search.
3. Synthesize all 41 prompts.
4. For each output, record latency, sample rate, sample count, audio duration,
   RTF, clipping ratio, requested voice, resolved voice, and error status.
5. Aggregate p50/p95 latency and RTF across successful prompts.
6. Mark the model `ok`, `partial`, `failed`, or `skipped_unavailable`.

`RTF = synthesis_latency_ms / generated_audio_duration_ms`; lower is better,
and values below 1.0 are faster than real-time. `Peak MB` is measured inside
the per-model subprocess using `psutil` when available, otherwise Python's
`resource.getrusage` fallback on macOS/Linux.

The run uses `--isolate-model-process` so each model is measured in a fresh
Python process. That avoids one model's loaded weights or native allocator state
polluting the next model's load time and memory reading.

**Why audio was disabled**

This run is the timing/runtime pass. `--no-write-audio` avoids disk I/O noise
and keeps the result folder small. Subjective quality is intentionally deferred
to a separate audio-writing pass, because runtime metrics alone cannot rank
naturalness, accent, or prosody.

**Run command**

```bash
uv run --extra tts --extra tts-piper --extra tts-omnivoice --extra tts-vieneu \
  python eval/eval_tts.py \
  --all \
  --isolate-model-process \
  --strict-prompts \
  --voice-policy default \
  --no-write-audio
```

**Results**

| Model                  | Tier | Runner             | Status | Voice        | Load ms | Lat p50 ms | Lat p95 ms | RTF p50 | RTF p95 | Peak MB | Notes                                                                                                                 |
| ---------------------- | ---: | ------------------ | -----: | ------------ | ------: | ---------: | ---------: | ------: | ------: | ------: | --------------------------------------------------------------------------------------------------------------------- |
| `piper_vi_vivos_x_low` |    A | `piper`            |     ok | `VIVOSSPK13` |    1078 |         51 |         83 |    0.01 |    0.01 |     706 | Fastest and lightest; terminal emitted missing-phoneme warnings, so treat as edge fallback rather than quality voice. |
| `vieneu_v2_turbo`      |    A | `vieneu`           |     ok | `XuanVinh`   |    2152 |        335 |        579 |    0.09 |    0.11 |    1616 | Strongest runtime challenger in this run.                                                                             |
| `valtec_multispeaker`  |    A | `valtec`           |     ok | `NF`         |    1024 |        418 |        550 |    0.10 |    0.14 |    2254 | Current stable baseline; good integration default.                                                                    |
| `mms_tts_vie`          |    A | `mms_transformers` |     ok | `default`    |    2162 |        478 |        610 |    0.09 |    0.11 |    2477 | Useful baseline, but measured peak memory is not lower than Valtec here.                                              |
| `vieneu_v2_standard`   |    A | `vieneu`           |     ok | `TrucLy`     |   50540 |        800 |       1142 |    0.20 |    0.22 |    1684 | Works, but load time is very high. Use only if subjective quality beats Turbo.                                        |
| `omnivoice`            |    B | `omnivoice`        |     ok | `auto`       |    2852 |       2767 |       4378 |    0.71 |    0.78 |    3031 | Quality/voice-clone path, not a low-latency default in auto mode.                                                     |

**Skipped candidates**

| Model           | Reason                                                           | Next action                                                             |
| --------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `kani_370m_vie` | `kani-tts-2` conflicts with SoCa's main Hugging Face stack.      | Test in a separate `uv` environment.                                    |
| `viet_tts_onnx` | Local VietTTS server was not running at `http://127.0.0.1:8298`. | Start the server and rerun this model only.                             |
| `vixtts`        | External command runner was not configured.                      | Set `SOCA_TTS_VIXTTS_COMMAND`.                                          |
| `f5_vi_hynt`    | Missing reference audio/text env vars.                           | Set `SOCA_TTS_F5_HYNT_REF_AUDIO` and `SOCA_TTS_F5_HYNT_REF_TEXT`.       |
| `f5_vi_zalopay` | Missing reference audio/text env vars.                           | Set `SOCA_TTS_F5_ZALOPAY_REF_AUDIO` and `SOCA_TTS_F5_ZALOPAY_REF_TEXT`. |

**Decision**

- Keep `valtec_multispeaker` as the stable baseline/default until subjective
  audio review is complete.
- Promote `vieneu_v2_turbo` to the leading runtime challenger: it is faster
  than Valtec/MMS in this run, has moderate memory, and produced 100% non-empty
  outputs with 0% errors.
- Keep `piper_vi_vivos_x_low` as the edge fallback: extremely fast and light,
  but needs listening review because of missing-phoneme warnings.
- Keep `omnivoice` for quality/voice-clone experiments. The saved
  `emgai_dangiu` voice must be benchmarked separately; this run intentionally
  used registry default `auto`.
- Do not prioritize `vieneu_v2_standard` for the interactive loop unless human
  listening shows a clear quality win over Turbo; 50.5s load time is too high
  for a default runtime.

**Next benchmark**

Run an audio-writing pass for subjective listening:

```bash
uv run --extra tts --extra tts-piper --extra tts-omnivoice --extra tts-vieneu \
  python eval/eval_tts.py \
  --model vieneu_v2_turbo,valtec_multispeaker,piper_vi_vivos_x_low,omnivoice \
  --voice-map omnivoice=emgai_dangiu \
  --isolate-model-process \
  --strict-prompts \
  --voice-policy default
```

**Caveats**

- This is a timing/runtime bake-off, not a final perceptual-quality ranking.
- `Peak MB` is per subprocess and includes Python/runtime overhead.
- Audio was not saved in this run, so quality conclusions must wait for the
  subjective listening pass.

---

## D3.0 — Valtec ONNX release (current, cutover complete)

**Purpose:** record the post-cutover numbers for the single Valtec ONNX runtime
that replaced the D3 registry. SoCa now ships one self-built, checksum-pinned
Valtec release; there is no model selector.

**Setup**

| Field          | Value                                                                                          |
| -------------- | ---------------------------------------------------------------------------------------------- |
| Runtime        | `soca.tts` → `create_tts_engine(voice=…)` → `ValtecOnnxTTS` (four ONNX graphs, fp32)           |
| Active release | `soca-valtec-20260724-50fd400` (`current.json`), rollback point `soca-valtec-20260722-a1b2c3d` |
| Checkpoint     | `valtecAI-team/valtec-tts-pretrained` rev `d58e991…`, license CC BY-NC 2.0                     |
| Prompt set     | `eval/prompts/tts_bakeoff_vi.jsonl` (bench 12 / loopback 12), 5 voices NF·SF·NM1·SM·NM2        |
| Reviewer       | listening + license: `vominhthinh` (fail-closed acceptance)                                    |
| Run date       | 2026-07-24                                                                                     |

**Results (fp32, active variant)**

| Metric                 | Value    | Release gate | Pass |
| ---------------------- | -------- | ------------ | ---- |
| Parity (5 voices)      | all pass | 5/5          | ✅   |
| Latency p50            | 271.0 ms | ≤ 300 ms     | ✅   |
| Latency p95            | 416.4 ms | ≤ 550 ms     | ✅   |
| RTF p50 (primary gate) | 0.070    | ≤ 0.12       | ✅   |
| ASR loopback CER       | 0.134    | ≤ 0.15       | ✅   |

**Variant decision**

- **fp32 active.** int8 dynamic quantization is **−27% slower** on M4 arm64 (dynamic
  quant has no arm64 kernel win here) while spectrally near-identical (parity cosine
  0.99999, waveform MAE 0.021). Slower + no quality gain → int8 is built and kept in
  the manifest but not selected.
- RTF is the primary latency gate (proves real-time headroom); the absolute-ms gates
  are secondary head-room bounds raised to 300/550 to admit long sentences.

**Caveats**

- Loopback CER uses 12 prompts, so a single ASR mis-hear moves it noticeably; 0.134
  passed the gate and the subjective listening review, but is worth re-checking on a
  larger slice if pronunciation regressions surface.
- Latency measured on the D3 common hardware (Apple M4), single process, warm engine.

---

## D3.1 — E2E Voice Loop Benchmark (audio fixture → ASR → runtime → TTS)

> **Historical snapshot (2026-06-01).** The profile names and multi-engine commands in this section
> document the experiment as it ran. They are not supported by the current singleton `baseline`
> runtime and must not be copied as current CLI instructions.

**Purpose:** measure the full SoCa voice loop from audio input to first
available output audio, using fixed WAV fixtures rather than live microphone
input. This complements the TTS-only bake-off above.

**Setup**

| Field             | Value                                                                                                           |
| ----------------- | --------------------------------------------------------------------------------------------------------------- |
| Runtime           | `VoicePipeline.turn_streaming()` with `AssistantRuntime`                                                        |
| Input audio       | `eval/audio/voice_loop_smoke/*.wav` generated from `eval/prompts/voice_loop_smoke_vi.jsonl`                     |
| Fixture generator | `omnivoice`, saved voice `emgai_dangiu`                                                                         |
| Fixture prompts   | 6 Vietnamese utterances: greeting, nutrition, time, knowledge-search prefix, safety, unsupported scheduling     |
| Knowledge vault   | `eval/fixtures/knowledge_vault`                                                                                 |
| Memory            | Disabled (`--no-memory`) to avoid personal/local profile state                                                  |
| Playback          | `NullAudioPlayer`; audio is synthesized but not sent to speakers                                                |
| Process policy    | One fresh shell command per profile for cleaner load/memory readings                                            |
| Run date          | 2026-06-01                                                                                                      |
| Output paths      | `eval/results/voice_loop/20260601_190159`, `20260601_190228`, `20260601_190239`, `20260601_190309` (gitignored) |

**Measured profiles**

| Profile           | ASR                | LLM                      | TTS                    | Voice          | Role                                 |
| ----------------- | ------------------ | ------------------------ | ---------------------- | -------------- | ------------------------------------ |
| `baseline`        | `phowhisper_base`  | `arcee_vylinh_3b_q4_k_m` | `valtec_multispeaker`  | `NF`           | Stable local baseline                |
| `edge`            | `phowhisper_base`  | `qwen3_0_6b_q8_0`        | `piper_vi_vivos_x_low` | `VIVOSSPK13`   | Low-latency fallback                 |
| `balanced_vieneu` | `phowhisper_base`  | `arcee_vylinh_3b_q4_k_m` | `vieneu_v2_turbo`      | `XuanVinh`     | Practical quality/latency challenger |
| `quality`         | `phowhisper_small` | `arcee_vylinh_3b_q4_k_m` | `omnivoice`            | `emgai_dangiu` | Saved-voice quality profile          |

**Measurement method**

For each profile:

1. Load ASR, LLM, TTS, knowledge source, tool runtime, and AssistantRuntime.
2. Read each fixture WAV, resample to 16 kHz mono, and call
   `VoicePipeline.turn_streaming()`.
3. Record ASR latency, runtime latency, first TTS chunk latency, TTFA, total
   turn latency, route, transcript, response text, chunk count, reject status,
   error status, and process RSS.
4. Use `NullAudioPlayer` so playback hardware latency does not affect TTFA.

`TTFA` means **time to first audio chunk available**, measured from the start of
pipeline processing to the first synthesized TTS chunk. `Total` means the full
turn until all chunks are synthesized and the pipeline emits `done`.

**Fixture generation command**

```bash
uv run --extra tts --extra tts-omnivoice python eval/eval_voice_loop.py \
  --profile baseline \
  --generate-fixtures \
  --overwrite-fixtures \
  --fixture-tts-model omnivoice \
  --fixture-voice emgai_dangiu \
  --vault eval/fixtures/knowledge_vault \
  --no-memory \
  --no-playback
```

The fixture-generation run is not used as the official profile measurement,
because it loads the fixture TTS engine before benchmark timing. The profile
measurements below were run afterwards in separate fresh processes.

**Historical profile run commands (no longer supported)**

```bash
uv run --extra tts python eval/eval_voice_loop.py \
  --profile baseline \
  --vault eval/fixtures/knowledge_vault \
  --no-memory \
  --no-playback

uv run --extra tts --extra tts-piper python eval/eval_voice_loop.py \
  --profile edge \
  --vault eval/fixtures/knowledge_vault \
  --no-memory \
  --no-playback

uv run --extra tts --extra tts-vieneu python eval/eval_voice_loop.py \
  --profile balanced_vieneu \
  --vault eval/fixtures/knowledge_vault \
  --no-memory \
  --no-playback

uv run --extra tts --extra tts-omnivoice python eval/eval_voice_loop.py \
  --profile quality \
  --vault eval/fixtures/knowledge_vault \
  --no-memory \
  --no-playback
```

**Results**

| Profile           | Load ms | ASR p50 ms | Runtime p50 ms | TTS0 p50 ms | TTFA p50 ms | TTFA p95 ms | Total p50 ms | Total p95 ms | Avg chunks | Peak MB | Error |
| ----------------- | ------: | ---------: | -------------: | ----------: | ----------: | ----------: | -----------: | -----------: | ---------: | ------: | ----: |
| `edge`            |    1867 |        326 |             85 |          46 |         487 |         677 |          487 |          677 |        1.0 |    3090 |  0.0% |
| `balanced_vieneu` |    2219 |        343 |            561 |         385 |        1190 |        3285 |         1763 |         7542 |        5.3 |    4872 |  0.0% |
| `baseline`        |    1428 |        364 |            555 |         696 |        1331 |        3694 |         1957 |         8391 |        5.3 |    5004 |  0.0% |
| `quality`         |    5593 |       1278 |            507 |        7054 |        9211 |       12691 |        14270 |        79801 |        5.2 |    6044 |  0.0% |

**Observed transcripts and route caveat**

This run uses OmniVoice-generated fixtures because the earlier Valtec-generated
fixtures caused severe ASR keyword drift. OmniVoice substantially improved the
audio fixture quality: common natural utterances were transcribed correctly, and
the realtime question now routes through `local_time.now`. The remaining route
failures are useful product findings rather than benchmark noise:

| Fixture intent         | Expected text                            | Observed transcript (`baseline`)          |
| ---------------------- | ---------------------------------------- | ----------------------------------------- |
| greeting               | `xin chào`                               | `xin chào.`                               |
| nutrition              | `bữa sáng nhanh nhưng đủ chất nên ăn gì` | `bữa sáng nhanh nhưng đủ chất nên ăn gì.` |
| time                   | `mấy giờ rồi`                            | `mấy giờ rồi.`                            |
| knowledge              | `wiki chất đạm`                          | `quy ki chất đạm.`                        |
| safety                 | `nếu tập mà chóng mặt thì nên làm gì`    | `nếu tập mà chóng mặt thì nên làm gì.`    |
| unsupported scheduling | `đặt hẹn giờ 5 phút`                     | `đặt hẹn giờ năm phút.`                   |

The route counts for each profile were `tool_direct: 1`, `blocked: 1`, and
`free_chat: 4`. That confirms the full audio path can reach a tool route and
can also block unsupported scheduling requests before LLM execution. This
fixture set still does not validate knowledge routing end-to-end because the
knowledge prompt is too brittle around the spoken word `wiki`. The old
in-memory timer prototype has been removed because it did not set a real OS/app
timer; scheduling requests stay blocked until a real scheduler backend exists.

**Decision**

- `edge` is the fastest E2E profile by a large margin, but it uses the weakest
  LLM fallback and Piper emitted missing-phoneme warnings in the TTS bake-off.
- `balanced_vieneu` remains the best practical quality/latency challenger, but
  on this fixture set its longer generated responses make total latency higher
  than the `edge` profile and slightly better than `baseline`.
- `baseline` remains the stable integration default until subjective listening
  and route-coverage fixtures are improved.
- `quality` with OmniVoice `emgai_dangiu` is too slow for default interactive
  use in this measured configuration, but remains useful for voice-quality
  demos and offline response generation.

**Next E2E benchmark**

- Keep the OmniVoice fixture set for latency regression and realtime-tool
  coverage.
- Add route-specific fixtures for knowledge retrieval, likely with a more
  natural phrase such as `tìm trong ghi chú về chất đạm` instead of relying on
  the spoken token `wiki`.
- Keep scheduling/timer requests blocked unless a real scheduler backend is
  implemented.
- Add an optional real-mic fixture set that stays local/ignored if it contains
  personal voice data.

---

## D3.2 — Phase 7 clause chunking + PCM continuity (`tts-improvement`)

**Purpose:** record what Phase 7 (I1–I6) actually changed and what was measured for
it. Phase 7 has three independent parts: safe first-clause chunking (text side, lower
TTFA), tail-holding equal-gain cross-fade (DSP), and a persistent playback session
(device continuity, no inter-chunk gap). Each is measured by a different gate; **numbers
are not conflated**.

**Setup**

| Field           | Value                                                                  |
| --------------- | ---------------------------------------------------------------------- |
| Branch          | `tts-improvement` (cut from `main` after `tts-refactor` merged, PR #2) |
| Active release  | `soca-valtec-20260724-50fd400`, role `release`, fp32                   |
| Manifest sha256 | `e61889df0aaf867f79266e8aa5ac60cf7adb38fa149e28c6a35bb8bc79df7a04`     |
| Run date        | 2026-07-24, Apple M4, warm engine, single process                      |
| Unit tests      | 580 passed / 1 skipped; ruff + compileall clean                        |

**(a) Offline A/B waveform** — `eval/eval_valtec_chunk_join.py`, 5 voices × 30 prompts →
450 WAV built from **identical** synthesized chunks (`hard` / `equal_gain_8ms` /
`equal_gain_12ms`). Output `eval/results/valtec_chunk_join/` (gitignored).

| Metric                      | Value                   | Gate     | Pass |
| --------------------------- | ----------------------- | -------- | ---- |
| `peak_abs` max (all)        | 0.9633                  | ≤ 1.0    | ✅   |
| `hard_boundary_jump`        | median 0.0, p95 0.0002  | — (info) | —    |
| Multi-chunk rows (A/B real) | 125 / 150               | —        | —    |
| `chunk_latency_ms`          | median 204.7, p95 453.2 | —        | —    |

> Finding: at these clause boundaries the hard-join sample jump is already ≈ 0 (Valtec
> chunks begin/end near silence), so cross-fade **introduces no artifact** but also shows
> no measurable offline win. Listening review (user, 2026-07-24) confirmed **no audible
> difference** between hard / 8 ms / 12 ms → listening gate **PASS**; `pcm_crossfade_ms`
> default kept at **12 ms** because that switch also keeps the gap-free session path on.

**(b) Device playback** — real `SoundDevicePlayer` on the default output ("Loa MacBook
Pro"), 5 turns / 7 boundaries. ASR + LLM **bypassed** (they produce no device metric);
this isolates the TTS → pump → session → speaker path Phase 7 changed.

| Metric                   | Value                  | Gate            | Pass |
| ------------------------ | ---------------------- | --------------- | ---- |
| `audible_ttfa_ms`        | p50 239, p95 310       | improves        | ✅   |
| `tts_ready_ttfa_ms`      | p50 211                | separate #      | ✅   |
| ready → audible delta    | ~28 ms                 | device cost     | ✅   |
| `synthesis_slack_ms`     | p50 2829, p05 2137     | p50≥100, p05≥40 | ✅   |
| `crossfade_ms`           | 12.0 on 7/7 boundaries | overlap real    | ✅   |
| `output_underflow_count` | 0                      | == 0            | ✅   |
| `crossfade_fallback`     | 0 (0.0%)               | < 1%            | ✅   |

**(c) E2E voice loop** — `eval/eval_voice_loop.py --profile baseline` (real
`phowhisper_base` + `arcee_vylinh_3b_q4_k_m` + Valtec), NullAudioPlayer, 12 prompts,
`eval/results/valtec_chunk_join_live/` (gitignored).

| Metric                   | Value         | Note                                          |
| ------------------------ | ------------- | --------------------------------------------- |
| `output_underflow_count` | 0 on all rows | Phase 7 continuity holds end-to-end ✅        |
| TTFA p50                 | 3071 ms       | **ASR-bound, not comparable to D3.1 1331 ms** |

> Honest caveat: the E2E TTFA here is dominated by ASR (per-row `stage_latencies_ms/asr`
> ≈ 1.7–2.6 s) because `--generate-fixtures` synthesized the **long** chunk-join prompts
> into long input audio. That is a different (harder) prompt set than the D3.1 baseline
> (ASR p50 364 ms), so this run does **not** measure the first-clause TTFA effect and is
> not a valid comparison to 1331 ms. It only confirms `output_underflow_count == 0` in the
> full loop.

**(d) First-clause TTFA A/B (controlled)** — `eval/measure_first_clause_ttfa.py`. Captures one
real LLM (`arcee_vylinh_3b_q4_k_m`) token stream per prompt **with per-token arrival times**, then
replays that exact stream (same tokens, same delays) through the runtime with `first_clause`
ON vs OFF. ASR is excluded and held constant, so the delta isolates the flush-point effect. 8
conversational transcripts.

| Metric (positive = first-clause faster) | p50     | range         | Prompts helped |
| --------------------------------------- | ------- | ------------- | -------------- |
| Δ time-to-first-sentence (text side)    | +184 ms | −0 … +453 ms  | 7 / 8          |
| Δ tts_ready (text + Valtec synth)       | +395 ms | −14 … +928 ms | 7 / 8          |

> This closes the "does I1 lower TTFA" question with a **positive, honest** result: on 7/8
> conversational prompts first-clause flushes the first chunk ~184 ms earlier (text side), and
> ~395 ms earlier to first playable audio because the shorter first clause also synthesizes
> faster. The 1/8 no-benefit case is a response with no clause boundary before the first period
> (on == off) — expected, not a regression. This is the LLM→first-chunk delta attributable to
> first-clause, not an absolute E2E TTFA figure.

**Harness (now CLI-native):** `eval/eval_voice_loop.py` gained `--playback` (routes a real
`SoundDevicePlayer` through the pump's persistent-session + crossfade path) and
`--first-clause` / `--no-first-clause` (overrides the profile for an on/off A/B, via a new
`first_clause_enabled` override on `resolve_voice_runtime_config`). A `--playback` smoke on 2
prompts confirmed the harness populates device metrics: `playback_sink=SoundDevicePlayer`,
`audible_ttfa_ms` non-null, ready→audible ≈ 37 ms, `output_underflow_count` 0. (Absolute TTFA
there is still ASR-bound on the long fixtures — the controlled A/B (d) remains the valid
first-clause number.) The standalone `eval/measure_*.py` scripts are kept as focused probes.

**Still open (not blockers):**

- DuplexAecSink far-path is unit-tested but not validated live (needs mic + barge-in loop).

---

## D2.5 — ASR Robustness (adapted from Barański et al. ICASSP 2025)

Five sub-deliverables: **(A)** non-speech dataset, **(B)** Vietnamese BoH
construction, **(C)** heuristic threshold calibration, **(D)** runtime
pipeline (`soca.asr.RobustASR`), **(E)** Table VII-style benchmark.

### A. Non-speech dataset

**Setup**

| Field                 | Value                                                                                                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ESC-50                | `ashraq/esc50` HF, `train` split, streaming                                                                                                                                                |
| ESC-50 inclusions     | 500 samples after exclusion filter                                                                                                                                                         |
| ESC-50 exclusions     | `crying_baby`, `sneezing`, `clapping`, `breathing`, `coughing`, `footsteps`, `laughing`, `brushing_teeth`, `snoring`, `drinking_sipping`, `crying`, `speaking` (avoid voice contamination) |
| Synthetic silence     | 100 samples, durations ∈ {1, 3, 5, 10, 20}s, amplitude 0                                                                                                                                   |
| Synthetic white noise | 100 samples, amplitudes ∈ {0.001, 0.003, 0.005, 0.01}                                                                                                                                      |
| Synthetic pink noise  | 100 samples, 1/√f spectrum, same amplitudes                                                                                                                                                |
| Total                 | 800 samples                                                                                                                                                                                |
| Sample rate           | 16 kHz mono (resampled via librosa where needed)                                                                                                                                           |
| Seed                  | 42                                                                                                                                                                                         |
| Reproducible via      | `uv run python -m local.collect_noise --target 800`                                                                                                                                        |

### B. BoH construction (PhoWhisper-tiny on 800 non-speech)

**Setup**

| Field            | Value                                                                                                      |
| ---------------- | ---------------------------------------------------------------------------------------------------------- |
| Model            | `huuquyet/PhoWhisper-tiny` (same as D1)                                                                    |
| Decode           | Greedy, `max_new_tokens=128`, no KV cache                                                                  |
| Providers        | `CoreMLExecutionProvider` → `CPUExecutionProvider` fallback (4 threads)                                    |
| Normalization    | NFC → lowercase → collapse whitespace → strip boundary punctuation (`. , ! ? ; : " ' " " ' ' ( ) [ ] { }`) |
| Candidate filter | `count ≥ 2 AND len ≥ 5 chars`                                                                              |
| Manual review    | Interactive CLI: per-phrase keep/reject (`local.boh_manual_review`)                                        |
| Wall time        | 482 s on M4 Pro CoreML (~0.6 s/sample)                                                                     |
| Run date         | 2026-05-20                                                                                                 |
| Output path      | `data/asr/boh/phowhisper_tiny_vi_boh_v1.json` (gitignored)                                                 |

**Results**

| Metric                                    | Value                                    |
| ----------------------------------------- | ---------------------------------------- |
| Non-empty outputs on 800 noise            | **100%**                                 |
| Unique normalized outputs                 | 100                                      |
| BoH candidates after auto-filter          | 78                                       |
| BoH after **manual review** (`keep=True`) | **74**                                   |
| Phrases rejected during manual review     | 4                                        |
| Rejected phrases                          | `đúng rồi`, `các em`, `hôm nay`, `chúng` |
| Coverage (sum of kept counts / total)     | 49.9%                                    |
| Top phrase recurrence                     | 100 / 800                                |

**Paper comparison**: Whisper-large-v3 hallucinates on ~40% non-speech;
PhoWhisper-tiny hallucinates on 100% because it is ~40× smaller (39M vs 1550M).
High rate confirms why D2.5 mitigation is essential for the on-device tier.

**Manual review note**: `các em` was rejected in a second review pass after
it caused 2/200 false positives on real FLEURS speech (it is legitimate
Vietnamese for "you/children" but entered BoH from the `"các em nhá..."`
hallucination pattern). Short phrases (≤3 words) are the main false-positive
risk; all other kept phrases are ≥4 words.

**YouTube data leak**: 30+ of 74 BoH phrases match the `"các em nhá thấy..."`
pattern (teacher addressing "you guys"). This is the Vietnamese analogue of
the WhisperX issue #1086 `"La La School"` leak — Whisper inherits YouTube
auto-caption artifacts cross-language.

### C. Heuristic threshold calibration

**No ASR is involved in this measurement.** Thresholds are derived from the
distribution of three metrics computed on **ground-truth transcripts** of
real Vietnamese speech, so they characterize what natural speech looks like.

**Setup**

| Field             | Value                                                                        |
| ----------------- | ---------------------------------------------------------------------------- |
| Dataset           | `google/fleurs`, config `vi_vn`, split `test` (first 200 streamed samples)   |
| Sample rate       | 16 kHz mono (resampled if needed)                                            |
| Inputs to metrics | Ground-truth `raw_transcription` text + audio duration                       |
| Metrics           | 3 (see below)                                                                |
| Derivation policy | `recommended = p99 × (1 + margin)`                                           |
| Margins           | 0.15 for repetition metrics, 0.50 for density                                |
| Default rounding  | Recommended rounded **up** to leave headroom                                 |
| Applied where     | `soca.asr.hallucination_heuristics.check_heuristics`, Stage 5 of `RobustASR` |
| Run date          | 2026-05-20                                                                   |
| Output path       | `data/asr/threshold_calibration.json` (gitignored)                           |
| Reproducible via  | `uv run python -m local.calibrate_thresholds`                                |

**Metric definitions**

- `unigram_repetition` = `1 - (unique_tokens / total_tokens)`. Higher = more repetition.
- `3gram_repetition` = `1 - (unique_3grams / total_3grams)`. Catches looping windows.
- `chars_per_100ms` = `len(text) / (audio_duration_ms / 100)`. Density check for "too much text from too little audio".

**Full distribution (200 FLEURS vi samples)**

| Metric             | Mean  | P50   | P90   | P95   | P99   | Max   | Recommended | Default (rounded) |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ----- | ----------- | ----------------- |
| unigram_repetition | 0.084 | 0.069 | 0.189 | 0.212 | 0.278 | 0.339 | 0.319       | **0.35**          |
| 3gram_repetition   | 0.006 | 0.000 | 0.022 | 0.038 | 0.095 | 0.114 | 0.110       | **0.12**          |
| chars_per_100ms    | 1.092 | 1.082 | 1.346 | 1.427 | 1.567 | 1.751 | 2.350       | **2.50**          |

Defaults are hard-coded in [soca/asr/hallucination_heuristics.py:73-78](soca/asr/hallucination_heuristics.py#L73-L78).

**Expected false-positive rate**

By construction, ~1% of real speech crosses p99. Rounding up adds 1-2pp
margin, so practical false-positive rate < 1% absolute (confirmed in
Section E: 0/100 real-speech samples were flagged by heuristics).

**Caveats**

- `n=200` is borderline for stable p99 estimation. Recommend ≥500 for
  production; current values may shift by ~5% with larger sample.
- Distribution is from ground truth, not ASR output. ASR adds tokenization
  artifacts (e.g. spacing, missing punctuation) that may bias the metric
  slightly. Monitor false-rejection in Section E as the ground-truth check.
- Vietnamese-specific. Other languages need their own calibration.

### D. Runtime pipeline (`soca.asr.RobustASR`)

Six stages, all configurable:

1. `SpeechDetector` (Silero VAD, `threshold=0.5`, **Whisper-tuned**: `min_silence=500ms`, `pad=200ms`)
2. `VietnameseASR` (D1 config, applied to VAD-trimmed speech only)
3. ASR confidence guard (`avg_logprob`, `compression_ratio`) on raw model output
4. `remove_consecutive_repeats` (de-loop)
5. `VietnameseBoH` (Aho-Corasick match against 74-phrase BoH after manual review)
6. `check_heuristics` (filler → unigram_rep → 3gram_rep → density, short-circuit on first rejection)

**VAD param rationale**: Silero defaults are 100/30 ms (too aggressive for Whisper),
faster-whisper uses 2000/400 (good for long-form audio but adds UX latency for
voice command). 500/200 ms is the hybrid chosen for sub-second push-to-talk.

**ASR confidence calibration** (`uv run python -m local.calibrate_asr_confidence
--n-speech 200 --n-noise 50`):

| Metric / event                  | Value                       |
| ------------------------------- | --------------------------- |
| Speech detected by VAD          | 200/200 (100%)              |
| Noise detected as speech by VAD | 1/50 (2%)                   |
| `avg_logprob` speech p01        | -0.250                      |
| `avg_logprob` noise max         | -1.200                      |
| Applied `min_avg_logprob`       | **-0.725** (midpoint)       |
| `compression_ratio` speech p99  | 1.543                       |
| Applied `max_compression_ratio` | **2.400** (Whisper default) |

The only VAD-leaked noise sample produced raw text `"thôi."` with
`avg_logprob=-1.200`; the calibrated confidence guard rejects it before
de-loop/BoH/heuristics.

### E. Table VII replication

**Setup**

| Field             | Value                                                                    |
| ----------------- | ------------------------------------------------------------------------ |
| Speech eval       | 200 first samples of FLEURS vi_vn test split                             |
| Noise eval        | 50 first samples of `data/noise_for_boh/manifest.jsonl`                  |
| Configurations    | 6 (subset toggles of `RobustASR` stages — see table below)               |
| BoH snapshot      | 74 phrases (post manual review, `các em` rejected)                       |
| WER normalization | `lower().strip()` on both ref and hyp before `jiwer.wer`                 |
| CER               | Same input, via `jiwer.cer`                                              |
| Hallucination     | `noise_output.strip() != ""` (1 = hallucinated)                          |
| Latency           | End-to-end per sample (VAD + ASR + post-processing, excludes model load) |
| Warmup            | 1 ASR call before timing                                                 |
| Providers         | `CoreMLExecutionProvider` → `CPUExecutionProvider` fallback              |
| Run date          | 2026-05-21T11:27 UTC                                                     |
| Output path       | `eval/results/table7_replication.json` (gitignored)                      |
| Reproducible via  | `uv run python -m local.eval_table7 --n-speech 200 --n-noise 50`         |

**Results**

| Config                | WER    | CER    | Halluc rate | Lat p50 ms | Lat p95 ms |
| --------------------- | ------ | ------ | ----------- | ---------- | ---------- |
| (1) Raw ASR           | 25.45% | 12.52% | 100%        | 1235       | 2562       |
| (2) De-loop only      | 24.62% | 12.03% | 100%        | 1233       | 2701       |
| (3) Silero VAD only   | 25.22% | 12.40% | **2%**      | 1260       | 2417       |
| (4) BoH only          | 25.45% | 12.52% | 100%        | 1256       | 2730       |
| (5) De-loop + BoH     | 24.62% | 12.03% | 100%        | 1213       | 2724       |
| (6) **Full pipeline** | 25.22% | 12.40% | **0%**      | 1235       | 2471       |

**Metric caveat**

"Hallucination rate" = `non-empty noise output / total noise`. Favors VAD
(which skips noise → empty output) and **undervalues BoH** (which removes
matched phrases but residual punctuation/words remain non-empty).

A finer measurement on the same eval set:

| BoH effect                         | Value            |
| ---------------------------------- | ---------------- |
| Noise samples modified by BoH      | 27/50 (54%)      |
| Noise samples fully emptied by BoH | 22/50 (44%)      |
| BoH false positives on real speech | **0/200 (0.0%)** |

→ BoH catches 44% of noise hallucinations on its own, with **zero**
real-speech false positives after `các em` was rejected (was 2/200 with the
75-phrase set). The 100% rate in column 4/5 above is a metric artifact, not
a regression.

**Prior escaped edge case fixed by confidence guard**: `esc50_0048.wav`
(label `"insects"`) made VAD detect 720 ms of "speech" and ASR emit
`"thôi."` (= "stop"). De-loop / BoH / text heuristics all passed because a
single valid-looking word has no repetition, no excess density, and is not a
filler. After calibration, `avg_logprob=-1.200 < -0.725`, so
`RobustASR` rejects it as `low_confidence:-1.20`.

**Key finding**

Full pipeline reduces hallucination rate from 100% to **0%** (50/50 noise
samples correctly rejected) with **−0.23pp WER** on real Vietnamese speech
(25.45% raw → 25.22% full) and **0% observed false positives** in this run.
Note: full pipeline WER is **slightly lower** than raw because de-loop fixes
some over-decoded outputs on real speech faster than VAD/BoH/confidence
over-rejection adds error.

Matches the qualitative mitigation pattern from Barański et al. for
Whisper-large-v3 on LibriSpeech-augmented, adapted for Vietnamese
PhoWhisper-tiny.

---

## P2 — Hybrid RAG retrieval + tool router + retrieved memory

**Purpose:** measure the P2 knowledge/memory stack — hybrid retrieval, the
validated tool router, and query-aware retrieved memory — on real data, and
record the numbers that the P2 acceptance gates refer to.

Measured 2026-07-27 · macOS arm64 · Python 3.11.14 · `fastembed` 0.8.0 ·
`sentence-transformers` 5.6.1. Retrieval index built once and reused
(`index_reuse=true`).

### P2.1 — Hybrid retrieval on real XQuAD-Vietnamese

Corpus: `eval/fixtures/real_rag_vault` (48 Vietnamese Wikipedia articles from
XQuAD, CC BY-SA 4.0, `SOURCE_MANIFEST.json` SHA-256 verified). Cases:
`eval/prompts/real_rag_vi.jsonl` (1,193 questions). Provenance guarded by
`tests/test_real_rag_fixture.py` (passing).

The table below is the historical pre-incident baseline captured before the
no-evidence guard was added; the current-tree v2 run, with the guard active, is
recorded in P2.1.1.

```bash
uv run --all-extras python eval/eval_hybrid_retrieval.py \
  --vault eval/fixtures/real_rag_vault --cases eval/prompts/real_rag_vi.jsonl \
  --variant chunk_sparse --warm-repeats 1 --output eval/results/real-rag-chunk-sparse.json
uv run --all-extras python eval/eval_hybrid_retrieval.py \
  --vault eval/fixtures/real_rag_vault --cases eval/prompts/real_rag_vi.jsonl \
  --variant hybrid --backend fastembed --warm-repeats 1 --output eval/results/real-rag-hybrid.json
```

| Variant                    | Recall@5 | MRR@10 | nDCG@10 | p50 latency | p95 latency |
| -------------------------- | -------- | ------ | ------- | ----------- | ----------- |
| chunk_sparse (BM25)        | 0.979    | 0.919  | 0.937   | 36.5 ms     | 40.8 ms     |
| hybrid (BM25 + dense, RRF) | 0.994    | 0.971  | 0.978   | 40.1 ms     | 44.6 ms     |

Hybrid (RRF, `k=60`) beats sparse on every metric for ~4 ms extra p95. By slice,
hybrid scores Recall@5 0.995 on the XQuAD Wikipedia slice (`learning_notes`) and
0.667 on the small `life_vault_project` slice (3 cases).

### P2.1.1 — Vietnamese relevance incident and model bake-off

Measured 2026-07-27 after reproducing an unsupported Bayes query against the
real nutrition-only vault. The incident was caused by lexical retrieval
returning weak top-k matches without a no-evidence decision. The temporary
fail-closed guard is corpus-derived (IDF coverage + score separation), not a
Vietnamese stopword or subject-word rule. Full model-selection details are in
[`docs/10-vietnamese-rag-model-selection.md`](docs/10-vietnamese-rag-model-selection.md).

The pre-migration smoke corpus had eight labelled queries and all variants
scored 1.0 on Recall@5, MRR@10, and nDCG@10; the meaningful comparison is the
XQuAD slice.
The following table uses the first 200 XQuAD cases so model candidates can be
compared without hiding the current code's latency/memory trade-offs:

| Variant | Recall@5 | MRR@10 | nDCG@10 | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cached sparse + guard | 0.945 | 0.881 | 0.899 | 29.4 ms | 30.6 ms |
| `multilingual-e5-small` hybrid + guard | 0.975 | 0.960 | 0.964 | 93.2 ms | 97.7 ms |
| `Vietnamese_Embedding_v2` hybrid + guard | 0.970 | 0.968 | 0.970 | 115.2 ms | 140.3 ms |

The Vietnamese encoder is a promising candidate, but it is not silently made
the default. A full 1,193-case run is now recorded below; a calibrated
reranker/no-answer test is still required. `Vietnamese_Reranker` improved
no-answer score separation in a small diagnostic, but lowered smoke-set MRR to
0.9375 by misordering the RAG architecture query, so it remains an opt-in
research component.

Full v2 run (`rrf_k=60`, `warm_repeats=1`, `encoding_repeats=1`, same current
guard):

| Variant | Cases | Recall@5 | MRR@10 | nDCG@10 | warm p50 | warm p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Vietnamese_Embedding_v2` hybrid + guard | 1,193 | 0.975692 | 0.958761 | 0.963662 | 111.06 ms | 126.01 ms |

Slice metrics: `learning_notes` Recall@5 `0.975630`, MRR@10 `0.959330`,
nDCG@10 `0.964086`; `life_vault_project` Recall@5 `1.000000`, MRR@10
`0.733333`, nDCG@10 `0.795618` (only 3 cases). Cold mean/p95 were `132.51 /
127.92 ms`; isolated query encoding mean/p95 were `16.83 / 24.36 ms`. The
machine-specific JSON is ignored by git; the exact command and model-source
notes are in [`docs/10-vietnamese-rag-model-selection.md`](docs/10-vietnamese-rag-model-selection.md).


### P2.1.2 — Vector-search backend probe

This is a backend microbenchmark, not a Vietnamese retrieval-quality benchmark.
It uses normalized random float32 vectors, 40 singleton queries, and a stable
score-descending / row-ID-ascending exact oracle.

**Setup**

| Field | Value |
| --- | --- |
| Harness | `eval/eval_vector_backend.py` |
| Script SHA-256 | `af8eb272b9262313babb83a19cc9b04556f840feec01a900fcd67855204e3327` |
| Baseline commit | `acd94b7339a22df7ecbbabe93d756ce9ec6d3f76` |
| Machine | macOS 15.7.4 ARM64 |
| Python / NumPy | 3.11.14 / 2.4.4 |
| Optional backends | faiss-cpu 1.14.3; usearch 2.25.2 |
| Query workload | 40 singleton queries, k=10 |

**50,000 rows × 384 dimensions**

| Backend/config | p95/query | Recall@10 vs exact | Build | Serialized/raw bytes |
| --- | ---: | ---: | ---: | ---: |
| Current Python full sort | 13.914 ms | 1.0000 | — | 76.8 MB |
| NumPy stable partition exact | 0.814 ms | 1.0000 | — | 76.8 MB |
| FAISS Flat exact | **0.474 ms** | **1.0000** | 4.69 ms | 76.8 MB |
| FAISS HNSW M32, ef=128 | 0.866 ms | 0.4275 | 6.43 s | 90.4 MB |
| FAISS HNSW M32, ef=512 | 3.038 ms | 0.8825 | 6.43 s | 90.4 MB |
| FAISS HNSW M32, ef=1024 | 4.766 ms | 0.9700 | 6.43 s | 90.4 MB |
| USearch M32, expansion=128 | 2.302 ms | 0.4200 | 23.21 s | 90.6 MB |
| USearch M32, expansion=1024 | 10.773 ms | 0.9375 | 23.21 s | 90.6 MB |

At 50,000 rows × 1024 dimensions, NumPy exact measured 1.729 ms p95 and
FAISS Flat measured 1.475 ms p95; both retained Recall@10 = 1.0.

**Decision**

Keep normalized `.npy` as the canonical vector generation and make deterministic
NumPy exact search the production default after replacing the current Python
full sort. Keep FAISS Flat as an optional exact candidate; do not add HNSW or
USearch to the runtime until real Vietnamese vectors and at least 1,000 queries
pass the quality, memory, platform, crash, and ≥2× p95 speedup gates.

Reproduce:

```bash
uv run python eval/eval_vector_backend.py --help
uv run pytest -q tests/test_eval_vector_backend.py
```

Raw machine-specific JSON remains under `eval/results/` (gitignored); the
script, hash, summary numbers, and decision are tracked here and in
`docs/10-vietnamese-rag-model-selection.md`.

### P2.1.3 — Transactional index lifecycle probe

This probe validates lifecycle mechanics, not Vietnamese answer quality. It uses
a deterministic 16-dimensional fake embedder: build a synthetic vault, warm-load
a READY generation, edit one note, then rename one note.

| Field | Value |
| --- | --- |
| Harness | `eval/eval_index_lifecycle.py` |
| Fixture | 24 synthetic Markdown documents / 24 chunks |
| Machine | macOS 15.7.4 ARM64 |
| Python / NumPy | 3.11.14 / 2.4.4 |
| Full build | 15.695 ms |
| Warm snapshot | 12.540 ms |
| One-note edit | 17.871 ms; 1 row embedded |
| One-note rename | 20.011 ms; 24 rows reused |
| Model document rows | 25 total across all builds |

Query-time `snapshot()` reported `ready` without calling
`embed_documents`; an edit made dense `stale` while sparse remained usable;
a rename reused vectors by embedding-input hash.

Reproduce:

```bash
uv run python eval/eval_index_lifecycle.py --documents 24
uv run pytest -q tests/test_indexing_v2.py tests/test_eval_index_lifecycle.py
```

This probe does not authorize changing the embedding model or adopting ANN;
those decisions remain governed by P2.1.2 and the model-quality gates.

### P2.2 — Tool router (v2, offline)

Dataset: `eval/prompts/tool_router_vi.jsonl`. With no predictions file the eval
runs the deterministic `DefaultRuntimeToolRouter` live.

```bash
uv run --all-extras python eval/eval_tool_router.py \
  --dataset eval/prompts/tool_router_vi.jsonl --output eval/results/tool-router.json
```

The v2 dataset has **100 cases across 8 slices**. The offline deterministic
baseline is reproducible and does not call a provider:

| Metric                              | Overall | Notes                                                              |
| ----------------------------------- | ------: | ------------------------------------------------------------------ |
| Accuracy                            |   0.610 | Expected: deterministic only; NL slices intentionally fall through |
| Coverage                            |   1.000 | Every row gets a recorded decision                                 |
| False-trigger rate                  |   0.000 | Hard-negative + no-tool slices have no tool calls                  |
| Deterministic command/read accuracy |   1.000 | Explicit commands and scoped reads                                 |

```bash
uv run python -m eval.eval_tool_router \
  --dataset eval/prompts/tool_router_vi.jsonl \
  --output eval/results/tool-router-v2-baseline.json
```

The semantic and cascade captures use the same precision/recall/F1, slice,
false-trigger, coverage, and p50/p95 latency schema. A live provider run is
strictly opt-in and capped by default:

```bash
uv run python -m eval.eval_tool_router_live --provider openrouter \
  --model <openrouter-model-id> --dataset eval/prompts/tool_router_vi.jsonl \
  --predictions eval/results/tool-router-live.predictions.jsonl --max-cases 25
```

That command sends router prompts to OpenRouter and never executes a tool; it
requires a stored OpenRouter key and may incur provider cost.

### P2.3 — Retrieved memory

Dataset: `eval/prompts/retrieved_memory_vi.jsonl` (62 cases across preference,
project, code-switch, no-hit, adversarial). Vault: `eval/fixtures/memory_vault`.

```bash
uv run --all-extras python eval/eval_retrieved_memory.py --mode retrieval-only \
  --vault eval/fixtures/memory_vault --cases eval/prompts/retrieved_memory_vi.jsonl \
  --output eval/results/retrieved-memory.json
```

Retrieval-only: recall **0.629**, forbidden-leakage **0.0**, mean latency
0.33 ms. The 10 `no_hit` cases carry an empty `expected_contains`, so they always
score `hit=False` by construction, capping maximum recall at 0.839.

Answer mode grounds a real LLM on the retrieved context. Run against OpenRouter
`google/gemini-2.5-flash-lite`:

| Variant                  | Answer accuracy | Forbidden leakage | Mean latency | Cost (12 cases) |
| ------------------------ | --------------- | ----------------- | ------------ | --------------- |
| blob (full profile)      | 0.75            | 0.0               | 1.49 s       | ~$0.0003        |
| retrieved (chunk_sparse) | 0.42            | 0.0               | 1.20 s       | ~$0.0004        |

Blob beats retrieved here because `memory_vault` is a 2-file fixture — dumping
the whole profile gives more context than the top-3 chunks. On a large personal
vault the retrieved variant is expected to win; this small fixture is for CI
reproducibility, not a tuned benchmark.

### P2.4 — Memory-plane and checkpoint contract smoke

Measured on the current tree with `.venv` Python 3.11. The model-free contract
suite covers 65 tests, including normal-turn archive non-access, explicit
semantic archive assembly, approved-core separation, checkpoint round-trip,
legacy checkpoint migration, private permissions, resume, clear/delete, and
RAM-only non-persistence. Ruff and Pyright report zero findings for the touched
memory/runtime modules.

Decision: the access-plane split and opt-in checkpoint implementation are
accepted for the P2 branch. `ram_only` remains the product default; durable
resume is available only when `session_persistence=local_resumable` is
explicitly selected. This smoke is a contract test, not evidence that
episodic memory or automatic core promotion is production-ready.

---

## P3.1 — Conversational Robustness (barge-in + turn-taking)

**Purpose:** turn the already-built barge-in (`DuplexAecSink`) and adaptive
endpoint (Smart-Turn) into measured, reproducible numbers, phrased in
Full-Duplex-Bench vocabulary. Method = **frame-stepped offline replay**: the
decision arithmetic is lifted out of the sounddevice loops and driven from
`(far, near)` buffers, with time = frame index (deterministic, machine-
independent). AEC + VAD injected → unit-tested without hardware, then fed the
production WebRTC AEC3 + Silero for the real runs. Full writeup:
`notes/conversation_research.md`.

**Setup**

| Field         | Value                                                                                            |
| ------------- | ------------------------------------------------------------------------------------------------ |
| Replay core   | `eval/barge_in_replay.py` (`BargeInDecider`, `TurnEndpointDecider`)                              |
| Metrics       | `eval/conversation_metrics.py`                                                                   |
| Tier 1 data   | AEC-Challenge `real/` (16 kHz mic+lpb pairs), 13,626 pairs, 150/condition (seed 42)              |
| Tier 1 synth  | FLEURS `vi_vn` far+user + MIT IR Survey RIR echo (`data/rir/mit`, 270 real RIRs @16k)            |
| Tier 2 data   | FLEURS `vi_vn` shaped into clean + mid_pause(800 ms) timelines, 60 utterances                    |
| Barge-in gate | sustained 400 ms, Silero threshold 0.7 (production `DuplexAecSink` defaults)                     |
| Policies      | `fixed` (700 ms) vs `p_based` (floor 1000 + span·P, ceil 3000; Smart-Turn v3.2)                  |
| Run date      | 2026-07-25                                                                                       |
| Output paths  | `eval/results/conversation_tier1{,_synth}.json`, `conversation_tier2.json` (gitignored)          |
| Reproduce     | `uv run python -m eval.eval_conversation` / `eval.eval_barge_in_synth` / `eval.eval_turn_taking` |

**Tier 1 — barge-in (false-interrupt ≈ FDB Takeover Rate; detection = recall)**

| Run                         | pairs / scenarios | false-interrupt | detection | notes                       |
| --------------------------- | ----------------: | --------------: | --------: | --------------------------- |
| Real echo (AEC-Challenge)   |               300 |            2.7% |     94.7% | static 96.0% / moving 93.3% |
| Synth VN over real-RIR echo |               240 |            2.5% |     92.5% | backchannel-fire 3.8%       |

Synth cross-validates real (2.5 vs 2.7% false-int, 92.5 vs 94.7% detection) on
disjoint audio → the RIR synthesis is realistic and barge-in survives real echo.
Synth median stop-latency 2344 ms / p90 5336 ms (gated by the 400 ms sustained
floor + read-speech VAD; grows under stronger echo).

**Tier 2 — turn-taking (120 scenarios, 800 ms within-turn pause)**

| Policy      | cut-in rate | premature-close | median over-wait |
| ----------- | ----------: | --------------: | ---------------: |
| fixed       |      100.0% |           61.7% |           704 ms |
| **p_based** |    **3.3%** |       **18.3%** |          1312 ms |

**Key findings**

- Adaptive `p_based` drops cut-in **100% → 3.3%** and premature-close **61.7% →
  18.3%** (30x / 3.4x) for ~608 ms more patience — the FDB TOR vs response-
  latency trade-off, measured for Vietnamese.
- `p_based` still closes 18.3% of VN turns early because Smart-Turn is
  English-trained → argues for a Vietnamese turn model (future work).
- The 400 ms sustained gate filters 400 ms backchannels only because it needs
  416 ms (13×32 ms); a 500 ms "vâng ạ" would leak → a backchannel classifier is
  the real fix. Honest, not hidden.

**Caveats**

- Latency is a system number (sustained floor + VAD on read speech), not a pure
  front-end reaction time; FLEURS has more micro-pauses than a short command.
- Backchannel is a synthetic 400 ms FLEURS head, not recorded "vâng/dạ".
- Tier 1 synth uses one echo level (alpha 0.5) and MIT RIRs only; a full SER
  curve + OpenSLR simulated RIRs would strengthen the acoustic claim.

---

## P4 — Local working-memory summary bake-off (not yet a release decision)

The compaction path is deliberately held at `trim_only` until a local candidate
passes the held-out quality and cold-process resource gates.  The benchmark
fixtures are committed synthetic data, not user transcripts:

- `eval/prompts/summary_session_vi_v1.jsonl`: 200 records, eight semantic
  families, family-level train/validation/test split;
- `eval/prompts/summary_rolling_vi_v1.jsonl`: 40 multi-generation correction
  and decision chains;
- the full candidate protocol is zero-shot constrained JSON generation at
  `n_ctx=4096`, `temperature=0`, `max_tokens=384`.

### Arcee-VyLinh 3B Q4_K_M smoke capture

| Field | Value |
| --- | --- |
| Date | 2026-07-28 |
| Machine | MacBook M4 Pro, macOS 15.7.4 arm64, Python 3.11.14 |
| Runtime | `llama-cpp-python 0.3.16`, Metal, `n_gpu_layers=-1`, 8 threads |
| Model | `Arcee-VyLinh.Q4_K_M.gguf` (local, already provisioned) |
| Input | first 5 synthetic `constraints` records only |
| Process mode | one warm process; **not** cold-per-job |
| Schema-valid rate | 5/5 (100%) |
| Structured-field recall | 0.000 (0/5) |
| Latency p50 / p95 | 1981.8 / 2200.0 ms |
| Raw artifact | `eval/results/summary_arcee_smoke_20260728.json` (gitignored; no raw transcript persisted) |

**Decision:** this is a failure, not a winner.  Although grammar-constrained
JSON was valid, the candidate misplaced the explicit user constraint into other
fields and retained synthetic situation labels.  The run is only a runtime
smoke: it is too small, warm-process, single-family, and has no rolling,
adversarial, resource-cleanup or 16 GiB measurement.  It cannot satisfy the
quality-first gate; `background_summary` stays disabled and there is no remote
fallback or automatic model download.

Reproduce the same non-release smoke explicitly (it never downloads weights):

```bash
uv run python eval/eval_summary_bakeoff.py \
  --dataset eval/prompts/summary_session_vi_v1.jsonl \
  --model-key arcee_vylinh_3b_q4_k_m \
  --model-path models/arcee-vylinh-gguf/Arcee-VyLinh.Q4_K_M.gguf \
  --limit 5 --output eval/results/summary_arcee_smoke.json
```

### Provisioned summary-candidate warm captures

On 2026-07-28 all five declared summary weights were explicitly provisioned
under ignored `models/summary/<candidate>/<immutable-HF-revision>/`.  The
provisioner pins repository revision, file byte count and SHA-256, verifies the
result, then sets the GGUF file to `0600`; the runtime never invokes it or
downloads a weight.  Gemma required the account owner to accept its Hugging
Face terms and provide `HF_TOKEN` locally; that token is only read into the
provisioning process and never written to output.

The following `summary_session_vi_v1` runs use Metal, 8 threads,
`n_ctx=4096`, `temperature=0`, `max_tokens=384`, and one warm process for all
200 synthetic rows.  Artifacts in `eval/results/` are gitignored and contain
only synthetic fixture IDs/safe metrics.  They are **not cold-process release
measurements**.

| Candidate | Schema valid | Forbidden-leak records | Unexpected-item records | p50 / p95 ms | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Arcee-VyLinh 3B Q4_K_M | 100% | not captured by old smoke evaluator | not captured by old smoke evaluator | 2143 / 2877 | No winner; strict exact field recall 0.000 |
| Qwen3 1.7B Q8_0 | 100% | not captured by old smoke evaluator | not captured by old smoke evaluator | 1715 / 3024 | No winner; strict exact field recall 0.000 |
| Gemma 3 4B IT QAT Q4_0 | 100% | 11.5% | 96.0% | 2831 / 4132 | Fails zero-leak gate |
| Sailor2 1B Chat Q4_0 | 59.0% | 0.0% | 100.0% | 2529 / 4119 | Fails schema-valid gate (82/200 errors) |
| Sailor2 8B Chat Q4_K_M | 100% | 1.5% | 100.0% | 3956 / 5271 | Fails zero-leak gate; peak RSS observed ~6.4 GB |

`field_recall_mean` was deliberately not used for the decision: the current
synthetic labels demand literal wording while valid summaries paraphrase the
same fact (for example, a user constraint).  The evaluator now reports
per-field strict recall, unexpected structured items, and forbidden-claim
leaks; the fixture still needs accepted semantic variants/atomic annotations
before a quality score can be a release gate.  This does **not** excuse the
observed forbidden leaks or schema failures.  No candidate has passed all
quality, rolling, cold-process, or target-16-GiB gates, so the approved mode
remains `trim_only`.

### P4.1 — Real-data and structured-state v2 decision

Run date: 2026-07-28. Full design, source provenance, candidate revisions,
license notes, commands, and failure analysis:
`docs/12-local-summary-model-selection.md`.

The v1 fixtures and captures above are now **invalidated** for model selection:
200 single rows contained only eight unique expected payloads, while the
40-session rolling set had one effective expected state. The files were removed
from the current tree. V2 contains 200 unique inputs/151 expected states and
40 distinct four-generation sessions. Negative injection/noise cases
intentionally share empty expected state.

Public real-data was explicitly provisioned and pinned:

| Dataset | Local rows | Comparative probe per candidate |
| --- | ---: | ---: |
| VSoLSCSum-VI | 141 | 8 |
| SEAHORSE-VI | 64 | 8 |
| WikiLingua-VI | 64 | 8 |
| XL-Sum-VI | 64 | 8 |
| DialogSum-EN control | 64 | 8 |

Public probe (40 records/candidate; overlap is report-only):

| Candidate | Schema | Token F1 | ROUGE-L F1 | Model2Vec cosine |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B Q8_0 | 77.5% | 0.2522 | 0.1719 | 0.6759 on 31 parses |
| Qwen3-1.7B Q8_0 | 97.5% | 0.2667 | 0.1728 | 0.6746 on 39 parses |
| Qwen2.5-3B-Instruct Q4_K_M | 100% | **0.2707** | **0.1899** | 0.6655 |
| Qwen3-4B Q4_K_M | 100% | 0.2524 | 0.1705 | 0.6331 |
| Qwen3-4B-Instruct-2507 Q4_K_M | 100% | 0.2690 | 0.1713 | **0.6876** |

Final prompt fingerprint: `aa317641bb249d5b`. Only the best finalist,
Qwen3-4B-Instruct-2507 Q4_K_M, advanced to final-prompt full captures:

| Run | Denominator | Schema | Fact recall | Negative clean | Forbidden surface | p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single state v2 | 200 | 100% | 80.0% | 100% | 0% | 1.508 / 2.787 s |
| rolling v2 | 40 sessions / 160 generations | 100% | 72.5% | n/a | 0% | 11.111 / 13.863 s/session |

Single family recall: constraints 100%, open items 100%, commitments 96%,
corrections 92%, decisions 84%, mixed code/path 8%. Rolling ranged from 67.5%
for correction chains to 82.5% for open-item chains.

Eight cold-process jobs measured:

- outer p50/p95 5.318/6.387 s;
- child load p50 300 ms; generation p50 2.357 s;
- max child peak RSS 3,822 MiB;
- clean exit 8/8 and supervisor-observed stopped 8/8.

The explicit real-flow smoke passed local generation, CAS publication, worker
idle, private `0600` checkpoint, reload, structured-state render, and render
round-trip. No worker remained after the run.

**Decision:** resource lifecycle passes on the development machine, quality
does not. There is no production winner. Keep `trim_only`; retain
Qwen3-4B-Instruct-2507 only as a research finalist for a typed, ID-addressed
delta + deterministic merge experiment. Do not add keyword/regex merge rules,
auto-download, remote fallback, or lower the release gate.

Post-decision disk cleanup removed the four rejected GGUF files (0.6B, 1.7B,
Qwen2.5-3B, and Qwen3-4B base), reducing `models/summary` from about 9.0 GB to
2.3 GB. The Instruct-2507 finalist remains; all removed artifacts are
reprovisionable from pinned registry metadata.

### P4.2 — Production acceptance and 15K auto-compaction

Run date: 2026-07-28. This decision supersedes the P4.1 `trim_only` production
decision after explicit product-owner acceptance of the measured quality
trade-off. It does not invalidate the P4.1 measurements.

The selected production model is `Qwen3-4B-Instruct-2507 Q4_K_M`. The revised
gate is:

- schema 100%, negative-state clean 100%, forbidden surface 0%;
- single annotated fact recall >= 80%;
- rolling annotated fact recall >= 70%;
- child clean exit and stopped-worker rates 100%;
- peak summary-child RSS <= 8,192 MiB.

The measured 80.0% single recall, 72.5% rolling recall, clean safety metrics,
and process lifecycle pass this gate. Decision recall 84%, correction recall
92%, and mixed Vietnamese/code/path placement recall 8% remain documented
technical debt.

Production wiring:

- `working_v2_16k` target/high/hard = 12,000/15,000/16,384 tokens;
- automatic and manual compaction use one coordinator in chat, voice, and UI;
- worker context is allocated per job from 4K through a 32K maximum;
- selected weight is stored privately under
  `~/.local/share/soca/models/summary/<key>/<revision>/`;
- runtime never downloads the weight and falls back to `trim_only` if it is
  missing, malformed, non-private, or the worker fails.

Real 15K production smoke:

| Metric | Result |
| --- | ---: |
| auto-trigger | 15,288 approximate tokens |
| allocated llama.cpp context | 20,480 tokens |
| load / generation | 466 ms / 44.78 s |
| peak child RSS | 6,030 MiB |
| exit / stopped | 0 / true |
| publish / checkpoint 0600 / reload / render | pass / pass / pass / pass |

The final full-flow run used the persisted real answer backend
`openrouter:qwen/qwen3.7-flash`. The local summary was present in the answer
prompt, the active `TTS B` decision was visible, the provider was called, and
the returned content was non-empty. No summary process remained afterward.

### P0 — Shared chat/voice capability-routing baseline

Run date: 2026-07-29. This is a diagnostic baseline for the next
`voice_knowledge_gate_plan.vi.md` phase. It does not select production
thresholds or enable semantic voice routing. The run was made from the clean
`main` merge at `e549676` plus the P0 evaluator/data changes on branch
`feat/voice-knowledge-p0-baseline`.

The benchmark is independent of the showcase vault. Routing/source/parity rows
are SoCa-authored annotations. Grounding uses the checked-in
`eval/fixtures/real_rag_vault`, whose answerable slice is XQuAD Vietnamese with
the source/license manifest; the showcase/demo corpus is not used.

**Frozen inputs and hashes**

| Input | Rows | SHA-256 |
| --- | ---: | --- |
| `eval/prompts/p0/turn_routing_vi.jsonl` | 66 | `4249290a397a303ec2ec2c9b76ddc2d9a408bfb0607477a14602b71df7f61f58` |
| `eval/prompts/p0/retrieval_source_vi.jsonl` | 40 | `07ab5450b130197f94d4ff64369c1adbdfb89f0228e011e49176d678dba29ed7` |
| `eval/prompts/p0/grounding_vi.jsonl` | 20 | `965bbd8c238afdad715700d22e07c7235ec5fb83741a96e60e78e192a68e2791` |
| `eval/prompts/p0/memory_context_policy_vi.jsonl` | 16 | `394fd431e5ae1484388a7432663dbe12dce46c5e2c26f5bd77465dab491ab95a` |
| `eval/prompts/p0/voice_parity_vi.jsonl` | 14 | `def361f5f136023493dfcd7013fc2658e5de4652a5f1fa39e280c279992c775d` |

All route/grounding families are grouped into train/validation/test; no family
crosses a split. The route baseline uses local
`intfloat/multilingual-e5-small` through FastEmbed, threshold `0.58`, margin
`0.04`, deterministic → semantic cascade, and no LLM-router fallback. Raw
scores are retained in the ignored JSONL capture so threshold fitting can be
done later without rerunning the corpus.

**Current baseline results**

| Metric | Overall | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| Disposition accuracy | 48.48% | 76.19% | 33.33% | 37.50% |
| Exact source-set accuracy | 78.79% | 90.48% | 76.19% | 70.83% |
| Unsupported → executable tool | 0/15 (0%) | 0/3 | 0/6 | 0/6 |
| Direct-tool exact | 5/9 (55.56%) | — | — | — |

The main failure is not an accidental weather tool call: the router abstains as
`unresolved` too often under the current global margin. Confusion counts are:

```text
direct_tool→direct_tool 5, direct_tool→unresolved 4
retrieval_request→retrieval_request 15, retrieval_request→unresolved 12
smalltalk→smalltalk 4, smalltalk→unresolved 5
out_of_scope→out_of_scope 2, out_of_scope→unresolved 13
unresolved→unresolved 6
```

Source selection is explicitly multi-label rather than binary:

| Expected profile | Correct / total |
| --- | ---: |
| knowledge | 2 / 10 |
| memory | 1 / 10 |
| both | 7 / 10 |
| neither | 10 / 10 |

Overall source exact accuracy is 20/40 (50.00%). This is a baseline signal to
fix the route/source calibration and corpus, not a reason to add substring or
keyword rules. Source recall was knowledge 20%, memory 10%, both 70%, neither
100%; predicted-profile precision was 100%, 50%, 100%, and 34.48%
respectively. The raw source-score distribution from the route capture was
knowledge mean/p50/p95 `0.8415/0.8346/0.9202` and memory
`0.8453/0.8401/0.9202`; their overlapping distributions explain why one global
source floor is not a safe calibration.

**Grounding and parity baseline**

Using cached sparse retrieval plus the production relevance gate over 12
answerable and 8 unanswerable rows:

| Metric | Result | Wilson 95% interval |
| --- | ---: | ---: |
| Answerable path recall@5 | 12/12 (100.00%) | 75.75–100.00% |
| Unanswerable accepted-evidence rate | 7/8 (87.50%) | 52.91–97.76% |
| Retrieval+evidence mean / p95 | 32.88 / 32.81 ms | local cached-sparse run; single retrieval |

The 87.50% false-evidence rate is a release blocker and confirms that an empty
or unrelated vault question cannot be trusted merely because sparse retrieval
returned snippets. The per-row raw/accepted paths and evidence reasons are in
`eval/results/voice_knowledge_p0/grounding.json` (ignored local artifact).

On 7 clean/ASR parity pairs, 1 pair changed disposition; parity mismatch is
14.29%. Disposition accuracy was 78.57% and source exact accuracy 92.86%.
The mismatch is an ASR-shaped file-mention hard negative becoming unresolved,
which must be addressed in the shared policy rather than hidden by a voice-only
regex.

**Commands and decision**

The exact commands and metric definitions are pinned in
`eval/gates/voice_knowledge_p0.json`. The route evaluator is
`eval/eval_full_cascade.py`; source selection is
`eval/eval_source_policy.py`; grounding is `eval/eval_grounding.py`.

Decision: P0 data/schema/baseline gate is complete, but production calibration
is not. Do not enable natural-language knowledge in voice, do not choose a
global threshold, and do not add a hardcoded lexical fallback until P1 closes
the disposition/source contract and the next held-out run reduces
unresolved-route and false-evidence failures.

**Encoder/aggregation bake-off (same threshold/margin, no fitting)**

| Encoder / aggregation | Disposition | Source exact | Direct tool exact | Test disposition |
| --- | ---: | ---: | ---: | ---: |
| E5 asymmetric / max | 48.48% | 78.79% | 55.56% | 37.50% |
| E5 asymmetric / top-2 mean | 43.94% | 78.79% | 44.44% | 37.50% |
| E5 asymmetric / centroid | 43.94% | 77.27% | 44.44% | 37.50% |
| E5 symmetric / max | 48.48% | 78.79% | 66.67% | 33.33% |
| E5 symmetric / top-2 mean | 42.42% | 78.79% | 66.67% | 33.33% |
| E5 symmetric / centroid | 34.85% | 74.24% | 22.22% | 20.83% |
| Model2Vec symmetric / max | 46.97% | 78.79% | 88.89% | 29.17% |
| Model2Vec symmetric / top-2 mean | 25.76% | 68.18% | 33.33% | 20.83% |
| Model2Vec symmetric / centroid | 36.36% | 66.67% | 88.89% | 33.33% |

No encoder/aggregation combination passes a production route gate. Model2Vec
max has a useful direct-tool diagnostic but loses held-out disposition quality;
it is not a production choice. E5 asymmetric/max remains the compatibility
baseline only, not a calibrated winner. The raw bake-off predictions are in
`eval/results/voice_knowledge_p0/bakeoff_*.jsonl` (ignored local artifacts).

The post-review rerun also verifies that all evaluators reject missing or stale
prediction IDs instead of scoring a partial intersection. The grounding
capture reuses the exact retrieved hit set for relevance admission, and the
bake-off creates its output directory before its first capture. Source routing
keeps raw floating-point scores for threshold/margin decisions; six-decimal
values are telemetry only.

**Memory policy baseline**

The model-free policy harness confirms the current split: approved core/profile
is read on all 16 rows, archive retrieval is called only on the 8 explicit
archive rows (50% of all rows, 100% of archive cases), six synthetic relevant
archive hits are injected with 100% harness precision, and the context cost is
61.3 mean / 50 p50 / 91 p95 approximate tokens. Answer delta is intentionally
`not_measured`; it needs a paired real answer-LLM run and is a P1/P6 item, not a
number to invent from a policy-only harness. Report:
`eval/results/voice_knowledge_p0/memory.json`.

P0 therefore closes schema, family split, hard-negative coverage, raw-score
capture, encoder/aggregation bake-off, and model-free memory access capture.
P1 remains blocked on contract calibration, evidence false-positive reduction,
and a real LLM paired memory answer-delta measurement.

### P1 — Disposition/source contract and calibration candidate

Run date: 2026-07-29. Branch: `feat/voice-knowledge-p1-contract-calibration`,
based on merged P0 `main` at `33eef8b`. This phase does not enable a new
production default.

P1a changes the production route-example schema from the ambiguous
`tool:"none"` shape to:

```json
{"route":"retrieval_request","handler":null,"arguments":{},"sources":["knowledge"]}
```

`direct_tool` is the only route allowed to name a handler. `smalltalk`,
`out_of_scope`, and `unresolved` have no handler or source. Retrieval sources
are an explicit multi-label profile: `knowledge`, `memory`, `both`, or
`neither`. P0 fixtures remain backward-compatible and frozen; only the
production `eval/prompts/turn_routing_vi.jsonl` catalog uses the new schema.
Trace payloads now include disposition, handler, selected routes, source
profile inputs, all route/source scores, runner-up, margin, and reason without
logging memory text.

P1b calibration uses only the 42 route rows and 28 source rows in train plus
validation. The 24 route test rows and 12 source test rows remain sealed and
are not used for fitting. Candidate artifact SHA-256:
`d15e1b20586b5f42878325a6f150285615b687961603f5595befa73d2f18fa0c`.

| Candidate metric | Fit result | Release interpretation |
| --- | ---: | --- |
| Route disposition accuracy | 26/42 (61.90%) | fails quality gate |
| Unsupported → direct tool | 0/9 | passes fit safety check only |
| Source profile exact accuracy | 14/28 (50.00%) | fails quality gate |
| Final test usage | 0 rows | sealed |

The fitted artifact selected route thresholds
`direct_tool=0.890477`, `retrieval_request=0.865770`,
`smalltalk=0.884843`, `out_of_scope=0.939997`, `unresolved=0.896812`,
with route margin `0.0`; source thresholds are
`knowledge=0.865064`, `memory=0.868267`, with source margin `0.0`.
These numbers are diagnostic, not a production recommendation. The candidate
is explicitly recorded as `release_candidate_not_enabled`; P1 cannot pass with
the current small/noisy route-source corpus, so no semantic default or voice
behavior was changed.

Exact command and gate policy:
`eval/gates/voice_knowledge_p1.json` and
`eval/results/voice_knowledge_p1/router_calibration.json` (ignored artifact).

### P3 — Shared chat/voice semantic routing wiring

Run date: 2026-07-29. Branch: `feat/voice-knowledge-p3-shared-routing`, based
on merged P2 `main` at `bb4d4f2`.

This phase removes the ASR-only `RetrievalIntentGate` and the
`voice_knowledge_mode`/`knowledge_intent_threshold` configuration path. Text
and voice now build the same deterministic + semantic router policy, including
the same explicit knowledge prefixes, route catalog, source selection and
fail-closed behavior when the local embedder is unavailable. The optional LLM
router remains disabled for voice; this phase does not promote an uncalibrated
router artifact or claim a new quality score.

Regression verification on the branch:

| Check | Result |
| --- | ---: |
| shared routing/parity targeted tests | 45 passed |
| semantic router voice setup test | passed |
| Pyright (`soca`, eval voice loop) | 0 errors |
| Ruff (touched production/eval/tests) | passed |

The parity test uses the same route contract with `source="text"` and
`source="asr"`, verifies the same `retrieval_request`/Knowledge decision, and
asserts one retrieval call per surface. It is a control-flow regression, not a
quality benchmark; calibration and real voice latency remain P4/P6 gates.

An additional local embedding capture was run against the existing frozen P0
route set (66 rows; dataset SHA-256
`4249290a397a303ec2ec2c9b76ddc2d9a408bfb0607477a14602b71df7f61f58`; no answer
LLM and no network):

```bash
python -m eval.eval_full_cascade \
  --dataset eval/prompts/p0/turn_routing_vi.jsonl \
  --examples eval/prompts/p0/turn_routing_examples_vi.jsonl \
  --run-local \
  --predictions /tmp/soca-p3-route-predictions.jsonl \
  --output /tmp/soca-p3-route.json
```

| Metric | Result |
| --- | ---: |
| Full disposition accuracy | 48.48% (32/66) |
| Source exact accuracy | 78.79% (52/66) |
| Unsupported → executable direct tool | 0/15 |
| Router latency | p50/p95 recorded in `/tmp/soca-p3-route.json` |

This confirms the important safety property (no unsupported case became a
real tool call), but also confirms that the current semantic corpus/threshold
is not a quality release gate: many low-confidence cases correctly fall to
`unresolved`. P3 therefore wires parity and fail-closed behavior; it does not
pretend this capture is a calibrated production win. The `/tmp` raw report is
local-only and is not a repository artifact.

### P4 — Retrieval evidence contract and groundedness calibration

Run date: 2026-07-29. Branch: `feat/voice-knowledge-p4-evidence-contract`, based
on merged P3 `main` at `5a18d2c`.

This phase uses the checked-in public XQuAD Vietnamese corpus, not the
showcase or private/demo vault:

- corpus: `eval/fixtures/real_rag_vault`
- labels: `eval/prompts/p0/grounding_vi.jsonl`
- grounding dataset SHA-256: `965bbd8c238afdad715700d22e07c7235ec5fb83741a96e60e78e192a68e2791`
- corpus manifest SHA-256: `a43bc4ee832e04007051f06de5e81c003d33f1c4219d318d3123c28f5839398c`
- 20 rows: 12 answerable and 8 deliberately unanswerable; family splits are
  frozen by the P0 contract.

The evidence contract now carries source-local state (`ready`, `missing`,
`stale`, `degraded`, `unavailable`), query coverage, score separation, and
separate sparse/dense top scores through retrieval → context → runtime trace.
Dense retrieval is not gated by lexical overlap. Cached sparse hits require
both calibrated lexical coverage and sparse score ratio; dense hits use their
own cosine distribution. `memory.search` preserves backend scores instead of
falling back to an unscored legacy label. Dense-only failure is
`unavailable`; an empty but healthy index is `insufficient`.

Calibration commands:

```bash
PYTHONPATH=. .venv/bin/python -m eval.eval_grounding \
  --vault eval/fixtures/real_rag_vault \
  --dataset eval/prompts/p0/grounding_vi.jsonl \
  --variant cached_sparse \
  --output /tmp/soca-p4/grounding-cached-sparse.json

PYTHONPATH=. .venv/bin/python -m eval.eval_grounding \
  --vault eval/fixtures/real_rag_vault \
  --dataset eval/prompts/p0/grounding_vi.jsonl \
  --variant hybrid --backend fastembed \
  --output /tmp/soca-p4/grounding-hybrid-fastembed.json
```

| Retrieval policy | Answerable recall@5 | Unanswerable accepted evidence | First query | Warm p95 | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| cached_sparse (`coverage=0.65`, sparse ratio `0.75`) | 12/12 (100%) | 0/8 (0%) | 75.9 ms | 30.9 ms | calibrated sparse default |
| hybrid FastEmbed (`dense=0.85`, sparse fallback coverage `0.95`) | 12/12 (100%) | 0/8 (0%) | 12.4 s | 46.9 ms | opt-in; cold model load remains visible |

The previous policy admitted 7/8 unanswerable sparse cases. The new run
rejects all eight on this public set, but the Wilson upper bound is 32.44%
with only eight negatives; this is an initial calibration gate, not a claim
of production statistical certainty. Hybrid cold start is recorded separately
from warm retrieval and is not hidden in the mean.

Answer validation also emits a non-blocking shadow groundedness score by
comparing cited claims with selected evidence snippets. It does not block or
rewrite answers until a held-out human/model calibration set is available.

Decision: keep calibrated cached-sparse as the safe default, keep hybrid
opt-in, and retain all evidence/groundedness signals in telemetry. No raw
sparse-vs-dense score comparison is used.
