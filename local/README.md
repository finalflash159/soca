# Experimental ASR robustness workflow — Mac M-series CLI

This workflow runs directly on a Mac without Colab. It mirrors notebooks
`01`–`04`, but packages the process as a sequential CLI that does not depend on
Google Drive.

## When to use `local/` instead of `notebooks/`

| Situation | Use |
| --- | --- |
| Colab Free has run out of quota | `local/` |
| The dataset must be stored locally on a Mac for fast reruns | `local/` |
| You need to benchmark speed on an M4 Pro versus Colab CPU | `local/` |
| Large-model fine-tuning needs a real GPU (T4/A100) | `notebooks/` (deferred) |

## One-time setup

The repository already has a `.venv` after `uv sync`. If it does not:

```bash
uv sync --extra dev --extra eval
```

`silero-vad` and `torchcodec` are included in the base dependencies;
`pyahocorasick` is only in the `eval` extra because BoH is research tooling,
not part of the production runtime.

## Complete pipeline — five sequential CLIs

```text
collect_noise   →  experimental BoH build  →                  ┐
download_fleurs →  calibrate_thresh → eval_table7 (benchmark) ┘
```

### Step 1: Collect noise (~3–10 minutes)

```bash
uv run python -m local.collect_noise
```

Defaults: 500 streamed ESC-50 samples (excluding categories containing voice)
plus 300 synthetic samples (silence, white noise and pink noise), for 800
samples total.

Output:

```text
data/noise_for_boh/wav/*.wav
data/noise_for_boh/manifest.jsonl
data/noise_for_boh/noise_collection_config.json
```

Common options:

```bash
uv run python -m local.collect_noise --target 200      # smoke run
uv run python -m local.collect_noise --force            # rebuild an existing manifest
uv run python -m local.collect_noise --seed 7           # use a different RNG seed
```

### Step 2: Build BoH (~10–25 minutes on an M4 Pro with CoreML)

```bash
uv run python -m eval.experimental.asr_boh.build
```

Output:

```text
data/asr/boh/phowhisper_tiny_vi_boh_v1.json     # research artifact, model-specific
notebooks/outputs/{RUN_ID}/logs/boh_runs/phowhisper_tiny/phowhisper_noise_outputs.jsonl
notebooks/outputs/{RUN_ID}/config_snapshot.json
```

Common options:

```bash
uv run python -m eval.experimental.asr_boh.build --max-files 20
uv run python -m eval.experimental.asr_boh.build --providers cpu
uv run python -m eval.experimental.asr_boh.build \
    --model phowhisper_tiny --model phowhisper_base
uv run python -m eval.experimental.asr_boh.review \
    --boh-path data/asr/boh/phowhisper_tiny_vi_boh_v1.json
```

### Step 3: Download Vietnamese FLEURS speech (~2–3 minutes)

This is the Vietnamese speech evaluation set used for threshold calibration and
benchmarking.

```bash
uv run python -m local.download_fleurs --target 200
```

Output:

```text
data/fleurs_vi/wav/*.wav
data/fleurs_vi/manifest.jsonl
data/fleurs_vi/fleurs_download_config.json
```

### Step 4: Calibrate threshold heuristics (~5 seconds)

Measure `repetition_ratio`, `n_gram_repetition` and `chars_per_100ms` over the
FLEURS ground truth. The recommended threshold is `p99 + margin`.

```bash
uv run python -m local.calibrate_thresholds
```

Output:

```text
data/asr/threshold_calibration.json
```

After calibration, inspect the recommended values in the report. Apply them as
explicit arguments to `check_heuristics()` or update the corresponding
calibration artifact used by the selected evaluation profile. Do not silently
copy thresholds between ASR models.

### Step 5: Table VII benchmark (~10–30 minutes on an M4 Pro)

Compare six configurations: `raw`, `deloop`, `vad`, `boh`, `deloop_boh` and
`vad_deloop_boh`. This is the central D2.5 deliverable for the robustness
evaluation.

```bash
uv run python -m local.eval_table7 --n-speech 50 --n-noise 20
```

Output:

```text
eval/results/table7_replication.json
```

Sample output:

```text
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Config              ┃    WER ┃   CER ┃ Halluc rate ┃ Lat p50 ms ┃ Lat p95 ms ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ (1) Raw ASR         │  ~23%  │ ~12%  │  ~30-50%    │  ~700 ms   │  ~1200 ms  │
│ (3) Silero VAD only │  ~25%  │ ~13%  │   ~0-3%     │  ~500 ms   │  ~1200 ms  │
│ (6) Production + experimental BoH │ ~25% │ ~13% │ ~0-2% │ ~500 ms │ ~1200 ms │
└─────────────────────┴────────┴───────┴─────────────┴────────────┴────────────┘
```

This is a research ablation benchmark. Production `RobustASR` uses VAD,
confidence guard, de-loop and heuristics; BoH, if enabled, is applied only by
the evaluator after the production pipeline so historical comparisons remain
possible.

Options:

```bash
uv run python -m local.eval_table7                                # default: 50 speech + 20 noise
uv run python -m local.eval_table7 --n-speech 200 --n-noise 100  # serious run
uv run python -m local.eval_table7 \
    --configs production_no_boh,production_with_boh              # paired A/B
uv run python -m local.eval_table7 --providers cpu               # force CPU
```

## GPU execution on an M4 Pro

ONNX Runtime on macOS arm64 exposes these providers:

```text
['CoreMLExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']
```

The CLI prioritizes `CoreMLExecutionProvider` (Apple Neural Engine + GPU) and
uses `CPUExecutionProvider` when an ONNX node cannot run on CoreML. If a node
does not support CoreML, that node is executed on CPU rather than crashing.

Silero VAD uses the PyTorch CPU backend. Because the VAD model has only about
1.8M parameters, moving it to MPS provides no measured benefit; its CPU cost is
about 30 ms for five seconds of audio.

Approximate PhoWhisper-tiny (39M parameter) measurements:

| Provider | Latency per 5 s noise sample | Total for 800 files |
| --- | ---: | ---: |
| CPU, 4 threads | ~150–200 ms | ~30–40 minutes |
| CoreML plus CPU node handling | ~60–100 ms | ~10–25 minutes |

These are planning estimates, not release evidence. A serious run must retain
the exact machine, provider list, model revision, dataset manifest, seed,
configuration and raw local log.

## Sanity checklist after `build` completes

```bash
ls data/asr/boh/

uv run python -c "
import json
data = json.load(open('data/asr/boh/phowhisper_tiny_vi_boh_v1.json'))
print(f\"Hallucination rate: {data['metadata']['hallucination_rate']:.2%}\")
print(f\"BoH size: {len(data['boh'])}\")
print('Top 10:')
for item in data['boh'][:10]:
    print(f\"  count={item['count']:3d}  '{item['phrase'][:80]}'\")
"
```

Expected diagnostic ranges:

- non-speech hallucination rate: **30–50%** (the BoH paper reports 40.3% for
  Whisper-large-v3 on 301k files);
- top 30 phrases cover roughly 70–77% of hallucinations;
- after filtering with `count >= 2` and `len >= 5`, BoH contains roughly 30–100
  phrases for 800 samples.

If the rate is below 10% or BoH contains fewer than five phrases, investigate
the pipeline before scaling to 800 samples. These ranges are diagnostic
expectations, not acceptance gates.

## Notebooks 05 and 06 (fine-tuning and ONNX export) — deferred

Plan v3 (`zplan/asr_robustness_colab_plan.md`, line 510) explicitly defers this
work until the non-training robustness pipeline is stable.

Reasons:

- large-model training needs a real GPU (Colab Pro or local NVIDIA); Mac M4 Pro
  Metal does not support all required PyTorch training operators;
- curated Vietnamese command-domain speech is not yet available;
- D2.5 remains defensible without tuning once Table VII is complete.

Only after the robustness deliverable is complete should notebooks 05 and 06 be
reopened.

## Relationship with `notebooks/`

`local/` and `notebooks/02` share the model registry, `MIN_COUNT`, `MIN_CHARS`
and normalization configuration. Their output JSON uses the same schema. BoH
files produced by either workflow are evaluator/ablation artifacts and must
not be auto-loaded by the voice runtime. Metadata distinguishes the execution
mode (`"local"` versus `"colab"`).

Notebooks 03 and 04 are currently placeholders. The implementation for those
two deliverables is `local/calibrate_thresholds.py` and
`local/eval_table7.py`. If the workflow must run on a Colab GPU later, port the
logic from `local/` to the notebook rather than creating a second behavior.
