# Soca

Offline-first Vietnamese voice assistant toolkit for local ASR, LLM, TTS, knowledge, memory, and tool-runtime experiments.

Soca is being built as a fully local Vietnamese assistant. The current repo already runs a local voice loop on macOS: microphone input is endpointed with VAD, transcribed by PhoWhisper ONNX, answered by a local GGUF LLM through llama.cpp, synthesized with Valtec TTS, and played back through the system audio device.

The project is intentionally research-heavy: every model choice is backed by small local bake-offs before it becomes the default path.

## Current Status

Works today:

- Local voice loop demo: `mic -> VAD endpoint -> PhoWhisper ASR -> AssistantRuntime -> Valtec TTS -> audio output`.
- ASR registry for `phowhisper_tiny`, `phowhisper_base`, `phowhisper_small`, and `phowhisper_medium`.
- RobustASR wrapper with VAD, de-looping, confidence guards, BoH matching, and hallucination heuristics.
- llama.cpp LLM registry with PhoGPT baseline, Arcee-VyLinh candidate, Qwen low-RAM fallback, and larger research probes.
- Valtec Vietnamese TTS wrapper with multi-speaker support.
- Local Markdown knowledge vault, manual long-term profile memory, and RAM-only session memory.
- ToolRuntime foundation with typed tool specs, argument validation, side-effect levels, and local tools.
- AssistantRuntime with staged guardrails, swappable tool routing, knowledge/memory prompt assembly, citations, trace output, and voice-loop integration.
- CLI, smoke tests, eval scripts, and benchmark reports.

Still in progress:

- Runtime eval harnesses for tool selection, knowledge citation, and guardrail behavior.
- ASR robustness recalibration for larger PhoWhisper models.
- Raspberry Pi or ARM-board validation. Current numbers are Mac local measurements.

## Architecture

```text
Audio input
  -> endpoint / VAD
  -> RobustASR
       -> PhoWhisper ONNX
       -> model-specific confidence / BoH guards
  -> AssistantRuntime
       -> guardrails
       -> ToolRuntime for time/knowledge
       -> manual profile memory + RAM session memory
       -> llama.cpp GGUF LLM fallback
  -> sentence chunking
  -> Valtec TTS
  -> audio output
```

Assistant-runtime layer:

```text
User turn / ASR transcript
  -> guardrails
  -> knowledge retrieval when useful
  -> ToolRuntime for deterministic local actions
  -> LLM response generation
  -> final response checks
```

## Quickstart

### 1. Create the Python environment

Soca uses Python 3.11 and `uv`.

```bash
uv sync --extra dev --extra eval --extra tts
```

If you need to rebuild `llama-cpp-python` with Apple Metal support:

```bash
CMAKE_ARGS="-DGGML_METAL=on" FORCE_CMAKE=1 \
  uv pip install --force-reinstall --no-cache-dir llama-cpp-python
```

### 2. Inspect registered models

```bash
uv run soca asr-models
uv run soca llm-models
```

### 3. Download local models

Fast voice-loop test setup:

```bash
uv run python scripts/download_phowhisper.py --model phowhisper_base
uv run python scripts/download_llm.py --model arcee_vylinh_3b_q4_k_m
```

Quality-testing ASR setup:

```bash
uv run python scripts/download_phowhisper.py --model phowhisper_small
```

For bake-offs:

```bash
uv run python scripts/download_phowhisper.py --profile bakeoff
uv run python scripts/download_llm.py --profile bakeoff
```

### 4. Initialize a local knowledge vault

The default runtime expects a local Markdown vault at `~/KnowledgeVault`.

```bash
uv run python scripts/init_knowledge_vault.py ~/KnowledgeVault
```

Put compiled notes under `~/KnowledgeVault/wiki/`, and manually curated profile memory in:

```text
~/KnowledgeVault/memory/profile.md
```

The runtime reads local Markdown only. It does not auto-write long-term memory.
Your local vault contents are not committed to this repo.

### 5. Run the voice loop

```bash
uv run python scripts/demo_voice_loop.py --profile baseline
```

Then press Enter, speak, stop speaking, and wait for the assistant to answer.

Useful variants:

```bash
# Stable local baseline
uv run python scripts/demo_voice_loop.py --profile baseline

# Lightweight/fast fallback
uv run --extra tts-piper python scripts/demo_voice_loop.py --profile edge

# Higher-quality saved OmniVoice voice
uv run --extra tts-omnivoice python scripts/demo_voice_loop.py --profile quality

# Practical VieNeu Turbo challenger
uv run --extra tts-vieneu python scripts/demo_voice_loop.py --profile balanced_vieneu

# Disable memory
uv run python scripts/demo_voice_loop.py --no-memory

# Use PhoGPT baseline
uv run python scripts/demo_voice_loop.py --llm-model phogpt_4b_q4_k_m

# Use the current quality-testing ASR default explicitly
uv run python scripts/demo_voice_loop.py --asr-model phowhisper_small
```

## Other Demos

Record one endpointed utterance:

```bash
uv run python scripts/demo_endpoint_record.py
```

Run ASR on test audio:

```bash
uv run python scripts/record_test_audio.py
uv run python scripts/smoke_test_asr.py --model phowhisper_base
```

Run an LLM smoke test:

```bash
uv run python scripts/smoke_test_llm.py --model arcee_vylinh_3b_q4_k_m
```

Ask a local LLM using Markdown knowledge context:

```bash
uv run python scripts/demo_knowledge_llm.py \
  "Bữa sáng nhanh nhưng đủ chất nên ăn gì?" \
  --model arcee_vylinh_3b_q4_k_m \
  --show-context
```

Chat with profile memory and session memory:

```bash
uv run python scripts/demo_memory_chat.py --model arcee_vylinh_3b_q4_k_m
```

Run the text-only AssistantRuntime with tools, knowledge, memory, guardrails, and trace:

```bash
uv run python scripts/demo_assistant_runtime.py \
  --model arcee_vylinh_3b_q4_k_m \
  --show-trace
```

Smoke-test the knowledge vault after adding notes under `~/KnowledgeVault/wiki/`:

```bash
uv run python scripts/smoke_test_knowledge.py --query "bữa sáng nhanh lành mạnh"
```

Run an end-to-end voice-loop benchmark from reproducible WAV fixtures:

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

Quickly compare the current runtime profiles in one process:

```bash
uv run --extra tts --extra tts-piper --extra tts-omnivoice --extra tts-vieneu \
  python eval/eval_voice_loop.py \
  --profile baseline,edge,quality,balanced_vieneu \
  --generate-fixtures \
  --fixture-tts-model omnivoice \
  --fixture-voice emgai_dangiu \
  --vault eval/fixtures/knowledge_vault \
  --no-memory \
  --no-playback
```

The benchmark writes grouped reports under `eval/results/voice_loop/<timestamp>/`
and updates `eval/results/voice_loop/latest.md`. Generated fixture audio under
`eval/audio/voice_loop_smoke/` is local runtime data and is not committed.

The published fixture set currently uses OmniVoice `emgai_dangiu`; the earlier
Valtec-generated fixtures were useful for finding ASR fragility, but too noisy
for route-coverage claims.

For publishable numbers, run one profile per shell command so each profile gets
a fresh process; see [BENCHMARKS.md](BENCHMARKS.md) for the exact commands.

## Benchmarks

Detailed benchmark notes are in [BENCHMARKS.md](BENCHMARKS.md). The short version:

- PhoWhisper-tiny ONNX baseline on FLEURS Vietnamese: 23.60% WER, 12.44% CER, about 20x real-time on the D1 setup.
- ASR model-size bake-off showed `phowhisper_base` as the best next real-time candidate among tiny/base/small for the current greedy decoder path.
- PhoGPT-4B Q4_K_M baseline through llama.cpp/Metal: about 61 ms TTFT and 62.8 tok/s in the D2 run.
- LLM bake-off currently keeps PhoGPT as the historical baseline, Arcee-VyLinh as the leading free-chat candidate, and Qwen3-0.6B as a low-RAM fallback.
- TTS bake-off keeps Valtec as the stable baseline, Piper as the fastest edge fallback, VieNeu Turbo as the practical challenger, and OmniVoice as the saved-voice quality path.
- RobustASR for `phowhisper_tiny` reduced tested non-speech hallucinations to explicit rejects in the D2.5 pilot benchmark while keeping speech false positives at zero in that pilot slice.

Important caveats:

- Benchmark numbers are local Mac measurements, not Raspberry Pi claims.
- LLM behavioral rates are heuristic screeners, not human preference scores.
- RobustASR confidence and BoH artifacts are model-specific. The existing calibrated artifacts target `phowhisper_tiny`; larger ASR models should be recalibrated before production use.

## Model Artifacts

Model files are downloaded locally and are not committed.

Default locations:

```text
models/
external/valtec-tts/
~/KnowledgeVault/
```

`external/valtec-tts` is vendored source used by the TTS wrapper. If it is missing, clone it before running TTS:

```bash
git clone https://github.com/tronghieuit/valtec-tts.git external/valtec-tts
rm -rf external/valtec-tts/.git
```

The Valtec model weights are downloaded by the upstream package into the user cache on first use.

## Repository Layout

```text
soca/
  asr/        PhoWhisper ONNX runtime, VAD, RobustASR, BoH, de-looping
  core/       endpointing, streaming pipeline, metrics, audio output
  llm/        llama.cpp runner, prompt styles, model registry, memory wrapper
  tts/        Valtec TTS wrapper and audio result types
  knowledge/  Markdown vault search and context packing
  memory/     manual long-term profile memory and RAM session memory
  tools/      ToolRuntime, tool specs, local time/knowledge tools

scripts/      local demos, smoke tests, download helpers
eval/         ASR and LLM evaluation harnesses
notes/        research notes
zplan/        local planning docs, gitignored
```

## Development Checks

Run focused checks while developing:

```bash
uv run ruff check soca tests --fix
uv run python -m compileall -q soca tests
uv run pytest -q
```

LLM and ASR smoke tests require downloaded models:

```bash
uv run soca asr-smoke --model phowhisper_base
uv run soca llm-smoke --model arcee_vylinh_3b_q4_k_m
```

## Artifact Policy

This repo should commit source code, configs, docs, tests, and lightweight scripts only.

Ignored local artifacts include:

```text
models/
data/
eval/results/
notebooks/outputs/
notebooks/runs/
notebooks/artifacts/
runs/
logs/
*.wav
*.mp3
```

## Roadmap

Near-term work:

- D10 knowledge ingest/lint tooling for reviewed local wiki updates.
- Recalibrate ASR robustness for the selected larger PhoWhisper model.
- Add post-router LLM evaluation instead of sending every prompt directly to the LLM.
- Improve streaming latency and first-audio timing.
- Package a clean local demo flow for public GitHub review.

## License

Soca project code is MIT licensed. Third-party models, datasets, and vendored source packages keep their own licenses and model-card restrictions.
