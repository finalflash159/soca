# 08 — Registries, Profiles & CLI

This is the operational layer: declare ASR/LLM models in registries, bind
explicit ASR/LLM/TTS choices to named runtime profiles, and resolve deliberate
overrides through the CLI or UI.

## Three-Layer Model

```mermaid
flowchart LR
    subgraph Profile["VoiceRuntimeProfile (explicit product choices)"]
        P[baseline · qwen-release · qwen-reference]
    end
    P --> RC[resolve_voice_runtime_config<br/>+ ASR/LLM/voice overrides]
    RC --> ASR[ASR registry]
    RC --> LLM[LLM registry]
    RC --> TTS[valtec_multispeaker]
```

## Registries: Metadata Only

| Registry | File              | Example Keys                                                                                                                          |
| -------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| ASR      | `asr/registry.py` | `phowhisper_tiny` · `phowhisper_base` · `phowhisper_small` · `phowhisper_medium`                                                      |
| LLM      | `llm/registry.py` | `phogpt_4b_q4_k_m` · `arcee_vylinh_3b_q4_k_m` · `qwen3_0_6b_q8_0` · `qwen3_4b_q4_k_m` · `vinallama_*` · `dataops_*` · `bkai_llama2_*` |
| TTS      | `tts/config.py`   | một cấu hình cố định `VALTEC_TTS_CONFIG` (`valtec_multispeaker`)                                                                      |

Each entry describes local paths, runtime role, prompt style, license notes, and
related metadata. Registries are metadata only; model weights load when a
runtime is built.

## Profiles (`core/profiles.py`)

`VoiceRuntimeProfile` combines one explicit ASR selection, one LLM model, the
fixed Valtec TTS key, retrieval policy and generation parameters such as
`max_tokens` and `temperature`. **Default = `baseline`**
(`DEFAULT_VOICE_RUNTIME_PROFILE_KEY`).

| Profile    | ASR              | LLM             | TTS / voice              | Mục đích                  |
| ---------- | ---------------- | --------------- | ------------------------ | ------------------------- |
| `baseline` | phowhisper_small | arcee_vylinh_3b | valtec_multispeaker / NF | Runtime mặc định duy nhất |
| `qwen-release` | qwen3_asr_0_6b service | arcee_vylinh_3b | valtec_multispeaker / NF | Explicit Qwen release selection |
| `qwen-reference` | qwen3_asr_1_7b service | arcee_vylinh_3b | valtec_multispeaker / NF | Explicit larger reference selection |

`validate_voice_runtime_profiles()` checks exactly these profiles against their
registries and enforces `valtec_multispeaker` plus a known Valtec voice. Retired
names such as `quality` and `edge` fail fast; a profile never silently aliases
another ASR or model after a startup failure.

### Settings drive both chat and voice

`soca ui`, `soca ask`, and `soca chat` resolve defaults from `baseline`. A
selected provider/model setting is shared by chat and voice through
`SocaEngine`; a CLI `--llm-model` remains a deliberate local diagnostic/eval
override. ASR profile selection is still explicit and independent from the LLM
provider setting.

## CLI (`soca/cli.py`)

```mermaid
flowchart TD
    M([soca]) --> RUN{command group}
    RUN --> V[voice — CLI voice loop]
    RUN --> A[ask — one text turn]
    RUN --> C[chat — multi-turn text session]
    RUN --> U[ui — TUI status/chat/voice]
    RUN --> INFO[profiles · asr-models · llm-models]
    RUN --> SMOKE[asr-smoke · llm-smoke]
    RUN --> BENCH[benchmark-asr · calibrate-asr]
```

| Command                                   | Description                                                                                    |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `soca voice [profile]`                    | Microphone voice loop. Supports `--no-speak-repairs`, `--press-enter-to-record`, and `--usage` |
| `soca ask <text>`                         | One text turn, with optional trace/usage output                                                |
| `soca chat`                               | Multi-turn text session with session memory                                                    |
| `soca ui [status\|chat\|voice] [profile]` | Ink terminal UI, requires the `ui` extra                                                          |
| `soca profiles`                           | List profiles without loading models                                                           |
| `soca asr-models` / `llm-models`          | List models and local file status                                                              |
| `soca asr-smoke` / `llm-smoke`            | Smoke-test one model                                                                           |
| `soca benchmark-asr` / `calibrate-asr`    | Benchmark/calibrate ASR thresholds, Table-VII style                                            |

## Optional Dependencies (`pyproject.toml`)

Install only what you need so the environment stays manageable:

| Extra  | Purpose                                                         |
| ------ | --------------------------------------------------------------- |
| `dev`  | pytest, ruff, jupyter                                           |
| `eval` | jiwer (WER), datasets, matplotlib                               |
| `llm`  | llama-cpp-python, Metal build when needed                       |
| `tts`  | Valtec dependencies: vinorm, viphoneme, underthesea, torchaudio |

Examples:

```bash
uv sync --extra dev --extra eval
uv run soca ui voice baseline
```

## Changing the Supported Stack

1. Benchmark the candidate without adding a second public profile.
2. Update the `baseline` entry only after the candidate passes its quality and latency gates.
3. Run tests that call `validate_voice_runtime_profiles()` and reject retired profile names.
4. Smoke-test with `soca asr-smoke --model ...`, `llm-smoke`, or `soca voice baseline`.
5. Record the decision in `BENCHMARKS.md`.
