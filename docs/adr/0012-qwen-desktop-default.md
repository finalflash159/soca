# ADR 0012: Qwen Release is the required desktop ASR

Date: 2026-08-25
Status: accepted by product direction

## Context

The prior desktop shell moved its app data under Tauri ownership while the
verified Qwen runtime and immutable model store remained in their explicit
source locations. The shell consequently selected the retired PhoWhisper
baseline, then described Qwen profiles as unavailable even when both Qwen
artifacts had already been verified locally.

## Decision

`qwen-release` using `qwen3_asr_0_6b` is the required default voice profile.
The retired `baseline` profile migrates atomically to `qwen-release` when its
selection is read. No production voice profile selects PhoWhisper.

Desktop stores the user-selected Qwen runtime and the parent Qwen model-store
directory independently from its general on-device/TTS model root. It passes
only explicit absolute selections to the sidecar as `SOCA_QWEN_RUNTIME_ROOT`
and `SOCA_QWEN_ASR_MODEL_ROOT`; the latter points at the store's `asr/`
directory. Missing, invalid, unsupported or uncalibrated Qwen components remain
typed non-ready states. They never cause a model, backend or profile fallback.

The UI exposes both locations as an actionable Qwen setup flow. Selecting a
directory restarts the engine so status is derived from the selected runtime,
receipt and artifact rather than stale client state.

## Consequences

- Existing verified source installations can be connected to a packaged desktop
  shell without copying model weights or constructing an unverified runtime.
- A fresh installation needs an explicit Qwen runtime and an immutable Qwen
  model store before Voice can use the microphone.
- ADR 0010's qualification caveats still apply to operational rollout and do
  not authorize an automatic downgrade.
