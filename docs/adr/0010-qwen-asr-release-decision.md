# Qwen ASR release qualification decision

Date: 2026-08-03

The release evidence run used the pinned private code-switch manifest, pinned
Vietnamese FLEURS test manifest, pinned non-speech manifest, and both verified
local Qwen snapshots. TTS-generated audio and the demo vault were not used as
release evidence. Raw predictions, private transcripts, and audio remain in the
ignored local run directory.

## Decision

Keep `qwen3_asr_0_6b` as the explicit release candidate, but do not promote it to
the production default yet. The decision is `blocked`, not a quality rejection:

- Quality gates pass for the 0.6B candidate: Vietnamese WER, code-switch slices,
  term behavior, hard-negative rejection, context-echo review, and final RTF.
- The full operational duration matrix reports partial p95 `11,618.877 ms`,
  above the configured `2,000 ms` ceiling. The current CPU Transformers path
  re-decodes the growing prefix, so partial captions are not a safe default on
  this machine. The runtime may disable partials after its observable warmup
  check; this does not turn the failed benchmark gate into a pass.
- A final decode budget of 128 remains selected for realtime. Larger budgets
  improve some joined-turn transcripts but exceed the 30-second production
  deadline and do not provide a safe general long-turn solution. When the model
  reaches the cap, the typed telemetry is surfaced and the transcript is
  rejected rather than sent downstream as if complete.
- The 1.7B model has the stronger quality result, but its reference operational
  run hit `QwenServiceTimeout` at the production deadline. It remains an
  explicit reference/demo choice and is never an automatic fallback.

## Real-flow evidence

Eight recorded voice samples ran through the 0.6B ASR service, OpenRouter
(`google/gemini-3.5-flash-lite`) and Valtec TTS. All eight turns reached a normal
terminal status, including abstention, blocked-action and uncertain-input paths;
six called the remote LLM. A two-sample local voice smoke also passed. These
runs prove wiring and provider execution, but they are not a substitute for the
missing full-stack memory gate.

## Plan mutations and follow-up

The partial decode budget was raised from 32 to 64 after the 32-token sweep
showed a mid-word cut. The partial latency gate was recorded as 2 seconds after
the user's request to preserve answer quality, but the subsequent full matrix
still fails it. This is an observed `blocked` result, not permission to keep
nudging the threshold.

The next release boundary must either provide a measured stateful streaming
implementation for the supported platform or explicitly define and benchmark a
final-only voice capability with matching product requirements. It must also
measure peak memory for Qwen + VAD/AEC + TTS + remote LLM and summary-worker
overlap. No model fallback, silent background retry, or automatic downgrade is
allowed while resolving these blockers.

Evidence index: `docs/evidence/qwen-asr-release-20260803.json`.

## MPS rerun update — 2026-08-03

The release matrix was rerun on the Apple M4 Pro with explicit MPS/float16
artifacts after the device-contract fixes. Both artifacts completed the
20-cycle start/stop probe, typed crash probe, concurrency probe, and partial
latency probe without an ASR fallback or orphan process. The 0.6B partial p95
was 1,018 ms and the 1.7B reference partial p95 was 1,704 ms against the
configured 2,000 ms ceiling. Manual context-echo review and the automatic
quality gates passed for both identities.

The remote trajectory was rerun with eight recorded private samples per
profile using OpenRouter `google/gemini-3.5-flash-lite` and Valtec TTS. Both
profiles reached eight normal terminal rows, including typed uncertain-input
and blocked-action paths. Full-stack RSS in these single trajectories was
2,877 MB for 0.6B and 2,770 MB for 1.7B; the two-sample local smoke reached
5,859 MB. These are trajectory measurements, not the missing repeated memory
pressure gate. Summary-worker overlap was not exercised. The 1.7B probe also
emitted one Python `resource_tracker` leaked-semaphore warning; it is retained
as a blocker rather than suppressed.

Therefore the decision remains `blocked`: MPS materially resolves the prior
CPU partial-latency failure, but rollout still requires repeated full-stack
resource evidence, summary-worker overlap evidence, and root-cause closure for
the semaphore warning. The complete sanitized record is
`docs/evidence/qwen-asr-mps-20260803.json`. No runtime model fallback or
automatic downgrade is enabled.
