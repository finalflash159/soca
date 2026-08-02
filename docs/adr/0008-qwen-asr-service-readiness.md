# ADR 0008: Truthful Qwen ASR service readiness

Status: accepted for service integration

Date: 2026-08-02

## Decision

The Qwen ASR worker starts only from a typed launch contract containing one
packaged artifact specification and one absolute local model generation. Active
launches additionally require a verified artifact receipt. The backend, client
and server no longer accept a Hugging Face repository identifier or provide a
remote model default. Worker processes force Hugging Face and Transformers
offline and remove model-registry tokens from their environment.

The ready marker is only a startup synchronization hint. A worker becomes ready
only after a bounded ping returns a strict, versioned identity containing the
artifact role and digest, pinned upstream and optional mirror revisions, worker
lock and context-policy digests, backend/device/dtype, package versions, PID,
uptime, in-flight count, logprob support, lifecycle state and last typed failure.
Unknown fields, missing fields or any expected-versus-live mismatch are terminal;
the client tears down the child and never selects another artifact or backend.

Lifecycle is explicit: `starting`, `ready`, `busy`, `failed`, `stopping` and
`stopped`. Backend and metadata failures remain visible as `failed` until the
service is explicitly restarted. Shutdown remains bounded graceful request,
terminate and kill escalation. Stale marker, startup crash, timeout and identity
mismatch paths all remove process/socket state.

Static `/status` inspection validates the private runtime receipt, runtime lock,
worker Python and persistent artifact receipt without importing or loading Qwen.
It reports the release artifact separately as `missing`, `invalid`,
`provisioned` or `unsupported`. The existing `Voice ASR` line remains truthful:
it is not relabelled as Qwen until the voice-runtime integration actually selects
and owns this service.

## Evidence

On the supported Apple M4 Pro host, `/status` reported the exact 0.6B release
revision as provisioned, service stopped and artifact verified without loading a
model. Contract coverage passed 86 focused non-model tests. The repository gate
passed 1,504 tests with four expected skips; the complete UI passed 88 tests,
TypeScript typecheck and production build.

Both persistent artifacts then ran through the real subprocess client from an
empty `HF_HOME` with both offline flags enabled. Each completed strict active
identity handshake, partial transcription without context, final transcription
with the typed tech context, selected-token logprob output, graceful exit and
socket/marker cleanup. The final combined warm-cache run passed in 13.01 seconds.

The 1.7B reference recognized the clip's “log level”, “debug” and “info” terms.
The 0.6B release recognized “debug” and “info” but substituted “lock level” for
“log level” in one observed transcript. This readiness run is not a release
quality benchmark; the substitution is retained as an explicit input to the
later calibration and qualification decision rather than hidden by this gate.
Sanitized machine-readable evidence is in
`docs/evidence/qwen-asr-service-readiness-20260802.json`. Raw local logs remain
ignored.

## Consequences

- Provisioned is not equivalent to live ready; status says when the service is
  stopped.
- A wrong artifact, stale marker, corrupt receipt, incompatible protocol or
  unavailable runtime cannot become ready.
- This boundary does not switch the production voice recognizer. Voice ownership,
  dynamic context, confidence calibration and end-to-end voice behavior remain
  the next integration boundary.
- The 0.6B artifact remains a release candidate, not quality-qualified, until the
  full corpus and resource gates are run.
