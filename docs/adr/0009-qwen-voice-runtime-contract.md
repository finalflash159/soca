# ADR 0009: Typed Qwen voice runtime and context contract

Date: 2026-08-02
Status: accepted for integration; release qualification pending

## Decision

Voice configuration selects ASR through a typed engine/model/artifact-role tuple.
`baseline` remains PhoWhisper. `qwen-release` and `qwen-reference` are explicit
profiles and never act as fallbacks for each other or for PhoWhisper.
PhoWhisper is equally strict in production: missing confidence calibration blocks
readiness before its model is loaded instead of silently disabling the guard.

A voice bundle owns one long-lived Qwen service client. Partial captions call the
backend with an explicit empty context. Final transcription obtains a fresh immutable
context snapshot whose text, source provenance, limits, approximate token count,
policy digest and content digest are observable. Bundle shutdown is idempotent and is
used by normal stop, startup rollback, warmup failure and controller termination.

The context is built dynamically from available vault catalog metadata and recent
session turns. Unicode normalization, ordering and budgets are deterministic. There
is no fixed list of project terms in production. Empty context is valid.

Qwen confidence thresholds are looked up by a canonical identity containing the
artifact digest, runtime lock, device/dtype, decoder, language, output budget, VAD
policy, context policy and context-echo policy. A missing exact identity is
`not_ready`; an older threshold with a similar model name is not accepted.
The runtime derives the VAD digest from the detector instance it actually starts;
changing a VAD threshold therefore invalidates an older calibration identity.

## Why

The official Qwen implementation places `context` in the ASR text prompt for both
Transformers and vLLM inference. Its official toolkit describes context as guidance
for recognizing specific terms. This makes context part of inference behavior, not
mere display metadata, so calibration must include its policy identity.

- [Official Qwen3-ASR inference source](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py)
- [Official Qwen3-ASR repository](https://github.com/QwenLM/Qwen3-ASR)
- [Official Qwen3-ASR 0.6B model card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)

## Context-echo audit

Historical real-voice predictions confirm context echo is a real failure: 3/40 for
0.6B and 1/40 for 1.7B under the old fixed technical prompt. In the 1.7B confidence
calibration, 14 noise outputs echoed context and all 14 passed the model-confidence
threshold. The current overlap detector caught those observed echoes and had no
observed false positive on the 200 speech calibration rows.

These observations are diagnostic, not release evidence: the historical runs did
not capture exact model revisions or hardware identity and used a fixed context.
Their local raw paths and dataset digest are retained in the integration evidence;
the qualification run must replace them with fully reproducible metadata.

This does not qualify the overlap detector for dynamic context. Its threshold was not
selected on an independent labelled set, and dual-decode/model-native alternatives
have not been measured under the new context policy. Therefore no context-echo winner
is declared here. The Qwen profiles remain `not_ready` until the qualification run
produces exact calibration identities and a release decision. The baseline path has
empty ASR context, so this unqualified detector is inactive there.

## Consequences

- Status can truthfully show provisioned weights while the profile remains
  `not_ready` because calibration is missing.
- Selecting one profile never downloads, provisions or starts another ASR backend.
- Integration tests may inject a calibration record to verify wiring; production
  cannot bypass the gate.
- The next qualification run must use the service path and dynamic context policy,
  publish sanitized metrics, and retain raw predictions locally.
- Default cutover is a separate decision after release gates pass.
