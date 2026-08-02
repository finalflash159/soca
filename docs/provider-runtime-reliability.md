# Provider runtime reliability

SoCa owns request shaping, retries, cancellation, and error normalization. The
OpenAI-compatible SDK is only a transport client; its automatic retries are
disabled. Production never switches provider, model, or backend after failure.

## Capability adapters

| Provider | Output limit field | Reasoning transport | Structured-output routing |
|---|---|---|---|
| OpenAI | `max_completion_tokens` | top-level `reasoning_effort` when the selected model advertises it | standard JSON schema |
| Gemini | `max_tokens` | top-level `reasoning_effort` when advertised | standard JSON schema |
| OpenRouter | `max_tokens` | unified `reasoning` object or `reasoning_effort`, according to catalog metadata | requires parameter support; optional data-collection denial |
| Groq | `max_completion_tokens` | top-level `reasoning_effort` when advertised | standard JSON schema on models that advertise it |

Reasoning controls are sent only from model capability metadata. Mandatory
reasoning overrides a user request to disable it. Unsupported or unknown
reasoning capability leaves the parameter absent, preserving the model default
instead of guessing. The requested output limit is clamped to the catalog's
model limit both in settings and again at the provider boundary.

Gemini documents the OpenAI-compatible `reasoning_effort` mapping and notes
that some Gemini models cannot disable thinking. OpenRouter exposes mandatory
reasoning and supported efforts in model metadata. Groq documents different
allowed effort values per model family, so SoCa must not infer support merely
from the provider name. See the official [Gemini OpenAI compatibility guide](https://ai.google.dev/gemini-api/docs/openai),
[OpenRouter reasoning guide](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens),
and [Groq reasoning reference](https://console.groq.com/docs/reasoning).

## Retry ownership

One call has one observable ledger:

- at most three attempts;
- exponential delays of 250 ms, 500 ms, then bounded by 4 seconds;
- a provider `Retry-After` value takes precedence but is capped at 4 seconds;
- retry only connection/timeout failures and HTTP 408, 409, 429, or 5xx;
- never retry authentication, permission, invalid request, missing model,
  refusal, output exhaustion, reasoning-only, empty response, or programming
  exceptions;
- a stream may retry only before emitting user-visible text. Once text was
  emitted, interruption is terminal to prevent duplicated output.

OpenRouter requests set `provider.allow_fallbacks=false`, so its upstream
router cannot silently move the request to another provider. OpenRouter's
official error guide describes typed errors, rate-limit status,
and possible `Retry-After` headers. It also notes that provider-side routing can
retry or choose an upstream provider before streaming; callers that require a
specific upstream must constrain routing explicitly. SoCa's own retry ledger is
separate and remains observable. See [OpenRouter errors and debugging](https://openrouter.ai/docs/api/reference/errors-and-debugging).

## Typed failures and cancellation

`RemoteLLMError` preserves provider, model, HTTP status, provider error code,
retryability, attempt count, finish reason, and retry delay. Empty response is
classified as refusal, output-limit exhaustion, reasoning-only output, generic
empty output, or malformed protocol. Unknown programming exceptions are
re-raised unchanged.

Closing a consumer stream closes the provider stream. Voice stop first calls
the LLM cancellation contract, then stops audio. Engine shutdown also cancels
an active chat call before joining its worker. Synchronous non-stream calls are
still bounded by the transport timeout; Python cannot safely kill an arbitrary
blocking SDK call from another thread.

## Real provider gate

Run the opt-in gate only with explicitly provisioned keys:

```bash
uv run --extra llm-remote python scripts/smoke_test_remote_llm.py \
  --all --require-all \
  --artifact artifacts/local/provider-smoke.json
```

Each provider receives one chat turn and one voice-transcript turn through the
same application engine factory. The receipt records provider/model, surface,
route, terminal state, usage, latency, response, retry ledger, and typed error.
The voice-transcript receipt is not a microphone, ASR, TTS, or audio-device
test. The artifact identifies itself as `real_provider_smoke`, records source,
scenario, model, environment, configuration, raw-log provenance, and decision
metadata, and is explicitly ineligible for benchmark or model-selection claims.
API keys and private transcripts are never written to the artifact.
