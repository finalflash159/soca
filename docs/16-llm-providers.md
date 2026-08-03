# LLM providers and settings

The LLM is one dependency behind the same `LLMEngine` contract. The product
default is local llama.cpp, but the user may explicitly select a remote provider
for both chat and voice. Selecting a remote provider is a data-boundary choice:
the transcript and assembled prompt context are sent to that provider.

## Provider boundary

| Backend | Implementation | Data leaves the machine? |
| --- | --- | --- |
| Local | `soca/llm/llamacpp_runner.py` | No |
| OpenAI | `RemoteOpenAILLM` | Yes, when selected |
| Gemini | OpenAI-compatible endpoint | Yes, when selected |
| OpenRouter | OpenAI-compatible endpoint | Yes, when selected |
| Groq | OpenAI-compatible endpoint | Yes, when selected |

`soca/llm/factory.py` is the only backend construction boundary. It reads the
non-secret `LlmSettings`, resolves a key through `SecretStore`, and builds the
selected engine. It raises a typed authentication/configuration error when the
selection cannot be built; it does not silently return the local model.

## Persistence and lookup

- `~/.config/soca/llm.json` stores provider/model/generation settings, never keys.
- Keys are written to the OS keyring; the private `keys.json` file is only the
  documented fallback when no keyring backend exists.
- Lookup is keyring, process environment, optional read-only `.env`, then private
  JSON storage. `.env` is read only when explicitly enabled.
- The UI masks keys and validates a new key with a live provider catalog call
  before persisting it.
- Settings are shared by chat and voice through `SocaEngine`. A running turn is
  not rebuilt in the middle of a call; a new selection applies to the next turn.

## Model capability resolution

The model catalog supplies context length, output limit and reasoning capability
when the provider publishes them. Requested and effective values are separate:

```text
effective_output = min(requested_output, model_max_output)
effective_reasoning = model_requires_reasoning
                       OR (user_enabled AND model_supports_reasoning)
```

If a capability is unknown, the request omits that provider parameter rather
than guessing. The prompt admission layer still refuses a known required
overflow. Details are in [model-aware context budget](14-model-aware-context-budget.md).

### One request lifecycle

For each chat or voice-transcript turn, the provider boundary performs the
following work:

1. resolve the persisted provider/model selection and secret without echoing
   the key into NDJSON, logs or receipts;
2. obtain the model capability record and build the shared prompt manifest;
3. clamp the requested output to the model-advertised limit and resolve
   reasoning according to provider/model capability;
4. send the request through the selected adapter with the retry ledger and
   explicit provider-routing options;
5. normalize usage, finish reason, latency and errors into the common runtime
   result; and
6. close or cancel the stream when the UI/voice controller stops the turn.

The same lifecycle is used for chat and voice after ASR. Voice adds the local
ASR/TTS edges around it; it does not create a second remote-provider policy.
This is also why a remote setting is visible in both runtime status sections.
Changing settings affects the next turn, while an in-flight request remains
bound to the provider/model that started it.

## Request reliability

`RemoteOpenAILLM` owns one bounded retry ledger. It retries only transient
connection/timeout/408/409/429/5xx failures, only before visible stream text,
and records every attempt. Authentication, invalid request, refusal, empty
output, reasoning-only output and programming errors are terminal. OpenRouter
upstream fallback is disabled in production request options.

See [provider runtime reliability](provider-runtime-reliability.md) for the
adapter matrix and typed failure contract.

## UI flow

1. `llm_providers` lists supported providers.
2. `llm_models` loads a catalog only after a provider/key is selected.
3. The user can reuse the saved provider/model or explicitly reconfigure it.
4. `llm_set_key` validates and stores a key; key text is never echoed in the
   protocol.
5. `llm_select` persists the chosen provider/model and effective generation
   controls.
6. `llm_config` reports the current selection and readiness, not a static
   baseline label.

The selection is visible in `/status` for both chat and voice. A provider/model
catalog failure is shown as not ready; it is not replaced by a stale model or a
different provider.

## Operational commands

```bash
uv sync --extra llm-remote
uv run soca ui
uv run python scripts/smoke_test_remote_llm.py --all --require-all \
  --artifact /tmp/soca-provider-smoke.json
```

The smoke command proves provider invocation and error normalization. It is not
a quality benchmark and its raw logs remain local. See [BENCHMARKS.md](../BENCHMARKS.md)
and [release gates](17-evaluation-and-release.md).
