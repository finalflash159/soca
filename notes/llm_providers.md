# Remote LLM Providers — third-party API backends for SoCa's core LLM

> Design note (P3). Narrative English. SoCa stays **local-first**; remote providers
> are strictly **opt-in**. Pricing is never fabricated — see §5.
> Status: **implemented** — landed on `p3-llm-providers`; local and remote chat
> verified end-to-end (`soca engine` NDJSON smoke).

## 1. Problem

SoCa's default runtime is on-device: PhoWhisper for ASR, a local GGUF model
(Arcee-VyLinh 3B) for the LLM, Valtec for TTS. The local LLM is small enough to run
on a laptop, which caps reasoning quality. For text-first use (`ask` / `chat`) some
users would rather borrow a stronger hosted model through an API — while keeping the
local model as the default and the only thing in the voice hot path.

The requirement, then: let the user swap the _core LLM_ for a third-party API,
without touching the guardrail / tool / knowledge / memory pipeline, and without
ever making remote the silent default.

## 2. One client covers four providers

OpenAI, Groq, OpenRouter, and Google Gemini all expose an **OpenAI-compatible**
chat-completions surface, so a single client class (`RemoteOpenAILLM`, built on the
`openai` SDK) covers all four — only `base_url` and the API key differ. Gemini uses
its compat shim at `/v1beta/openai/`.

| Provider   | Base URL                                                   |
| ---------- | ---------------------------------------------------------- |
| OpenAI     | `https://api.openai.com/v1`                                |
| Groq       | `https://api.groq.com/openai/v1`                           |
| OpenRouter | `https://openrouter.ai/api/v1`                             |
| Gemini     | `https://generativelanguage.googleapis.com/v1beta/openai/` |

`RemoteOpenAILLM` implements the same `LLMEngine` protocol as the local runner
(`generate` / `generate_stream`), so nothing downstream — runtime, guardrails,
voice loop — has to change. `stream_options.include_usage=true` gives real token
counts when streaming. Errors are normalised into `RemoteLLMError` with a
`category` (`auth` / `rate_limit` / `network` / `unknown`) so the UI can react
sensibly (e.g. "missing key" vs "rate limited").

## 3. The layers (bottom-up)

- **`soca/llm/providers/provider_registry.py`** — static registry of the four
  providers (key, label, base URL, env-var name, whether they expose a pricing API).
- **`soca/llm/providers/remote_openai_llm.py`** — the OpenAI-compatible engine.
- **`soca/llm/providers/model_catalog.py`** — `fetch_catalog(provider, api_key)`
  hits `/models`; `search_models` filters by case-insensitive AND-ed tokens.
- **`soca/config/llm_settings.py`** — `LlmSettings` (frozen dataclass:
  backend / provider / model / generation params) with schema-validated JSON
  persistence at `~/.config/soca/llm.json` (mode `0600`, **no keys in the file**).
- **`soca/config/secret_store.py`** — key storage & lookup (§4).
- **`soca/llm/factory.py`** — `build_llm_engine(settings, secrets)` routes
  local ↔ remote; raises `RemoteLLMError(category="auth")` when a remote key is
  missing. Wired into `build_text_runtime`, so the persisted backend is honoured
  every run.
- **`soca/app/engine.py`** — NDJSON commands (`llm_providers`, `llm_models`,
  `llm_set_key`, `llm_select`, `llm_config`); keys are masked in every event.
- **`ui/src/components/SettingsScreen.tsx`** — the Ink settings screen (§6).

## 4. Key security & lookup precedence

API keys are secrets and are handled as such:

- **Writes go only to the OS keyring** (service `soca-llm`, username = provider key).
  On a machine without a keyring backend, `set_key` falls back to
  `~/.config/soca/keys.json` at mode `0600`. **Keys are never auto-written to `.env`.**
- **Reads follow a fixed precedence**: keyring → process env → optional `.env`
  (read-only) → `keys.json`. To prevent the working directory from silently
  selecting a billing account, `.env` is read only when `SOCA_READ_DOTENV=1` or
  a caller explicitly supplies `dotenv_path`; the UI-saved keyring value always wins.
- Keys are **masked** as `sk-…abcd` for display, never logged, and never echoed raw
  into the NDJSON protocol.
- `.env` is git-ignored.

Concretely: the UI's "save key" validates the key with a live `/models` call first,
then stores it. Nothing is persisted if validation fails.

## 5. Pricing transparency — no fabricated numbers

Only **OpenRouter** returns per-token pricing from its `/models` endpoint
(`pricing.{prompt,completion}`), which the catalog converts to USD per 1M tokens and
labels `live`. OpenAI, Groq, and Gemini do **not** expose pricing via API. Rather than
ship a static table that goes stale (and risks quoting a wrong price), the shipped
pricing table is **empty**: those models are labelled `unknown` ("giá: không rõ")
instead of guessing. A user who wants static prices can add a verified row themselves.
OpenRouter also proxies OpenAI/Gemini/Groq models _with_ live prices, so it doubles as
the transparent-pricing path.

## 6. Settings UI (Ink)

Launch shows the main splash; choosing **chat/voice** routes through Settings first so
the user picks the LLM for the session (Esc skips, keeping the saved config). The
screen is a master–detail:

- **Provider row** — `←/→` moves the highlight (no side effects); `Enter` opens it.
- **Local** — one keystroke, no data leaves the machine.
- **Remote, no key** — a masked key field; save validates then stores to the keyring.
  A saved provider hides the field entirely (press `r` to replace an expired key).
- **Model** — the catalog loads on demand and **filters in real time** as you type;
  each row shows context length, price (live / unknown), and source. Picking a model
  saves the selection and returns to chat.

The active backend is shown live in the footer (`remote openrouter:<model>` in amber,
or `local <model>` in green). A persistent warning states that remote sends the
transcript to a third party and that local is the default.

## 7. How to configure

```bash
# 1) install the optional client + secure key storage
uv sync --extra llm-remote        # openai, keyring, tiktoken

# 2) run the UI, open Settings, pick a provider, paste the key, choose a model
uv run soca ui                    # splash → chat → Settings, or /settings any time
```

The chosen backend persists in `~/.config/soca/llm.json` and is used by `soca chat`
and the UI on the next run. To go back to local, pick **Local** in Settings.

## 8. Status & next

- P1–P5 implemented; P6 = this note + README.
- Verified: local chat and remote (`openrouter:gemini-2.5-flash-lite`) both answer
  end-to-end through `soca engine`.
- Not yet done: remote in the **voice** hot path (network RTT hurts time-to-first-audio;
  remote is text-first for now), and a first-party pricing table (deliberately omitted).
