# 18 — Engine NDJSON protocol

`soca engine` is the headless boundary every external UI speaks to. The Ink TUI
in [`ui/`](../ui/) is one client; the desktop app planned in
[`zplan/soca_desktop_app_plan.vi.md`](../zplan/soca_desktop_app_plan.vi.md) is a
second. This page is the contract both depend on.

Until this page existed the contract lived implicitly in
[`soca/app/engine.py`](../soca/app/engine.py) and
[`ui/src/store.ts`](../ui/src/store.ts). Two clients against an unwritten
contract is how clients drift apart. The conformance test in
[`tests/test_engine_protocol_contract.py`](../tests/test_engine_protocol_contract.py)
now fails when the engine emits an event this page does not describe.

## 1. Transport

| Property | Value |
| --- | --- |
| Framing | Newline-delimited JSON — one object per line, UTF-8, `ensure_ascii=False` |
| Commands | Client → engine on **stdin** |
| Events | Engine → client on **stdout** |
| Diagnostics | Everything else on **stderr** |
| Writer | `_ProtocolWriter`, mutex-guarded, flushed per line — safe from worker threads |
| Protocol version | `2` (`soca.core.workflow.events.PROTOCOL_VERSION`) |

Stdout is kept pristine: `run_engine` wraps the command loop in
`contextlib.redirect_stdout(sys.stderr)`, so a model loader that prints a banner
cannot corrupt a frame.

A client must tolerate unknown event types and unknown keys inside a known
event. Additive changes are not breaking; removing or retyping a field is.

### Lifecycle

```text
spawn ──▶ hello ──▶ context ──▶ [ commands / events ] ──▶ quit or EOF ──▶ bye ──▶ exit 0
```

`hello` is emitted before any command is read, and is always followed by a
`context` event. If settings failed to load, an `engine_error` with code
`llm_settings_invalid` follows those two.

The loop terminates on `{"cmd":"quit"}` **or** on stdin EOF. Both paths run
`shutdown()`, which joins the chat thread (10 s), voice threads (5 s), the
knowledge-index thread (30 s), and catalog threads before emitting `bye`.
A client should close stdin and wait for `bye` rather than killing the process:
`bye` is the only signal that voice devices and provider clients were released.

Cleanup failures do not block `bye`. They surface as `engine_error` with
`engine_cleanup_failed`, plus a specific code such as
`voice_thread_stop_timeout`.

### Malformed input

| Input | Result |
| --- | --- |
| Blank line | Ignored |
| Invalid JSON | `engine_error` with `line` (first 200 chars); loop continues |
| Valid JSON, not an object | `engine_error` `"command must be a JSON object"` |
| Unknown `cmd` | `engine_error` `"unknown command: ..."` |

None of these terminate the engine.

## 2. Commands

Every command is an object with a `cmd` key. Unlisted keys are ignored.

| `cmd` | Extra fields | Emits |
| --- | --- | --- |
| `status` | — | `status` |
| `context` | — | `context` |
| `memory` | — | `memory` |
| `memory_compact` | `action`: `request` \| `status` \| `cancel` (default `request`) | `memory_compaction`, then `context` and possibly `memory` |
| `memory_proposals` | — | `memory_proposals` |
| `memory_approve` | `proposal_id` (string) | `memory_action` |
| `memory_reject` | `proposal_id` (string) | `memory_action` |
| `usage` | — | `usage` |
| `llm_providers` | — | `llm_providers` |
| `llm_models` | `provider`, `query` (string) | `llm_catalog` |
| `llm_set_key` | `provider`, `key` | `llm_key_status` |
| `llm_select` | provider/model selection | `llm_config`, `context` |
| `llm_config` | — | `llm_config`, `context` |
| `chat` | `text` (string) | `chat` stream + traces (§4) |
| `voice_start` | `max_turns` (int, optional) | `voice` stream (§5) |
| `voice_stop` | — | `voice` `loop_stopped` |
| `voice_profile_select` | profile selection | `status` |
| `knowledge_init` | — | `knowledge_setup`, `status` |
| `knowledge_index` | — | `knowledge_setup` progress, `status` |
| `quit` | — | `bye`, then exit |

`chat` with empty/whitespace text is rejected with `engine_error`
`"chat text is empty"` — no turn starts.

## 3. Event envelope

Two envelope shapes exist and a client must handle both.

**Flat events** carry `event` plus payload keys at the top level:

```json
{"event": "usage", "turns": 3, "llm_turns": 2, "prompt_tokens": 1840}
```

**Workflow events** use the versioned envelope from
`soca.core.workflow.protocol.workflow_event_to_protocol`, where `event` is a
`EventType` value and the body sits under `payload`:

```json
{
  "event": "step_progress",
  "protocol_version": 2,
  "session_id": "…", "run_id": "…", "goal_id": "…",
  "sequence": 7,
  "surface": "chat",
  "timestamp": "2026-08-15T09:12:44.318Z",
  "node": "synthesize",
  "status": "active",
  "payload": {"operation": "retrieval"}
}
```

Discriminate on the presence of `protocol_version`, not on the event name.

## 4. Flat events

### `hello`

First frame. `stack` describes the configured components — `{"llm": …}` for a
text-only engine, or `{"asr", "llm", "tts", "voice"}` when voice is configured.

```json
{"event":"hello","version":2,"protocol_version":2,"supported_versions":[2],
 "profile":"qwen-release","no_model":false,"stack":{"llm":"openai:gpt-5.6-luna"}}
```

A client must reject an engine whose `protocol_version` is not in its own
supported set, rather than proceeding on best effort.

### `bye`

`{"event":"bye"}` — no fields. Last frame before exit 0.

### `engine_error`

`message` is always present. Optional `code` classifies it; optional `detail`
carries the exception type name (never the message, never a stack trace).

Known codes: `llm_settings_invalid`, `runtime_cleanup_failed`,
`chat_thread_stop_timeout`, `voice_thread_stop_timeout`,
`knowledge_index_stop_timeout`, `engine_cleanup_failed`.

`engine_error` is **not** a terminal state for a turn. A blocked or failed turn
reports through `chat` / `turn_terminal`; `engine_error` is an engine-level
complaint.

### `status`

```json
{"event":"status","profiles":[…],"knowledge_vault":…,"knowledge_index":…,
 "runtime_components":[{"name":…,"status":…}]}
```

`runtime_components` describes configured dependencies **without loading them** —
readiness is inspected, not proven by instantiation. `knowledge_index` is `null`
when no index exists (see [11 — index lifecycle](11-index-lifecycle.md)).

### `context`

Prompt-budget manifest. Two variants share the discriminator `ready`:

| `ready` | Meaning | Key fields |
| --- | --- | --- |
| `true` | Manifest built | `prompt_hash`, `prompt_manifest`, `resident_prompt_tokens`, `input_budget_tokens`, `available_dynamic_tokens`, `components[]` |
| `false` | `PromptBudgetError` | `context_error`, `context_error_detail`, `components: []` |

`estimated` distinguishes a projection (`true`, built from resident state) from
the manifest of a turn that actually ran (`false`). Do not present an estimated
figure as observed usage — see [14 — model-aware context budget](14-model-aware-context-budget.md).

`observed_prompt_tokens`, `provider_prompt_tokens` and `prompt_token_delta` are
`null` until a provider reports real counts.

### `memory`

`enabled: false` means no session memory is configured; `text`, `summary` and
`recent` are then empty strings and `stats` is `null`. When enabled, `stats` is
the dataclass form of `SessionMemory.stats()`.

### `memory_compaction`

Either `{"status":"disabled","detail":"memory disabled"}` or the dataclass form
of the compaction result. `status` values observed by the engine:
`accepted`, `running`, `idle`, `published`, `trim_only`, `stale`, `failed`,
`unavailable`.

`accepted` is followed by `context`. Any status other than `running`/`idle` is
followed by `memory` **and** `context`.

### `memory_proposals`

At most 64 proposals; `statement` truncated to 400 chars. `createdAt` is ISO-8601.

> Operational note: the proposal inbox is currently always empty in production —
> `MemoryProposal` is constructed only in tests and `eval/eval_memory_lifecycle.py`.
> A UI must render the empty state as the normal case, not as an error.

### `memory_action`

```json
{"event":"memory_action","proposal_id":"…","action":"approved","ok":true,"error_code":null}
```

`action` is `approved`/`rejected` — the past tense reflects the attempt, not the
outcome. Read `ok`. On failure `error_code` is `memory_unavailable`,
`command_failed`, or the store's own rejection status. `proposal_id` is
truncated to 80 chars on the failure paths.

### `usage`

Session totals: `turns`, `llm_turns`, `prompt_tokens`, `completion_tokens`,
`mean_ttft_ms`, `mean_tokens_per_second`.

### `llm_providers`

`providers[]` of `{key, label, has_key, has_pricing_api}`. Never contains key
material.

### `llm_catalog`

`{provider, models[], pricing_as_of}`. Each model exposes exactly:
`id`, `label`, `context_length`, `price_prompt_per_1m`,
`price_completion_per_1m`, `pricing_source`, `max_output_tokens`,
`reasoning_supported`, `reasoning_mandatory`.

`supported_parameters` is deliberately **not** in the protocol — it is an LLM
adapter concern and would push the OpenRouter catalog past a comfortable frame
size.

An empty `models[]` is emitted immediately while a fetch runs in the background;
a second `llm_catalog` follows when the catalog arrives. A client must treat the
first frame as "loading", not as "no models".

### `llm_key_status`

`{provider, ok}` plus optional `pending`, `masked`, `message`. `pending: true`
means validation is in flight. Never carries the key.

### `llm_config`

Active LLM configuration: `backend`, `provider`, `model`, `max_tokens`,
`effective_max_tokens`, `reasoning_enabled`, `effective_reasoning_enabled`,
`reasoning_supported`, `reasoning_mandatory`, `temperature`, `top_p`,
`pricing_as_of`, `pricing`, `context_length`, `runtime_ready`, `settings_error`.

The `effective_*` pair matters: a model may force reasoning on
(`reasoning_mandatory`), so `reasoning_enabled` is the request and
`effective_reasoning_enabled` is what will actually happen. A UI must display
the effective value.

Always followed by `context`.

### `knowledge_setup`

`{action, status, vault, detail}` plus optional `error_code`. `action` is `init`
or `index`; `status` is `ok`, `failed`, or a progress state. This is the only
event stream for vault creation and index builds.

### `chat`

Discriminated by `type`:

| `type` | Fields | Meaning |
| --- | --- | --- |
| `loading` | `text` | Building the text runtime (first turn only) |
| `ready` | `llm_status`, `knowledge_status`, `memory_status` | Runtime built |
| `start` | `text`, `run_id`, `goal_id` | Turn accepted; `text` echoes the input |
| `done` | see below | **Terminal.** Exactly one per successful turn |
| `error` | `text` | **Terminal.** Only when the turn failed before producing a result |

`done` payload:

```json
{"event":"chat","type":"done","text":"…","route":"knowledge",
 "blocked":false,"usage":{…},"citations":[…],
 "provider_trace":{…},"llm_error":{}}
```

`text` has citation labels stripped (`answer_text_without_citation_labels`);
the structured citations live in `citations[]`. A UI renders provenance from
`citations`, never by parsing `[K1]` out of the answer.

**`blocked: true` is not an error and must not be rendered as one.** Per
[ADR 0003](adr/), a blocked result is a legitimate terminal outcome — the system
declining to answer without evidence. See [13 — retrieval evidence gates](13-retrieval-evidence-gates.md).

`error` and `done` are mutually exclusive. If a result was already produced, a
later worker exception is logged, not emitted — worker cleanup is not a product
terminal.

### `router_trace`

Emitted once per turn after `chat:done`, and once per voice `runtime` event.

Core fields: `tier` (`deterministic`\|`semantic`\|`llm`\|`none`), `tool`,
`reason`, `disposition`, `handler`, `selected_routes[]`, `sources[]`, `scores{}`,
`source_scores{}`, `runner_up`, `margin`, `evidence_status`,
`evidence_completion_status`, `evidence_completion_reason`,
`evidence_completion_actions`, `answer_policy`, `answer_policy_reason`,
`grounding_policy_version`, `citation_count`, `memory_access_plan`.

The voice variant fills the same field set from turn metadata; `tool` is always
`null` there, and `tier` is normalized to `none` when it is not one of the four
known values.

### `retrieval_trace`

`{query, tier, latency_ms, columns[], fused, rejected_count, evidence}`.

`columns[]` is per-source hits before fusion; `fused` is the merged ranking.
`evidence` is the evidence-gate decision (`null` when the gate did not run) —
this is what a retrieval inspector should show alongside each snippet.

### `memory_trace`

`{mode, degraded_reason, hits[]}`. `hits[].id` is truncated to 120 chars;
`corpus` is the archive class. `mode` is normalized through
`_memory_protocol_mode`, which folds the raw mode, the degraded reason and the
hit count into one protocol value — do not reconstruct it client-side.

Background compaction state is folded in: `accepted` → `queued`,
`running` → `running`, `failed`/`unavailable` → `failed`, otherwise `idle`.

### `turn_progress`

Coarse per-turn progress, independent of the workflow stream:

```json
{"event":"turn_progress","surface":"chat","phase":"analyzing",
 "operation":"normalize_input","status":"active",
 "run_id":"…","goal_id":"…","sequence":3}
```

`sequence` is monotonic **per turn context**, not global. Optional
`terminal_status` and `detail` appear on the closing frame.

`goal_id` is `pending-<run_id>` until a goal is resolved. A client must not
treat that as a real goal identifier.

## 5. `voice` events

One event type with 20 `type` values, mirroring `VoiceMonitorEvent`:

```json
{"event":"voice","type":"asr_partial","text":"…","latency_ms":312.4,
 "metadata":{…},"usage":{…}}
```

| Group | `type` values |
| --- | --- |
| Session | `loading`, `warmup`, `ready`, `loop_started`, `loop_stopped` |
| Capture | `recording`, `voice_level`, `audio`, `recorded` |
| Recognition | `transcribing`, `asr_partial`, `repair` |
| Turn | `turn_start`, `progress`, `turn_end`, `done` |
| Output | `tts`, `playback_started`, `barge_in` |
| Failure | `error` |

`turn_start` opens a turn-progress context and emits `turn_progress`
`preparing`. `runtime` metadata drives the voice `router_trace` and refreshes
the cached prompt manifest.

`repair` is not an error: rejected speech becomes a Vietnamese repair prompt
rather than an invented transcript (see [04 — ASR robustness](04-asr-robustness.md)).
Render it as a turn, not as a failure.

`voice_level` is high-frequency. A client must throttle or coalesce it; do not
re-render the tree per frame.

## 6. Workflow events

Emitted through `WorkflowEventStream` with the versioned envelope from §3.

`event` (`EventType`):
`turn_started`, `step_started`, `step_progress`, `step_completed`,
`verification_started`, `verification_completed`, `answer_delta`,
`public_update`, `goal_resolved`, `turn_terminal`.

`status` (`EventStatus`): `started`, `active`, `completed`, `failed`, `cancelled`.

`node` (`TurnNode`):
`admit`, `resolve_goal`, `ask_clarification`, `choose_capability`,
`make_plan`, `authorize_action`, `execute_action`, `assess_observation`,
`revise_query`, `synthesize`, `verify_answer`, `repair_answer`, `finalize`.

`turn_terminal` carries a `TerminalStatus`: `achieved`, `needs_clarification`,
`insufficient_evidence`, `safe_failure`, `budget_exhausted`, `cancelled`,
`system_failure`.

**`answer_delta` fires once per chunk of answer text.** Both surfaces strip
citation labels (`[K1]`, `[M1]`) before publishing, so a delta never shows a
marker the final text removes. Provenance arrives as the structured `citations`
list, never as prose.

On the chat surface the chunks are guardrail-passed sentences, **stripped of
their surrounding whitespace**, so a client appends them **with a separator**
rather than concatenating them.

`pop_ready_sentence` (`soca/core/streaming.py`) returns `buffer[:end].strip()`
and hands back `buffer[end:].lstrip()`; the tail is `buffer.strip()` too. The
space between two sentences is therefore discarded on both sides, while
`chat/done.text` — built from the raw token join — keeps it:

```text
chunks : ["Xin chào! Mình là Sơn Ca.", "Hôm nay mình giúp gì được?"]
"".join → "Xin chào! Mình là Sơn Ca.Hôm nay mình giúp gì được?"   ✗ glued
" ".join → "Xin chào! Mình là Sơn Ca. Hôm nay mình giúp gì được?"  ✓
```

A client verifying its reassembly must therefore compare with whitespace
collapsed. What survives that is real: a dropped frame, or a trailing `Nguồn:`
footer that only the whole-answer cleaner removes.

> This is a rough edge, not a designed contract. Making chunk boundaries
> whitespace-preserving is an engine-side change in `pop_ready_sentence`, and it
> would touch every consumer of that splitter — including TTS — so it is
> recorded here rather than taken unilaterally. On the voice surface a delta is a raw model token,
so chunk boundaries fall mid-word and the concatenation is not byte-identical to
the caption; `voice/done.text` is authoritative there.

How many deltas a turn produces, and how far apart they land, depends on the
turn and the model:

- A turn the router resolved to **no capability** (`smalltalk`, `out_of_scope`)
  streams as the model generates, so deltas arrive progressively. This is the
  case a composing animation has time to run.
- A turn carrying a **tool or retrieval** is held by the bounded controller until
  synthesis and verification finish, then emits every chunk at once. No
  unverified answer text is ever published, so there is nothing to animate; drive
  the intervening UI from `turn_progress.phase`, not from delta arrival.

Measured on 2026-08-16 against `openai/gpt-5.6-luna`, one turn each:

| Route | Deltas | Arrival |
| --- | ---: | --- |
| `free_chat` | 2 | +7007 ms, +7068 ms — 61 ms apart |
| `knowledge_llm` | 5 | all at +10190 ms — 0 ms apart |

So "does chat stream?" has two answers, and the split is architectural rather
than incidental: publishing a capability turn progressively would put unverified
answer text on screen, which is what ADR 0003 forbids. A client that wants the
wait to read as progress must render `turn_progress.phase`, because for those
turns there is no partial answer to render.
- Some hosted models return the whole completion in a single SSE chunk. The turn
  is then one delta even on the streaming path. Treat a single delta as normal,
  not as an error.

`sequence` is monotonic per `run_id`. Ordering across `run_id`s is not defined.

## 7. Client obligations

1. **Version-check `hello`** before sending commands.
2. **Close stdin and wait for `bye`**; do not SIGKILL. Only `bye` proves audio
   devices and provider clients were released.
3. **Tolerate unknown events and unknown keys.** Do not validate exhaustively.
4. **Render `blocked: true` as a terminal outcome, not an error.**
5. **Throttle `voice_level`.**
6. **Never reconstruct routing, evidence, or memory-mode logic client-side.**
   The engine owns those; a UI that recomputes them will disagree with the CLI.
7. **Treat the first `llm_catalog` frame as loading**, not as an empty catalog.
8. **Show `effective_*` values**, not the requested ones.

## 8. Conformance test

[`tests/test_engine_protocol_contract.py`](../tests/test_engine_protocol_contract.py)
pins:

- the command set `dispatch` accepts,
- the flat event names the engine can emit,
- the `chat` and `voice` `type` vocabularies,
- the workflow enum values,
- the `llm_catalog` model field set,
- the protocol version.

Adding an event or field requires updating this page in the same change. That is
the point: the test fails on drift, so the two clients cannot silently diverge.

## Related

- [07 — TUI](07-tui.md) — the first client
- [05 — assistant runtime](05-assistant-runtime.md) — what produces these events
- [13 — retrieval evidence gates](13-retrieval-evidence-gates.md) — `blocked` semantics
- [14 — model-aware context budget](14-model-aware-context-budget.md) — `context` fields
