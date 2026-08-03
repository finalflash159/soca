# 05 — Assistant Runtime & Guardrails

`AssistantRuntime` (`core/runtime.py`) is the **brain of one text turn**. The same
runtime serves voice through `VoicePipeline`, text through `soca ask/chat`, and
the TUI chat mode. It does **not** contain ASR or TTS. Its scope is:
guardrails → tools → knowledge/memory → LLM → output.

## Routing One Turn

```mermaid
flowchart TD
    IN([text + source + metadata]) --> G1[check_input_text<br/>INPUT guardrail]
    G1 -->|block| B[(RuntimeRoute.BLOCKED)]
    G1 -->|allow| TR[tool_router.select]
    TR -->|ToolCall exists| TT[_run_tool_turn]
    TR -->|none| MEM[_build_memory_context]
    MEM --> KN[_build_knowledge_context]
    KN --> LLM[_run_llm_turn]

    TT --> RT{tool returns directly?}
    RT -->|knowledge read| KD[(KNOWLEDGE_DIRECT)]
    RT -->|other tool| TD[(TOOL_DIRECT)]

    LLM --> RU{used knowledge?}
    RU -->|yes| KL[(KNOWLEDGE_LLM)]
    RU -->|no| FC[(FREE_CHAT)]
```

## Routes (`RuntimeRoute`)

| Route              | When It Happens                                                 | Calls LLM? |
| ------------------ | --------------------------------------------------------------- | ---------- |
| `BLOCKED`          | A guardrail blocks at any stage                                 | No         |
| `TOOL_DIRECT`      | A non-knowledge tool can answer directly                        | No         |
| `KNOWLEDGE_DIRECT` | A knowledge file is read directly                               | No         |
| `KNOWLEDGE_LLM`    | LLM answers with knowledge context and citations                | Yes        |
| `FREE_CHAT`        | Normal chat answer without knowledge                            | Yes        |

> `LLM_FALLBACK` is the old alias of `FREE_CHAT`, kept for compatibility with
> tests and reports.

## Tool Routing: Deterministic First

`RuntimeToolRouter` (Protocol) and `DefaultRuntimeToolRouter` decide **before the
LLM call** whether the user text matches an explicit read/search command.
Natural-language capability selection belongs to the semantic router.
Knowledge-tool output is grounded context for the answer model rather than
prewritten answer text. Tools also have side-effect levels and parameter
validation.

## Guardrails: Multiple Stages

`core/guardrails.py`. **Stage** means where the check runs; **Action** is the
result.

```mermaid
flowchart LR
    subgraph Stages["GuardrailStage"]
        I[INPUT] --> R[RETRIEVAL] --> TI[TOOL_INPUT] --> TO[TOOL_OUTPUT] --> O[OUTPUT]
    end
    Stages -.-> A["GuardrailAction:<br/>ALLOW · WARN · BLOCK"]
```

| Function                    | Stage       | Checks                                                                |
| --------------------------- | ----------- | --------------------------------------------------------------------- |
| `check_input_text`          | INPUT       | Whether the user input violates policy                                |
| `check_knowledge_read_path` | RETRIEVAL   | Whether a knowledge path is safe and cannot path-traverse             |
| `check_untrusted_text`      | RETRIEVAL   | Whether retrieved untrusted content contains dangerous instructions   |
| `check_tool_call`           | TOOL_INPUT  | Whether tool parameters and permissions are valid                     |
| `check_tool_result`         | TOOL_OUTPUT | Whether tool output leaks private or unsafe content                   |
| `check_final_output`        | OUTPUT      | Whether the final answer makes unsupported claims, e.g. realtime data |

`GuardrailEvent` is frozen and records `stage`, `action`, `reason`, and
`message`. All events are stored in `RuntimeTrace.guardrail_events` for the
Inspector.

### Why Streaming Remains Safe

On LLM routes, `check_final_output` is a **stateless substring scan** that catches
unsupported realtime claims. Checking by **sentence** is therefore equivalent to
checking the full text, so the runtime can guard each `sentence` and send it to
TTS immediately without losing safety. This is the key to per-sentence streaming.
See [03](./03-voice-pipeline.md).

## Runtime Streaming (`stream_text_turn`)

The runtime emits `RuntimeStreamEvent`s:

```text
token*  → sentence*  → result
```

- `token`: raw token for live UI display.
- `sentence`: guardrail-checked chunk ready for TTS.
- `result`: full `RuntimeResult` with route, trace, citations, and usage.

Only **LLM routes** stream token-by-token. Tool, knowledge-direct, and blocked
routes produce fixed text through `_emit_fixed_result`, chunked into sentences.
This keeps pipeline handling uniform across routes.

`first_sentence_min_chars` lets the first sentence flush earlier than later
sentences so audio reaches the speaker faster.

## Knowledge & Memory in the Prompt

```mermaid
flowchart LR
    F[TurnFrame] --> M[_build_memory_context<br/>approved core + retrieved archive + RAM session]
    F --> K[_build_knowledge_context<br/>search markdown vault]
    M & K --> P[_build_llm_prompt]
    P --> LLM[llama.cpp generate / stream]
    LLM --> CL[output cleaning]
    CL --> OUT[response + citations]
```

- **Always-on memory**: approved items in `memory/core.json`.
- **Archive memory**: Markdown notes under `memory/`, retrieved only when the
  runtime selects an archive access plan.
- **Session memory**: RAM-only, multi-turn. The TUI uses **shared session
  memory** so voice ↔ chat keep the same context. See [07](./07-tui.md).
- **Knowledge**: Markdown vault. When used, the answer includes
  `KnowledgeCitation` values such as `[K1] path`.

## <a id="usage-telemetry"></a>Usage Telemetry

`core/usage.py` defines frozen dataclasses and does not depend on Rich:

```mermaid
classDiagram
    class LLMUsage {
        +int prompt_tokens
        +int completion_tokens
        +float ttft_ms
        +float total_latency_ms
        +float tokens_per_second
    }
    class TurnUsage {
        +str route
        +bool blocked
        +LLMUsage llm
        +dict stage_latencies_ms
        +float ttfa_ms
        +int tts_chunks
    }
    class SessionUsage {
        +int total_turns
        +int llm_turns
        +int total_prompt_tokens
        +float mean_ttft_ms
        +float mean_tokens_per_second
        +add(TurnUsage) SessionUsage
    }
    TurnUsage --> LLMUsage
    SessionUsage --> TurnUsage
```

- `ttft_ms` means time-to-first-token for LLM.
- `ttfa_ms` means time-to-first-audio for voice.
- `LLMUsage.from_llm_result`, `TurnUsage.from_runtime_result`, and
  `TurnUsage.from_voice` are duck-typed builders.
- `SessionUsage.add` returns a **new copy** (immutable aggregation). You can view
  this through `soca ... --usage` or `/usage` in the TUI.
