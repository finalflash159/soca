# 05 — Assistant Runtime & Guardrails

`AssistantRuntime` (`core/runtime.py`) is the **brain of one text turn**. The same
runtime serves voice through `VoicePipeline`, text through `soca ask/chat`, and
the TUI chat mode. It does **not** contain ASR or TTS. Its scope is:
guardrails → tools → knowledge/memory → LLM → output.

![Controlled turn](assets/diagrams/controlled-turn.svg)

Editable diagram source: [Lucid controlled turn](https://lucid.app/lucidchart/2a0f7744-4b8d-4b2d-89ec-af974a3f0fa0/view).

## Routing One Turn

```mermaid
flowchart TD
    IN([text + source + metadata]) --> G1[check_input_text<br/>INPUT guardrail]
    G1 -->|block| B[(RuntimeRoute.BLOCKED)]
    G1 -->|allow| TR[tool_router.select]
    TR -->|ToolCall exists| TT[_run_tool_turn]
    TR -->|retrieval_request| RET[select knowledge / memory / both]
    TR -->|smalltalk| MEM[_build_memory_context]
    TR -->|out_of_scope / unresolved| B
    RET --> MEM
    MEM --> KN[_build_knowledge_context]
    KN --> LLM[_run_llm_turn]

    TT --> RT{knowledge tool?}
    RT -->|yes| KC[tool result → KnowledgeContext]
    KC -->|LLM enabled| KL[(KNOWLEDGE_LLM)]
    KC -->|no LLM| KD[(KNOWLEDGE_DIRECT)]
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
| `KNOWLEDGE_DIRECT` | Knowledge tool result is returned because LLM is disabled             | No         |
| `KNOWLEDGE_LLM`    | LLM answers with knowledge context and citations                | Yes        |
| `MEMORY_LLM`       | LLM answers with selected archive-memory `[M#]` context          | Yes        |
| `OUT_OF_SCOPE`     | Capability is not implemented; no tool/answer LLM is called      | No         |
| `CLARIFICATION`    | Router cannot safely identify a local corpus                      | No         |
| `FREE_CHAT`        | Normal chat answer without knowledge                            | Yes        |

## Tool Routing: Deterministic First

`RuntimeToolRouter` (Protocol) and `DefaultRuntimeToolRouter` decide **before the
LLM call** whether the user text matches an explicit knowledge read/search command.
Natural-language capability selection belongs to the semantic/LLM router rather
than keyword rules. Knowledge tools are retrieval steps, not answer generators:
their result is converted to `KnowledgeContext` and passed to the LLM when one is
enabled. If no LLM is available, the raw tool result is returned. Tools also have
side-effect levels and parameter validation.

### Bounded evidence completion

`knowledge.read` returns a bounded receipt with the total line count and a
continuation cursor when the requested note is larger than one tool receipt. The
controlled runtime may follow that cursor so an exact read can cover a long note
without silently treating the first page as the whole document. The current
default permits the initial read plus up to six continuation reads. The shared
turn budget therefore defaults to seven planned/tool actions and 40 controller
transitions; other tool calls consume the same action budget.

These are finite admission limits, not an automatic retrieval fallback. When the
evidence-completion budget is exhausted, the trace records
`budget_exhausted`/`insufficient` and the workflow exposes that terminal state to
the caller. It must not claim that the goal was verified or silently switch to a
different retrieval path. The limits are defined by `RuntimeOptions` and
`TurnBudget` in `soca/core/runtime.py` and `soca/core/workflow/contracts.py`.

### Sufficient-context assessment

Knowledge retrieval and answer generation are separated by the controlled
workflow's `assess_context` node. `SufficientContextAutorater` makes one bounded
structured-output call over the goal and selected evidence. Its control record is
only a typed verdict, confidence, stable reason code, evidence IDs, model/prompt
identity, usage, latency, and provider trace; free-form reasoning is neither
requested nor stored.

An insufficient verdict terminates as `insufficient_evidence` before synthesis.
An unavailable provider, invalid schema, or missing configured assessor is a typed
`sufficiency_assessment_failed` system failure. No path silently reuses relevance
as sufficiency. Production builders keep this feature off until the pinned
Vietnamese quality gate passes; explicit evaluation can set
`sufficient_context_enabled=True`, and an enabled-but-unavailable assessor fails
closed.

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

Controlled free-chat LLM routes stream token-by-token. Retrieval-grounded turns
hold factual output until synthesis, citation validation, and terminal verification
finish, then emit verified chunks. Non-LLM tool, knowledge-direct, and blocked
routes produce fixed text through
`_emit_fixed_result`, chunked into sentences. This keeps pipeline handling uniform
across routes.

`first_sentence_min_chars` lets the first sentence flush earlier than later
sentences so audio reaches the speaker faster.

## Knowledge & Memory in the Prompt

```mermaid
flowchart LR
    F[TurnFrame] --> M[_build_memory_context<br/>bounded working/core]
    F --> R[semantic retrieval source decision]
    R --> K[_build_knowledge_context<br/>selected markdown vault]
    R --> A[_build_archive_memory_context<br/>selected memory archive]
    M & K --> P[_build_llm_prompt]
    P --> LLM[llama.cpp generate / stream]
    LLM --> CL[output cleaning]
    CL --> OUT[response + citations]
```

- **Working memory**: complete user→delivered-assistant turns, bounded by the
  configured working policy. The TUI shares it across voice ↔ chat. It is RAM-only by
  default; the versioned private checkpoint store is opt-in wiring.
- **Archive memory**: never retrieved implicitly. It enters a prompt only after
  the semantic source decision selects `memory`; its citations use `[M#]`.
- **Knowledge**: Markdown vault. The capability router receives a bounded
  manifest as navigation metadata. Content-answer prompts receive only selected
  retrieved passages; the whole-vault manifest is deliberately excluded so a
  title, heading, or path cannot masquerade as note-body evidence.
  `knowledge.inspect` supplies bounded inventory/relationship metadata for
  explicit navigation questions, while `knowledge.search` and `knowledge.read`
  supply answer evidence. The LLM must preserve internal citation labels and
  state that the vault lacks enough information when evidence is empty or
  insufficient. The presentation layer removes internal labels after validation
  and renders structured sources at the end; no answer text is assembled from
  snippets in code.

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
