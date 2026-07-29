# 09 — Hybrid RAG, Tool Router & Retrieved Memory

This is the **P2 knowledge layer**: how SoCa turns a Markdown vault into grounded
answers. Three cooperating pieces sit behind the `AssistantRuntime` (see
[05 — assistant-runtime](./05-assistant-runtime.md)):

1. **Hybrid retrieval** — sparse (BM25) and dense (ONNX embeddings) fused with
   Reciprocal Rank Fusion, over a transactional v2 vault index.
2. **Capability policy** — deterministic overrides followed by an open-set
   semantic disposition/source decision; retrieval is not an executable tool.
3. **Retrieved memory** — query-aware long-term memory ranked by relevance,
   recency, and importance, with consent-gated episodic capture.

Knowledge and memory reuse the **same retrieval implementation** but stay on
**separate corpora and cache namespaces**, so a `knowledge.search` can never
surface a private memory note.

## One diagram

```mermaid
flowchart TD
    Q[User text] --> R[Tool router cascade]
    R -->|deterministic hit| TC[ToolCall]
    R -->|semantic hit| TC
    R -->|LLM select| TC
    R -->|none + reason| CHAT[Free chat / no matching tool]
    TC --> VAL[validate_arguments + guardrail]
    VAL --> KS[knowledge.search / knowledge.read]
    KS --> HS[HybridKnowledgeSource]
    HS --> SP[sparse_chunk BM25]
    HS --> DE[dense ONNX embeddings]
    SP --> RRF[RRF fusion]
    DE --> RRF
    RRF --> CTX[grounded context + citations]
    Q -.query-aware.-> MEM[RetrievedMemory]
    MEM --> CTX
    CTX --> LLM[LLM answer]
```

## Hybrid retrieval

The retriever stack lives in `soca/knowledge/`:

| Concern              | Module                                   |
| -------------------- | ---------------------------------------- |
| Sparse (BM25) chunks | `retrievers/sparse_chunk.py`             |
| Sparse whole-doc     | `retrievers/sparse_document.py`          |
| Dense embeddings     | `retrievers/dense.py` (fastembed / ONNX) |
| Rank fusion          | `retrievers/rrf.py`                      |
| Fusion source        | `hybrid_source.py`                       |
| Cached vault index   | `cached_source.py`, `index/`             |
| Factory + config     | `factory.py` (`build_retrieval_source`)  |

`build_retrieval_source(vault, include_globs, config=RetrievalConfig(mode=...))`
selects `cached_sparse`, `chunk_sparse`, or `hybrid`; the `RetrievalConfig`
default uses the v2 lifecycle. There is no standalone `dense` mode. Sparse
sync is transactional and dense document embedding is an explicit index-build
operation, never part of `search()`/`retrieve()`. A query loads only a READY
generation whose corpus revision and embedding fingerprint match exactly;
otherwise it serves sparse-only and exposes the stale/missing state. Chunks
are line-anchored, so `KnowledgeCitation` carries `line_start`/`line_end` back
to the UI.

The lifecycle/operations details are in
[11 — index lifecycle](./11-index-lifecycle.md).

**RRF** scores a document as `Σ 1/(k + rank)` across the sparse and dense rank
lists (`k=60`). When the embedding model is unavailable the source degrades to
sparse-only — the pre-P2 behavior — so retrieval never hard-fails.

Historical pre-guard numbers on real XQuAD-Vietnamese (1,193 questions) are in
[BENCHMARKS.md → P2.1](../BENCHMARKS.md). The current guarded model comparison,
including the migrated local vault incident regression, is in
[BENCHMARKS.md → P2.1.1](../BENCHMARKS.md).

### Active local knowledge vault

The product runtime now queries the migrated local vault at
`~/KnowledgeVault/wiki/`. It contains two knowledge areas:

- learning notes: Bayes and ONNX Runtime;
- life/project notes: the TTS decision, hybrid RAG architecture, a clearly
  synthetic food ledger, and health safety boundaries.

The synthetic finance note remains explicitly marked as non-personal data. The
health note is a disclaimer-only guardrail. These notes are in the real local
vault and are not committed to the repository.

### Voice capability policy

Chat and voice construct the same semantic policy from
`eval/prompts/turn_routing_vi.jsonl` and use the same local route embedder.
The old ASR-only `RetrievalIntentGate` and `voice_knowledge_mode` path have
been removed. A transcript is routed once, regardless of whether it came from
text or ASR; retrieval then selects Knowledge/Memory and the runtime performs
the same evidence and answer policy. If the local embedder is unavailable,
both surfaces fail closed to deterministic explicit commands. The semantic
router remains disable-able with the shared `--no-semantic-router` flag for
diagnostics; this P3 wiring does not claim a new calibration-quality result.

## Tool router cascade

The router decides tool use without ever executing a tool itself — the model
only proposes; `ToolRuntime` validates and runs.

```mermaid
flowchart LR
    T[text] --> D[explicit override]
    D -->|direct hit| OUT[allow-listed ToolCall]
    D -->|miss| S[SemanticTurnRouter<br/>disposition + source set]
    S -->|direct_tool| OUT
    S -->|retrieval_request| R[knowledge / memory / both]
    S -->|smalltalk| CHAT[free chat]
    S -->|out_of_scope or unresolved| STOP[redirect / clarify]
```

| Stage         | Module                         | Notes                                                |
| ------------- | ------------------------------ | ---------------------------------------------------- |
| Config/schema | `core/tool_routing.py`         | Frozen config, robust JSON parser, decision schema   |
| Deterministic | `core/runtime.py`              | Explicit prefixes, scoped read paths, no NL guessing |
| Semantic      | `core/semantic_turn_router.py` | disposition + multi-source examples, threshold/margin |
| LLM           | `core/llm_tool_router.py`      | Prompt JSON or JSON-schema, one repair, fallback     |
| Cascade       | `core/router_cascade.py`       | Short-circuits deterministic → semantic → LLM        |
| Construction  | `core/router_setup.py`         | `build_runtime_tool_router(...)`                     |

Invariants worth remembering:

- **`none` is no longer a policy.** `smalltalk`, `out_of_scope`, and
  `unresolved` have distinct dispositions. OOS never falls through to an answer
  model and cannot execute a direct tool.
- **Text and voice use the same semantic contract.** The optional LLM-router
  tier remains independently gated off for voice; semantic capability routing
  itself is not ASR-specific.
- **The executable catalog has four product tools only:** `knowledge.search`,
  `knowledge.read`, `local_time.now`, and `memory.search`. There are no weather,
  device-control, scheduling, or memory-write tools.
- **Safety stays in guardrails.** Prompt injection, path traversal, schema,
  side-effect, and unsupported realtime-claim checks remain guardrail
  responsibilities; they are not capability classifiers.
- **Structured output is an optimization, not a requirement.** Remote uses JSON
  schema with `require_parameters=true`; local `llama.cpp` uses a JSON-schema →
  GBNF grammar. Both fall back to prompted JSON when unavailable.
- Router latency is timed end-to-end into
  `RuntimeTrace.stage_latencies_ms["tool_router"]`.

## Retrieved memory

Long-term memory is a **client of the same retriever**, not a second database.
The stack is in `soca/memory/`:

| Concern              | Module                                      |
| -------------------- | ------------------------------------------- |
| Query-aware retrieval | `retrieved.py` (`RetrievedMemory`)          |
| Search tool          | `tools/memory_tools.py` (`memory.search`)   |
| Ranking signals      | `scoring.py` (relevance/recency/importance) |
| Blob fallback + profile | `longterm.py` (`MarkdownLongTermMemory`)  |
| Profile + episode merge | `composite.py`                            |
| Working-memory summary | `compaction.py`                           |
| Episodic store       | `episodes.py`                               |
| Proposals + approval  | `proposals.py`, `commands.py`              |
| Background reflection | `reflection.py`                            |
| Safe frontmatter parse | `frontmatter.py`                          |
| Setup + wiring       | `core/memory_setup.py`                      |

Archive retrieval is now explicitly requested by the semantic source decision
or `memory:` override; it is not run for ordinary smalltalk/free-chat turns.
`MemoryContextBuilder.build(..., include_archive=False)` contributes only the
always-on bounded working/core context. Archive snippets become `[M#]`
grounded context only when selected. Production setup uses `memory/core.json`
for explicitly approved core items; `memory/profile.md` is not promoted into
core automatically. The compatibility builder still accepts an old profile
source when no core store is supplied by a test/legacy adapter.

`MemoryAccessPlan` records core/working inclusion, archive mode (`none`,
`semantic`, `episodic`, or `both`), archive query, and reason.
`PromptContextAssembler` combines already-selected blocks and never searches a
corpus, so archive access remains visible in the runtime policy.

Working memory is typed as complete `ConversationTurn`s instead of a flat
message deque. The production `working_v2_16k` policy uses
target/high/hard limits of 12,000/15,000/16,384 tokens. At the high watermark,
chat and voice start `Qwen3-4B-Instruct-2507 Q4_K_M` in an isolated local
process, publish its typed artifact with generation CAS, then unload it.
The summary artifact and decoder output budget are both 2,048 tokens; the model
is instructed to shorten the structured JSON before reaching the hard decoder
cap, with one in-process repair pass if the first candidate is invalid.
Summary context is allocated dynamically from 4K to 32K. A missing or invalid
private weight falls back to `trim_only`; there is no extractive/regex summary,
remote summary fallback, or automatic runtime download. `/compact`,
`status`, and `cancel` share the same coordinator in CLI and UI.

Session state is `ram_only` by default. Explicit `local_resumable` mode uses
`SessionCheckpointStore` under the XDG state directory with atomic writes,
private `0700/0600` permissions, schema-versioned wrapper payloads, legacy
working-checkpoint reads, and a revision guard. Checkpoints contain only
working turns/summary state; they do not contain API keys, retrieved snippets,
core values, tool results, or vectors. `clear` deletes the checkpoint.

### Human-in-the-loop capture

Durable memory is never written automatically.

- Episodic summaries persist **only** when `episodic_memory_enabled=true` and the
  user has given explicit consent; raw transcripts are never stored.
- Proposal and approval primitives (`proposals.py`, `commands.py`) are retained
  for an explicitly provisioned future capture workflow; no production chat or
  voice runtime currently creates a proposal or writes durable memory.
- If a pending proposal is provisioned externally, approval/rejection is a
  deterministic application command keyed by its exact ID — never an
  LLM-callable tool.
- Approved notes land under `<vault>/memory/captured/<uuid>.md` with restrictive
  permissions; `private/`, dot-directories, and symlinks are never indexed.

See the memory lifecycle and compaction numbers in
[BENCHMARKS.md → P2.3](../BENCHMARKS.md).

## Corpus isolation

| Corpus    | Include glob     | Cache namespace     |
| --------- | ---------------- | ------------------- |
| Knowledge | `wiki/**/*.md`   | `knowledge/default` |
| Memory    | `memory/**/*.md` | `memory`            |

Wiring happens at exactly two sites — `soca/core/voice_runtime.py` (voice) and
`soca/app/text_runtime.py` (text) — through `knowledge_setup.py`,
`memory_setup.py`, and `router_setup.py`. The runtime, guardrails, and voice
loop are untouched by the knowledge layer.
