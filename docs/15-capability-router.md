# Capability routing decision

This document records the current production routing contract and the evidence
used to replace the superseded semantic-router path. It is intentionally about
capability selection only; retrieval relevance, evidence reconciliation, and
answer groundedness remain separate runtime gates.

The structured provenance index for every result below is
[`docs/evidence/capability-router-20260730.json`](evidence/capability-router-20260730.json).
It records run IDs, explicit run types, source Git state, corpus/model
revisions, hardware, metrics and external raw-artifact paths. The remote
captures are characterization evidence only because the provider does not
expose an immutable model revision and those runs were made from a dirty
working tree; they are not release gates.

## Production contract

Chat and voice use the same cascade:

1. deterministic handling consumes explicit commands, safe read paths, and
   allow-listed local knowledge/memory commands;
2. the shared semantic turn router classifies a normalized utterance into a
   disposition and, for retrieval, a source set;
3. the bounded LLM router runs only when the semantic tier is unresolved or
   ambiguous and a configured LLM is available.

The LLM router cannot execute retrieval itself. It returns a validated route
contract, and `AssistantRuntime` remains responsible for source retrieval,
evidence status, prompting, citations, and abstention.

The LLM router has no production fallback to deterministic selection. Provider
failure, empty catalog, invalid JSON, unknown handler, disabled handler, or
invalid arguments produce an observable `unresolved` decision with a typed
reason. Explicit `mode="llm"` configuration without an available engine raises
`llm_tool_router_unavailable` during construction instead of silently changing
the routing mode.

The old `semantic_tool_router` implementation, evaluator, and tests were
removed from production. The remaining evaluator uses
`semantic_turn_router`, so benchmark code does not exercise a dead path.

## Frozen inputs and calibration

The held-out routing corpus is:

- `eval/prompts/p0/turn_routing_vi.jsonl`
- 94 rows, with family-level train/validation/test splits;
- 38 train rows, 24 validation rows, 32 test rows;
- family-level split invariant: no family crosses splits;
- SHA-256 for the current working corpus: `3b1f1e4611b789de07136da75da182aab7b639b7ca62f22635618e64972c2579`.

Production semantic examples load only rows with `split=train` or
`split=validation`. The sealed test rows are never loaded as runtime examples.
The router benchmark uses FastEmbed `intfloat/multilingual-e5-small` for the
capability classifier. This is distinct from the production knowledge dense
backend (`AITeamVN/Vietnamese_Embedding_v2`). The benchmark snapshot is
`614241f622f53c4eeff9890bdc4f31cfecc418b3`; the production knowledge model is
pinned to `18b44161e041bf1d3a333ab5144b5b7b93f914d2`.

The selected router defaults are:

| Setting | Value | Reason |
| --- | ---: | --- |
| semantic threshold | `0.58` | stable across the tested threshold sweep; keeps the existing calibrated operating point |
| semantic margin | `0.00` | held-out test was materially better than `0.02` and `0.04`; unresolved cases are delegated to the bounded LLM tier |
| direct-tool score floor | `0.85` | calibrated from train/validation direct-tool examples; low-confidence actions are delegated instead of executed |
| direct-vs-retrieval margin | `0.01` | prevents an action from winning when content retrieval is nearly as plausible |
| LLM repair attempts | `1` | one bounded schema repair, then fail closed |
| voice semantic routing | enabled | chat and ASR transcript share the same capability policy |

## Offline held-out benchmark

Command:

```bash
uv run python eval/eval_full_cascade.py \
  --dataset eval/prompts/p0/turn_routing_vi.jsonl \
  --examples eval/prompts/p0/turn_routing_vi.jsonl \
  --run-local --threshold 0.58 --margin 0.0 \
  --predictions /tmp/soca-router-local/predictions.jsonl \
  --output /tmp/soca-router-local/report.json
```

Evidence record: run ID `capability-local-heldout-20260730` in the structured
provenance index. Its report and predictions are external artifacts under
`/tmp/soca-router-local/`, not committed logs.

The loader uses train/validation as examples and scores all three splits. The
semantic-only run below is a calibration characterization, not a complete
remote cascade gate: uncertain direct actions are intentionally delegated to
the bounded LLM router.

| Metric | Train | Validation | Held-out test | Overall |
| --- | ---: | ---: | ---: | ---: |
| disposition accuracy | 100.00% | 100.00% | 71.88% | 89.36% |
| exact source-set accuracy | 100.00% | 100.00% | 84.38% | 94.68% |
| direct-tool exact | — | — | 11/13 (84.62%) | 11/13 (84.62%) |
| unsupported → executable tool | 0/8 | 0/9 | 0/10 | 0/27 (0.00%) |
| semantic latency p95 | — | — | — | 2.51 ms |

The held-out confusion is:

```text
direct_tool→direct_tool 11
direct_tool→unresolved 2 (delegated to LLM)
retrieval_request→retrieval_request 37
retrieval_request→out_of_scope 1
retrieval_request→unresolved 1
smalltalk→smalltalk 9
out_of_scope→out_of_scope 24
out_of_scope→smalltalk 2
out_of_scope→unresolved 1
unresolved→unresolved 4
unresolved→retrieval_request 1
unresolved→smalltalk 1
```

The semantic disposition threshold remains `0.58` with a global margin of
`0.00`. Direct actions have an additional calibrated score floor of `0.85`
and a `0.01` direct-vs-retrieval margin. This removed all unsupported direct
tool calls in the current test (`0/10`) while delegating two ambiguous catalog
paraphrases to the LLM tier. The two delegated cases must be evaluated in the
remote cascade; they are not counted as semantic-only successes.

## Real provider evidence

All raw provider reports and per-turn logs were written outside the repository
under `/tmp/soca-router-remote-20260730`. No API key or remote log is a release
artifact. The blocking and streaming records are respectively identified as
`runtime-openrouter-blocking-20260730` and
`runtime-openrouter-streaming-20260730` in the structured provenance index.

### Runtime blocking and streaming

Provider: OpenRouter, model `google/gemini-3.5-flash-lite`, hybrid retrieval,
the repository's sanitized/private-release quality suite, 14 cases and 18
turns. The showcase/demo vault was not used.

| Run | Cases passed | Provider errors | Route/terminal parity |
| --- | ---: | ---: | ---: |
| blocking | 1/14 | 0 | reference |
| streaming | 1/14 | 0 | 18/18 turns identical |

The low goal score is recorded as a real failure, not hidden: the corpus and
answer/citation goals still fail on several cases. The router itself selected
retrieval for 12 turns, rejected unsupported capabilities safely, and did not
produce an unsupported tool call. This is not evidence that RAG answer quality
is complete.

### Standalone remote route contract

The live evaluator sent the route prompt to OpenRouter and never executed a
tool. It used the tracked 100-row router corpus and two deliberately bounded
captures:

| Slice | Scored rows | Precision | Recall | F1 | False trigger |
| --- | ---: | ---: | ---: | ---: | ---: |
| first 25 mixed rows | 25 | 95.00% | 86.36% | 90.48% | 1/3 no-tool rows |
| 51 NL rows (time/knowledge/memory/smalltalk) | 51 | 88.24% | 38.46% | 53.57% | 0/12 no-tool rows |

The current focused live route checks with the configured OpenRouter model
confirmed `knowledge.inspect` for inventory and relationship requests,
`knowledge,memory` retrieval for a mixed note/memory request, and
`out_of_scope` for a general vault-definition question. These are
characterization runs from the current dirty working tree; they are not yet a
release gate.

## Remaining gaps

This decision closes router-consolidation scope, not the whole remediation
program. Remaining work includes:

- multi-turn goal/source correction after an acknowledgement;
- calibrated retrieval/evidence and answer-groundedness gates;
- memory/session lifecycle and compaction separation;
- chat/voice provider parity and provenance telemetry;
- provider reliability, UI status, audio/release gates.

These gaps remain explicit in the remediation inventory and must not be marked
complete from the route-only metrics above.
