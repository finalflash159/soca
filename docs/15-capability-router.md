# Capability routing decision

This document records the current production routing contract and the evidence
used to replace the superseded semantic-router path. It is intentionally about
capability selection only; retrieval relevance, evidence reconciliation, and
answer groundedness remain separate runtime gates.

## Production contract

Chat and voice use the same cascade:

1. deterministic handling consumes explicit commands, safe read paths, and
   executable local-time requests;
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
- 66 rows, 22 semantic families;
- 21 train rows, 21 validation rows, 24 test rows;
- family-level split invariant: no family crosses splits;
- SHA-256: `4249290a397a303ec2ec2c9b76ddc2d9a408bfb0607477a14602b71df7f61f58`.

Production semantic examples load only rows with `split=train` or
`split=validation`. The sealed test rows are never loaded as runtime examples.
The router benchmark uses FastEmbed `intfloat/multilingual-e5-small` for the
capability classifier. This is distinct from the production knowledge dense
backend (`AITeamVN/Vietnamese_Embedding_v2`).

The selected router defaults are:

| Setting | Value | Reason |
| --- | ---: | --- |
| semantic threshold | `0.58` | stable across the tested threshold sweep; keeps the existing calibrated operating point |
| semantic margin | `0.00` | held-out test was materially better than `0.02` and `0.04`; unresolved cases are delegated to the bounded LLM tier |
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

The loader uses train/validation as examples and scores all three splits.

| Metric | Train | Validation | Held-out test | Overall |
| --- | ---: | ---: | ---: | ---: |
| disposition accuracy | 100.00% | 100.00% | 79.17% | 92.42% |
| exact source-set accuracy | 100.00% | 100.00% | 83.33% | 93.94% |
| direct-tool exact | — | — | — | 9/9 (100.00%) |
| unsupported → executable tool | 0/3 | 0/6 | 0/6 | 0/15 (0.00%) |
| semantic latency p95 | — | — | — | 2.35 ms |

The held-out confusion is:

```text
direct_tool→direct_tool 9
retrieval_request→retrieval_request 26
retrieval_request→out_of_scope 1
smalltalk→smalltalk 9
out_of_scope→out_of_scope 13
out_of_scope→smalltalk 2
unresolved→unresolved 4
unresolved→retrieval_request 1
unresolved→smalltalk 1
```

The threshold sweep over `0.45, 0.50, 0.55, 0.58, 0.62, 0.66` produced the same
held-out metrics at margin `0.00`. Increasing the margin to `0.02` reduced
test disposition accuracy to `45.83%`; `0.04` reduced it to `41.67%`. The
margin is therefore not retained as a conservative ambiguity gate in the
production default; unresolved semantics are handled by the next explicit
tier or clarification policy.

## Real provider evidence

All raw provider reports and per-turn logs were written outside the repository
under `/tmp/soca-router-remote-20260730`. No API key or remote log is a release
artifact.

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

For the NL capture, time routing was 100% accurate, while knowledge recall was
6.67% and memory recall 16.67%. This is why the LLM tier is not treated as an
oracle and why the semantic corpus remains the primary production classifier;
the remote LLM tier is a bounded rescue/clarification path, not a reason to
remove the held-out gate.

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
