# ADR 0002: Provenance-first remediation evaluation

## Status

Accepted for Phase 0.

## Context

The existing test suite is useful for contracts and unit behavior, but it is
not a reproducible release gate for the controlled workflow. A quality result
must identify the source revision, exact dataset bytes, runtime environment and
configuration. The remediation work also explicitly excludes the small UI demo
vault from model, router and retrieval decisions.

## Decision

Phase 0 introduces a provenance envelope (`soca-eval-artifact-v1`), an
executable runtime baseline and a strict case loader for remediation suites.
Every quality suite must declare one of
`public_screening`, `sanitized_benchmark` or `private_release`. `demo_smoke` and
`unit_fixture` are allowed only for smoke/invariant tests and are rejected by
the quality loader.

The baseline case contract records:

- multi-turn input trajectory;
- regression or capability suite ownership;
- paraphrase-family identity, with cross-split leakage rejected;
- expected goal and terminal outcome;
- expected sources, tools and citation paths;
- linked machine-readable audit items;
- dataset class, split and provenance;
- commit, file hashes, Python/platform/hardware and evaluation config.

`eval.runtime_remediation_baseline` executes every trajectory through the real
`AssistantRuntime` in blocking or streaming mode. Each turn records route,
tool calls/results, source selection, retrieval evidence, citations, prompt
manifest, answer validation, wall/stage latency, normalized token usage,
response and terminal outcome. Runtime/provider exceptions are terminal
`system_failure` records; the runner never switches model, retriever or
execution path after a failure.

The checked-in suites are independently authored workflow trajectories and a
public XQuAD Vietnamese screening slice. They are not copied from
`knowledge_demo_vault`. The suite is a baseline and regression contract, not a
production retrieval decision by itself.

Static type checking is a required Python quality job. CI runs Pyright, Ruff
and pytest on Python 3.11 and 3.12.

SoCa does not adopt an open-ended generic agent loop. The later workflow
replacement must be a typed, bounded controller with explicit state,
authorization, action fingerprints, shared budgets, verification and exactly
one terminal outcome. Public acknowledgement is an update, never proof that a
goal is complete. This architectural choice is evaluated by outcome
trajectories here and expanded in ADR 0003.

## Consequences

Future bake-offs cannot silently mix UI smoke results with quality results.
Regression and capability rates remain separate so existing behavior cannot
hide a missing capability. A private release set still needs a pinned
provenance manifest and reviewed labels before it can become a release gate.
The baseline intentionally records current failures; it does not relabel them
as passes.

## Rollback

The harness is evaluation-only and does not change production behavior. A
rollback is an explicit revert of the evaluation commits; CI type checking is
not silently disabled.
