# ADR 0001: Keep working-memory summary in `trim_only`

- Status: accepted
- Date: 2026-07-28
- Scope: local session compaction for local and remote answer providers

## Decision

Do not select a production local summarizer yet. Keep `trim_only` as the
default and retain `Qwen3-4B-Instruct-2507 Q4_K_M` only as the research
finalist for the next typed-delta experiment.

The application must not auto-download a summary weight, send compaction data
to a remote answer provider, or silently lower the quality gate.

## Evidence

The finalist passed schema, negative-state, injection, cold resource, child
exit, checkpoint, reload, and rendered-context lifecycle checks. It failed the
final SoCa state gates:

- 80.0% annotated fact recall on 200 single-generation cases;
- 72.5% on 40 four-generation rolling sessions;
- 84% decision and 92% correction recall, below the required 95%;
- 8% mixed code/path placement recall.

Public real-data summarization placed it on the Pareto frontier, but public
overlap cannot compensate for lost session state.

Full methodology, dataset provenance, model revisions, resource numbers, and
reproduction commands are in
[`../12-local-summary-model-selection.md`](../12-local-summary-model-selection.md).

## Consequences

- Manual/background summary remains unavailable in normal production sessions
  unless an explicitly provisioned research worker is injected.
- The generic worker, private checkpoint, CAS publication, and evaluator remain
  usable for experiments.
- The next design should emit typed, ID-addressed state deltas and merge them
  deterministically; it must not add keyword/regex content rules.
- The same v2 single and rolling suites remain the release gate.
