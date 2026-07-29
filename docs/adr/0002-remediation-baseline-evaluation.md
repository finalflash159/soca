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

Phase 0 introduces a provenance envelope (`soca-eval-artifact-v1`) and a strict
case loader for remediation suites. Every quality suite must declare one of
`public_screening`, `sanitized_benchmark` or `private_release`. `demo_smoke` and
`unit_fixture` are allowed only for smoke/invariant tests and are rejected by
the quality loader.

The baseline case contract records:

- multi-turn input trajectory;
- expected goal and terminal outcome;
- expected sources, tools and citation paths;
- dataset class, split and provenance;
- commit, file hashes, Python/platform and evaluation config in the manifest.

The initial checked-in suites are independently authored workflow trajectories
and a public XQuAD Vietnamese screening slice. They are not copied from
`knowledge_demo_vault` and are not a production retrieval decision by
themselves.

Static type checking is a required Python quality job. The job runs Pyright,
Ruff and the existing pytest suite on Python 3.11 and 3.12.

## Consequences

Future bake-offs cannot silently mix UI smoke results with quality results. A
new private release set still needs a pinned provenance manifest and reviewed
labels before it can become a release gate. The workflow case set is a
characterization/evaluation contract; it does not claim that the legacy runtime
already passes the new terminal semantics.

## Rollback

The manifest and loader are evaluation-only. Removing the Phase 0 evaluation
files does not alter production runtime behavior; the CI typecheck job can be
disabled temporarily only with an issue/ADR explaining the exception.
