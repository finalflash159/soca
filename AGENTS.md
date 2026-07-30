# SoCa engineering rules

These rules are mandatory for every agent working in this repository.

## Read the plan before implementation

- Before starting any phase or large feature, reread the **entire active plan
  from top to bottom**, including its goals, architectural decisions,
  invariants, dependencies, data and evaluation strategy, rollout, rollback,
  audit matrix, and plan mutation protocol. Reading only the phase section near
  the end of the file is not sufficient.
- Read the audits, ADRs, benchmarks, notes, and references directly linked by
  the active plan when they are relevant to the phase.
- Before changing code, create a matrix mapping `plan requirement → current
  code → current test/artifact → gap/deviation`. A previous commit is not
  evidence that a phase is complete.
- When code differs from the plan, classify the difference as a bug, omission,
  or intentional plan deviation. Do not guess or expand scope silently.

## Research and decisions

- Research before coding whenever information is missing, uncertain, possibly
  outdated, or requires choosing a model, dependency, protocol, or
  architecture.
- Prefer primary sources: official documentation, original papers, upstream
  source code, and official repositories. Other GitHub implementations may be
  considered only after checking license, maintenance, compatibility, and
  trade-offs.
- Benchmark-driven decisions must preserve configuration, model and data
  revision, hardware, raw logs, metrics, failures, and the final decision.
  Smoke, demo, and fake results are not release evidence.
- Do not add domain hardcoding, regex shortcuts, silent fallbacks, or rules
  whose purpose is merely to make tests pass. If the production winner fails,
  raise a typed error and expose truthful readiness.

## Git and review workflow

Every phase or large feature must follow this workflow:

1. `git switch main`
2. `git pull --ff-only`
3. `git switch -c <branch>`
4. Implement the complete phase while keeping history compact. Use **one to
   three cohesive commits per phase or large feature** by default; do not split
   commits by individual file, test, or minor adjustment. Exceed this range
   only when independently reviewable or revertible boundaries genuinely
   require it. Commit prefixes must not include a scope, for example `feat:`,
   `fix:`, `test:`, `docs:`, `refactor:`, or `bench:`.
5. Run the relevant lint, typecheck, unit, integration, and real-flow gates.
6. Push the branch and open a pull request into `main`.
7. Wait for exactly **one complete Qodo review**. A PR summary or “Qodo is busy”
   is not a review; wait for the detailed review body and inline findings.
8. Read every review and inline comment, verify each finding, fix valid
   findings, add regression coverage, commit, and push.
9. Do not wait for a second Qodo review after pushing review fixes. Wait only
   for required CI to pass, then merge the PR.
10. Run `git switch main` and `git pull --ff-only`, and confirm a clean working
    tree before starting the next phase from the updated `main`.

Documentation-only pull requests that change project or agent execution rules
are exempt from the Qodo wait. After a local diff check, merge and close them
without waiting for an automated review.

Do not implement multiple phases in one branch or pull request. Do not modify
production code directly on `main`.

## Tests and real evidence

- Unit and fake tests prove local contracts; they are not full-flow or release
  evidence.
- Every substantial logic change requires an appropriate integration test and
  a real-flow run after unit tests. For LLM behavior, prioritize a capable
  remote model for semantic evaluation while still checking local wiring and
  resource behavior at a proportionate level.
- A real remote test must prove that the provider and model were actually
  called and record the route, tool or action, evidence, terminal outcome,
  usage, latency, and response.
- Retrieval and RAG testing must cover answerable, unanswerable, hard-negative,
  citation, empty-evidence, backend-failure, and no-silent-fallback paths.
- A text-only voice dry run may prove controller parity, but it must not be
  described as a microphone or audio-hardware test.
- Merge only when the phase exit gate has evidence. Anything unproven must be
  recorded as `deferred` or `blocked`, with an owner and reason, rather than
  marked complete.

## Code-change discipline

- Keep changes within scope and preserve unrelated user changes.
- When touching a file with an oversized architecture-style module docstring,
  remove or shorten it within that file's active scope. Do not launch a broad
  cleanup of unrelated files.
- Use typed state and provenance instead of inferring control state from text.
  Do not use chain-of-thought as control data or telemetry.
- Every fallback, retry, timeout, model call, tool call, and terminal outcome
  must be explicit, bounded, and observable.
