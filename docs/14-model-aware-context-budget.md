# Model-aware context budget

## Decision

SoCa now has one admission contract for a model prompt. `PromptAssembler` owns
the input budget, output reserve, optional-context dropping, required-context
overflow, prompt hash, tokenizer label and capability provenance. The runtime,
structured planner, answer-repair path and `/context` all consume the same
manifest shape.

The contract is:

`model window - safety margin - effective output reserve = input budget`

Required components are never silently dropped. Optional memory, archive and
knowledge components are admitted by priority and are reported as dropped when
they do not fit. A known required overflow raises `PromptBudgetError` before a
provider call.

The default safety margin is 128 tokens. It is deliberately a named policy
constant and is increased only when observed provider usage shows a positive
prompt-token delta. The observed delta is recorded in the prompt manifest.

## Capability provenance

Every manifest records:

- model identifier and context window;
- capability source (`local_registry`, `remote_catalog`, runtime options, or
  active engine metadata);
- tokenizer label and the counter actually used;
- requested and effective output tokens;
- input budget, selected components, dropped components and prompt hash.

Before the first turn, `/context` emits an estimated manifest assembled from
the configured model catalog. After a turn, it emits the exact manifest from
that turn, including provider-observed prompt usage when available. Dynamic
retrieval slots remain visible as `on_demand` components instead of being
counted as if retrieval had already happened.

## Working-memory coupling

Working memory is derived from usable model input when the model context is
known. The 32K production path keeps the selected 16,384 / 15,000 / 12,000
hard, high-watermark and target values. Smaller windows receive smaller
working, summary and recent budgets. The summary worker receives the per-job
artifact and generation budget, so it does not generate a 2,048-token artifact
for a small model that cannot admit it.

The 16K working values are therefore a product policy for the 32K-capable
runtime, not an assumption that every model can accept 16K input.

## Admission order

`PromptAssembler` builds the prompt in a stable order so the same turn can be
explained by the runtime trace and `/context`:

1. required system/runtime instructions and the current user request;
2. required workflow/tool contract and safety instructions;
3. working summary and the protected recent-turn window;
4. selected core memory, when configured;
5. query-dependent knowledge or archive-memory evidence;
6. optional catalog/navigation metadata and other diagnostic context;
7. the effective output reserve and final safety margin.

Required components are admitted or raise `PromptBudgetError`. Optional
components are considered by priority and can be dropped as a whole; the
manifest records every drop and its reason. Query-dependent slots are not
counted as present before retrieval, so `/context` can distinguish an estimate
from a post-turn observed prompt. The assembler never solves overflow by
silently truncating a citation, changing model, or switching retrieval backend.

For example, if a 4K model cannot fit archive memory after required system,
workflow and recent-turn sections, archive memory is reported as dropped and
the turn continues only if the required sections fit. If a required evidence
or instruction block itself cannot fit, the runtime returns a typed budget
failure before making a provider request. This makes the failure diagnosable
and prevents a fluent answer generated from an incomplete contract.

## Evidence

The deterministic test matrix covers context windows 2,048, 4,096, 16,384 and
32,768. It verifies preservation of required input, optional dropping, output
clamping, deterministic prompt hashes, capability provenance and model-scaled
working-memory policy.

A remote blocking run against the private release vault used OpenRouter
`google/gemini-3.5-flash-lite` and observed provider prompt counts 25–30 tokens
above the client estimate on the exercised turns. The 128-token admission
margin covered that delta. The run also exposed pre-existing semantic-router
misses on some unanswerable/instruction-boundary cases; those are routing and
evidence-gate work, not evidence that the context manifest overflowed. Raw
transcripts and run logs remain outside the repository.

### Calibration record

The release calibration used a non-demo private-vault run and the deterministic
matrix of 2,048, 4,096, 16,384 and 32,768-token context windows. Required input
was preserved at every size, optional context was dropped by priority, output
was clamped to model capability, and prompt hashes/provenance remained stable.
The run passed `1078 passed, 3 skipped, 3 warnings`; Ruff, Pyright, UI
typecheck, 49 Vitest tests and the UI build also passed at that revision.

For the remote smoke, OpenRouter `google/gemini-3.5-flash-lite` reported a
positive prompt-token delta of 16 tokens for free chat and 71–73 tokens for the
knowledge route. The 128-token admission margin covered the observed delta.
This calibrates admission safety only; it is not a retrieval-quality result.
If a remote catalog is unavailable, the manifest keeps `context_window=null`
and the runtime does not fabricate a hard limit.

## Non-goals

This contract does not claim that the semantic router always selects a
knowledge source, that retrieval is faithful, or that a provider catalog is
available. Those conditions have explicit later gates. An unknown context
window remains observable; the system does not fabricate a finite budget.
