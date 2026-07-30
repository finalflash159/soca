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

## Non-goals

This contract does not claim that the semantic router always selects a
knowledge source, that retrieval is faithful, or that a provider catalog is
available. Those conditions have explicit later gates. An unknown context
window remains observable; the system does not fabricate a finite budget.
