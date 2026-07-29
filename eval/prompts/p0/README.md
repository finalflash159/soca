# Voice/knowledge gate P0 benchmark data

This directory is the frozen P0 evaluation input for the shared chat/voice
capability router. It is deliberately separate from the showcase/demo vault.

## Files

- `turn_routing_vi.jsonl`: disposition labels for the complete cascade.
- `retrieval_source_vi.jsonl`: multi-label source-selection labels.
- `grounding_vi.jsonl`: retrieval/abstention labels over `real_rag_vault`.
- `memory_context_policy_vi.jsonl`: policy annotations for working/core/archive
  context injection; rows contain no personal content.
- `voice_parity_vi.jsonl`: clean transcript and ASR-shaped variants that must
  produce the same disposition/source/tool decision.
- `turn_routing_examples_vi.jsonl`: route examples used by the P0 local
  baseline. These are not copied from the held-out test families.

## Split policy

`split` is assigned at `family` level. All paraphrases in a family stay in one
split. The evaluator rejects duplicate IDs, unknown labels, and family leakage.
The test split is never used to choose a threshold or route examples.

## Provenance

Routing/source/parity rows are SoCa-authored evaluation annotations, not claims
about a user's private vault. Grounding answerable rows are selected from the
checked-in XQuAD Vietnamese fixture and retain its path references. Unanswerable
rows intentionally ask for facts absent from that corpus.
