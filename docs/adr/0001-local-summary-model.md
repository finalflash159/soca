# ADR 0001: Select Qwen3-4B-Instruct-2507 for working-memory summary

- Status: accepted
- Date: 2026-07-28
- Scope: local session compaction for local and remote answer providers

## Decision

Select `Qwen3-4B-Instruct-2507 Q4_K_M` as the production local summarizer.
Enable background summary by default at 15,000 approximate working-memory
tokens, with target/high/hard limits of 12,000/15,000/16,384.

The worker uses dynamic 4K–32K context, runs in a single-job subprocess, and
unloads after publication. The application must not auto-download its weight
or send compaction data to a remote answer provider. If the verified private
weight is unavailable, runtime falls back to `trim_only`.

## Evidence

The selected model measured:

- 80.0% annotated fact recall on 200 single-generation cases;
- 72.5% on 40 four-generation rolling sessions;
- 84% decision and 92% correction recall;
- 8% mixed code/path placement recall.

The product owner explicitly revised acceptance to 100% schema, 80% single
recall, 70% rolling recall, 100% negative cleanliness, zero forbidden surface,
clean worker shutdown, and <= 8 GiB child RSS. The model passes that gate.

The production 15K smoke triggered at 15,288 tokens, allocated a 20,480-token
context, peaked at 6,030 MiB RSS, published, unloaded, checkpointed, reloaded,
and passed a real OpenRouter follow-up with the summary present in its prompt.

Full methodology, dataset provenance, model revisions, resource numbers, and
reproduction commands are in
[`../12-local-summary-model-selection.md`](../12-local-summary-model-selection.md).

## Consequences

- Chat, voice, and the UI engine use the same production summary selection.
- Compaction is asynchronous and can take about 45 seconds at the measured 15K
  boundary; the current session remains available while it runs.
- `trim_only` remains the failure mode, not the normal mode.
- The next design should emit typed, ID-addressed state deltas and merge them
  deterministically; it must not add keyword/regex content rules.
- The 84%/92%/8% weak areas remain explicit technical debt and the v2 suites
  remain regression gates.
