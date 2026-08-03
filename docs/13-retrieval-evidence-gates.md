# Retrieval, Evidence Gate and Answer Verification

This note records the current retrieval and grounding path. The goal is not to make a
model sound fluent at any cost, but to keep a knowledge request on an explicit
chain:

```text
query → sparse/dense retrieval → relevance gate → selected evidence
      → context prompt → LLM synthesis → citation validation
```

When there is no evidence, runtime can still send an empty knowledge context to
the LLM. The grounding prompt tells the model to say that the vault is
insufficient; runtime does not assemble an answer from snippets in code.

## Retrieval

`HybridKnowledgeSource` runs sparse and dense retrieval independently when the
corresponding backend is enabled. Dense is not blocked by a lexical miss. Each
`KnowledgeHit` carries:

- `retrieval_backend`: `bm25`, `dense`, `hybrid` or `explicit_read`;
- `sparse_score`, `dense_score`, `fusion_score`;
- chunk id, path and line range for trace/citation.

The retrieval batch also exposes backend-local diagnostics: sparse/dense index
state (`ready`, `missing`, `stale`, `degraded` or `unavailable`), top scores,
and backend-local separation. `EvidenceDecision` carries those fields plus
query coverage into `RuntimeTrace`; an empty healthy result is
`insufficient`, while a missing model/index or failed dense-only call is
`unavailable`. The runtime never compares a sparse score with a dense cosine.

`RelevancePolicy` does not compare scores from different backends blindly. It
uses lexical coverage/normalized sparse score for sparse, a dense floor for dense,
and records top score, margin, accepted/rejected count. Known-backend hits below
the floor are rejected before prompt construction. Unversioned or unknown
backend hits are not admitted as grounded evidence. The diagnostic sparse
profile and production hybrid profile have separate calibrated policies.
Production hybrid uses the pinned `Vietnamese_Embedding_v2` distribution with
dense floor `0.52` and sparse coverage `0.95`; sparse-only profiles are
evaluation tools, not production fallbacks.

### From hit to admitted evidence

The retrieval result is not automatically prompt context. The runtime applies
the following typed sequence:

```text
backend snapshot
  → retrieve candidates
  → normalize backend-local diagnostics
  → apply the calibrated relevance policy
  → select bounded, non-duplicate passages
  → build citations and evidence decision
  → admit context to the selected LLM
```

The snapshot pins the vault revision, index generation, embedding fingerprint
and retrieval backend. The relevance policy then evaluates each backend using
its own score semantics. The selector limits evidence per document and keeps
line ranges so a citation can be rendered independently from the answer text.
If the snapshot is stale, missing or failed, the result is an unavailable
backend state; it is not converted into a lexical-only answer. If the snapshot
is healthy but every candidate is below the floor, the result is an empty
evidence decision and the LLM receives the explicit empty-context grounding
instruction.

## Context and citations

`KnowledgeContextBuilder` converts retrieval results into bounded context with a
grounding warning, evidence status and citations. If a tool returns raw hits but
the relevance gate rejects some of them, `AssistantRuntime` puts only selected
citations in `RuntimeTrace`; rejected hits no longer make the validator demand a
source that was not supplied to the model.

Memory search follows the same tool → `MemoryContext` → prompt → LLM path. Memory
snippets are untrusted references, not instructions. `memory.search` does not
return snippets directly as the final answer when an LLM is available.

## Answer validation and repair

The validator checks whether `[K#]`/`[M#]` provenance labels exist in the selected
citation set. `partial` does not automatically mean the answer is wrong: a model
may cite only the source directly used when several references were supplied.
It also emits a non-blocking shadow claim/evidence overlap score and
`supported`/`mixed`/`unsupported` label. This is telemetry for calibration, not
a regex pretending to be a factuality judge and not a production blocking rule.

For non-streaming LLM answers, if citations exist but labels are missing or
invalid, runtime makes at most one repair call. Engines with structured-output
support receive a strict JSON Schema whose citation enum is generated from the
selected evidence for that turn. The model selects the citation; runtime only
renders the selected valid labels and never inserts a factual answer. Engines
without structured output receive the same dynamic label set in a plain repair
prompt. If repair still fails, runtime blocks the answer with a stable safety
message; it does not release the uncited original.

`GroundingTurnPolicy` freezes the behavior:

| Evidence policy | Citation result | Action |
| --- | --- | --- |
| normal grounded | valid or partial | allow |
| normal grounded | missing or invalid | one repair, then block |
| Knowledge/Memory conflict disclosure | valid | allow |
| conflict disclosure | partial, missing or invalid | one repair, then block |
| abstain / unavailable with no citations | not applicable | allow |

Streaming validates the complete text after streaming. The controlled workflow
keeps progress/acknowledgement events separate from the terminal answer; a
failed verification is surfaced rather than replaced by an unvalidated answer.

### Citation presentation boundary

Citation identity and answer prose are separate protocol fields. The runtime
keeps the selected evidence list (`K1`, `K2`, … or `M1`, `M2`, …) in the
structured result and the UI renders a source block after the answer. The
speech/text sanitization boundary removes internal labels from spoken output
but does not remove the source receipt from the UI trace. A model cannot create
a new source ID: labels must belong to the evidence set for that turn.

## Showcase corpus boundary

`eval/fixtures/knowledge_vault` is for smoke tests and UI demos. It is sanitized,
not personal data and not a release benchmark. It now contains 16 substantial
learning notes and 18 life notes plus source notes covering fundamentals, DSA, systems, ML, DL,
LLM serving, networking, security, decisions, journal, finance and health. The
learning notes include an initial misunderstanding, examples,
invariants, trade-offs, failure cases and open questions. Life notes preserve
state/date/uncertainty; finance distinguishes budget, planned and receipt actual;
health keeps a safety boundary.

The outdated `wiki/dinh-duong/` tree and `wiki/life/project/` tree are removed.

## Checks and real-flow smoke

Automated checks on this branch:

- `uv run pytest -q`: **1643 passed, 4 skipped, 3 warnings**;
- `uv run ruff check soca tests`: **pass**;
- `uv run pyright soca`: **0 errors, 0 warnings**;
- showcase fixture: **36 indexed markdown notes**, structure/size/query smoke pass.

The local wiring smoke with `arcee_vylinh_3b_q4_k_m` passed:

```text
Knowledge answerable → structured repair → valid [K1]
Memory answerable    → structured repair → valid [M1]
Memory unanswerable  → abstain, zero citations, no repair
```

The remote smoke with OpenRouter `google/gemini-3.5-flash-lite` passed the
same three paths without repair: valid `[K1]`, valid `[M1]`, then an abstention
with zero citations. The key was loaded from `.env` and was not written to an
artifact. Remote is the primary answer-quality target; the small local model
is retained as a bounded offline smoke target, never as a runtime fallback and
not treated as an equivalent quality judge.

These are wiring/abstention smoke checks, not release-quality retrieval scores.
Entailment and citation correctness still need an independent labeled dataset;
the showcase corpus must not be used to claim benchmark quality.

The public screening result is recorded in `BENCHMARKS.md` and uses the
real XQuAD Vietnamese corpus under `eval/fixtures/real_rag_vault`, not the
showcase vault. The metric is now policy-accepted evidence recall, not raw
retriever recall. Cached sparse accepted 10/12 answerable paths and 0/8
unanswerable rows; raw sparse recall was 12/12. Hybrid FastEmbed accepted
12/12 and 0/8, with a 13.0 s cold model load and 45.9 ms warm p95. The cold
event is reported separately, and the cached-sparse recall trade-off remains a
follow-up calibration item rather than a release-quality claim.

## Remaining gaps

- entailment needs an independent benchmark and human calibration;
- answer verification remains bounded and typed; a failed release gate is
  surfaced rather than repaired by changing backend or model;
- unsupported platform or provider behavior must remain explicitly
  `deferred`/`blocked` with an owner in the release matrix.
