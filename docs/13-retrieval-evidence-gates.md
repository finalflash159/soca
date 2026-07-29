# 13 — Retrieval, evidence gate and answer verification

This note records the current Phase 5 retrieval and grounding path. The goal is not to make a
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

- `retrieval_backend`: `lexical_custom`, `dense`, `hybrid` or `unknown`;
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
the floor are rejected before prompt construction. Legacy sources without score
metadata remain accepted as `weak` for compatibility; that is a migration signal,
not a calibrated evidence claim. Cached sparse and hybrid have separate
calibrated policies: public XQuAD Vietnamese screening uses coverage `0.65` plus
sparse ratio `0.75` for cached sparse, and dense floor `0.85` with a conservative
sparse fallback coverage `0.95` for hybrid FastEmbed.

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

Streaming validates the complete text after streaming, but does not yet prevent
an earlier factual chunk from reaching the user. Pre-validation holdback is a
P6 release gate, not something hidden by the non-streaming validator.

## Showcase corpus boundary

`eval/fixtures/knowledge_vault` is for smoke tests and UI demos. It is sanitized,
not personal data and not a release benchmark. It now contains 16 substantial
learning notes and 18 life notes covering fundamentals, DSA, systems, ML, DL,
LLM serving, networking, security, decisions, journal, finance and health. The
learning notes include an initial misunderstanding, examples,
invariants, trade-offs, failure cases and open questions. Life notes preserve
state/date/uncertainty; finance distinguishes budget, planned and receipt actual;
health keeps a safety boundary.

The outdated `wiki/dinh-duong/` tree and `wiki/life/project/` tree are removed.

## Checks and real-flow smoke

Automated checks on this branch:

- `uv run pytest -q`: **1090 passed, 3 skipped**;
- `uv run ruff check soca tests`: **pass**;
- `uv run pyright soca`: **0 errors, 0 warnings**;
- showcase fixture: **35 indexed markdown notes**, structure/size/query smoke pass.

The P5 local wiring smoke with `arcee_vylinh_3b_q4_k_m` passed:

```text
Knowledge answerable → structured repair → valid [K1]
Memory answerable    → structured repair → valid [M1]
Memory unanswerable  → abstain, zero citations, no repair
```

The P5 remote smoke with OpenRouter `google/gemini-3.5-flash-lite` passed the
same three paths without repair: valid `[K1]`, valid `[M1]`, then an abstention
with zero citations. The key was loaded from `.env` and was not written to an
artifact. Remote is the primary P6 answer-quality target; the small local model
is retained as a bounded offline/fallback smoke target, not treated as an
equivalent quality judge.

These are wiring/abstention smoke checks, not release-quality retrieval scores.
Entailment and citation correctness still need an independent labeled dataset;
the showcase corpus must not be used to claim benchmark quality.

The P4 public screening result is recorded in `BENCHMARKS.md` and uses the
real XQuAD Vietnamese corpus under `eval/fixtures/real_rag_vault`, not the
showcase vault. The metric is now policy-accepted evidence recall, not raw
retriever recall. Cached sparse accepted 10/12 answerable paths and 0/8
unanswerable rows; raw sparse recall was 12/12. Hybrid FastEmbed accepted
12/12 and 0/8, with a 13.0 s cold model load and 45.9 ms warm p95. The cold
event is reported separately, and the cached-sparse recall trade-off remains a
follow-up calibration item rather than a release-quality claim.

## Remaining gaps

- dense production default and embedding model remain governed by the index
  lifecycle decision;
- entailment needs an independent benchmark and human calibration;
- streaming answer repair needs a holdback/controlled loop design;
- goal verification, bounded multi-step tool loops and retry policy belong to the
  controlled-loop phase.
