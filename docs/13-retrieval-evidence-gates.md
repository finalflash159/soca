# 13 — Retrieval, evidence gate and answer verification

This note records the current Phase 4 retrieval path. The goal is not to make a
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

`RelevancePolicy` does not compare scores from different backends blindly. It
uses lexical coverage/normalized sparse score for sparse, a dense floor for dense,
and records top score, margin, accepted/rejected count. Known-backend hits below
the floor are rejected before prompt construction. Legacy sources without score
metadata remain accepted as `weak` for compatibility; that is a migration signal,
not a calibrated evidence claim.

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
Full entailment/factuality remains a shadow signal that needs a held-out
evaluation set; regex is not pretending to be a judge.

For non-streaming LLM answers, if citations exist but labels are missing or
invalid, runtime makes at most one repair call. The repair prompt is limited to
selected evidence. If repair still fails, the original answer is kept and the
trace records `answer_repair_attempted/succeeded`; there is no unbounded loop and
no silent fact insertion. Streaming validates the complete text after streaming,
but does not repair a sentence already sent to the user. A holdback/controlled
loop is still a later design task.

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

Real local flow with `arcee_vylinh_3b_q4_k_m` passed:

```text
knowledge.search → relevance → KnowledgeContext → local LLM
route=knowledge_llm, tool=true, llm=true, Bayes answer with [K1]
```

Real remote flow with OpenRouter `google/gemini-3.5-flash-lite` passed the same
path. A no-answer query (`Sao Bắc Cực X9`) produced zero citations, evidence
`insufficient/no_hits`, and a Vietnamese answer saying there was not enough
information in the vault. The key was loaded from `.env` and was not written to
an artifact.

These are wiring/abstention smoke checks, not release-quality retrieval scores.
Entailment and citation correctness still need an independent labeled dataset;
the showcase corpus must not be used to claim benchmark quality.

## Remaining gaps

- dense production default and embedding model remain governed by the index
  lifecycle decision;
- entailment needs an independent benchmark and human calibration;
- streaming answer repair needs a holdback/controlled loop design;
- goal verification, bounded multi-step tool loops and retry policy belong to the
  controlled-loop phase.
