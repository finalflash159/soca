# Vietnamese RAG model selection

This note records the retrieval incident from 2026-07-27 and the model
selection evidence. It is deliberately separate from the LLM/remote-provider
choice: the failure happened before answer generation, in candidate retrieval.

## Incident

The default text knowledge source was `cached_sparse`. Its lexical scorer
returned the highest-scoring documents even when the query's distinctive terms
were absent from the vault. With the real `~/KnowledgeVault`, the query
`Ghi chú nói định lý Bayes thế nào?` returned nutrition notes although the
vault contains no Bayes note.

The first guard against this failure is corpus-derived: it uses document
frequency, IDF-weighted query coverage, and score separation. It contains no
Vietnamese stopword list or subject-specific vocabulary. It is a fail-closed
guard, not a replacement for a stronger retriever.

## Models considered

| Model | Role | Evidence | Practical trade-off |
| --- | --- | --- | --- |
| `intfloat/multilingual-e5-small` | Current FastEmbed dense backend | 384 dimensions; multilingual baseline already cached locally | Fast, but not Vietnamese-specialized and weaker than the larger candidates |
| `BAAI/bge-m3` | General multilingual reference | 1024 dimensions, 8192-token context, dense/sparse/multi-vector retrieval | Strong general reference; not Vietnamese-specialized |
| `AITeamVN/Vietnamese_Embedding_v2` | Vietnamese dense candidate | 0.6B parameters, 1024 dimensions, Apache-2.0; fine-tuned on Vietnamese retrieval triplets | About 2.29 GB in the local Sentence Transformers snapshot; CPU encoding is materially slower |
| `AITeamVN/Vietnamese_Reranker` | Vietnamese cross-encoder reranker | 0.6B parameters, 2304-token query+passage limit, Apache-2.0; fine-tuned on about 1.1M Vietnamese triplets | Best used only on top-N candidates; too expensive to run over the whole vault |
| `dangvantuan/vietnamese-document-embedding` | Long-document alternative | Vietnamese-focused, up to 8192 tokens, Apache-2.0 | Good long-context candidate, but not measured in this run |

The current repo's eval catalog now includes `aiteamvn_v2`; provision it with:

```bash
uv run python scripts/download_eval_embedding.py aiteamvn_v2 --local-files-only
```

The canonical model ID is `AITeamVN/Vietnamese_Embedding_v2`. The provisioning
helper has one explicit mirror fallback (`thanhtantran/Vietnamese_Embedding_v2`)
because the canonical snapshot was incomplete in this machine's cache. The
saved eval model still uses the candidate key `aiteamvn_v2`, so the production
retriever cannot accidentally consume a remote model by ID.

The reranker is provisioned separately for research and is not silently made a
production dependency.

## Measurements

Hardware: MacBook M4 Pro, macOS arm64, Python 3.11.14. The pre-migration smoke
corpus had eight labelled queries, so it is a correctness smoke test rather than
a model ranking benchmark.

### Eight-query smoke corpus

All measured variants reached Recall@5, MRR@10, and nDCG@10 of `1.0` on all
eight smoke queries. This is a ceiling effect; it does not justify choosing a
model.

### XQuAD Vietnamese slice (first 200 cases)

These runs use the same 200 labelled cases from
`eval/fixtures/real_rag_vault` and `eval/prompts/real_rag_vi.jsonl`.

| Variant | Recall@5 | MRR@10 | nDCG@10 | p50 query latency | p95 query latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current cached sparse + fail-closed guard | 0.945 | 0.881 | 0.899 | 29.4 ms | 30.6 ms |
| Current `multilingual-e5-small` hybrid + fail-closed guard | 0.975 | 0.960 | 0.964 | 93.2 ms | 97.7 ms |
| `Vietnamese_Embedding_v2` hybrid + fail-closed guard | 0.970 | 0.968 | 0.970 | 115.2 ms | 140.3 ms |

The v2 encoder improves ranking quality over the current dense model on this
slice, but it is not a decisive win on Recall@5. The reranker experiment on the
eight-note smoke set kept Recall@5 at `1.0`, but reduced MRR@10 to `0.9375`: the RAG
architecture query was reranked behind the ONNX note. This is why the reranker
is not default yet; it needs a candidate-quality and score-threshold benchmark
before it can be trusted as a no-answer gate.

### Full XQuAD Vietnamese run (1,193 cases)

This is the first full run after the fail-closed guard, using the current
working tree, `rrf_k=60`, `warm_repeats=1`, and `encoding_repeats=1`:

| Variant | Recall@5 | MRR@10 | nDCG@10 | warm p50 | warm p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Vietnamese_Embedding_v2` hybrid + guard | 0.975692 | 0.958761 | 0.963662 | 111.06 ms | 126.01 ms |

Slice breakdown for the same run:

| Slice | Cases | Recall@5 | MRR@10 | nDCG@10 | warm p50 | warm p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `learning_notes` | 1,190 | 0.975630 | 0.959330 | 0.964086 | 110.98 ms | 126.01 ms |
| `life_vault_project` | 3 | 1.000000 | 0.733333 | 0.795618 | 113.22 ms | 117.38 ms |

The cold pass had mean latency `132.51 ms` and p95 `127.92 ms`; warm mean was
`111.23 ms`. Isolated query encoding was `16.83 ms` mean and `24.36 ms` p95.
The small three-case life slice has perfect Recall@5 but is too small for model
selection. The full result is therefore evidence that v2 is viable, not proof
that it should replace the faster default.

### No-answer diagnostic

On the real nutrition-only vault, the Vietnamese reranker assigned strongly
negative scores to lexical candidates for Bayes/ONNX/RAG questions (roughly
`-9` to `-11`), while matching nutrition queries scored positive (roughly
`+2.7` to `+6.5`). This is promising evidence for a calibrated reranker
threshold, but these numbers were measured on a small diagnostic set and are
not yet a production threshold.

## Vector-search backend probe

Embedding-model selection and vector-search backend selection are independent:

- the embedding model determines vector geometry and retrieval quality;
- the search backend determines how an existing vector matrix is ranked;
- changing NumPy exact search to FAISS Flat must not re-embed documents;
- ANN introduces an additional recall trade-off that must be measured against
  an exact oracle.

The tracked `eval/eval_vector_backend.py` probe measured normalized random
`float32` vectors on the same MacBook M4 Pro. At 50,000 rows × 384 dimensions:

| Backend | p95/query | Recall@10 vs exact | Build |
| --- | ---: | ---: | ---: |
| Current matrix multiply + Python full sort | 13.914 ms | 1.0000 | — |
| Deterministic NumPy partition exact | 0.814 ms | 1.0000 | — |
| FAISS Flat exact | **0.474 ms** | **1.0000** | 4.69 ms |
| FAISS HNSW M32, `efSearch=512` | 3.038 ms | 0.8825 | 6.43 s |
| FAISS HNSW M32, `efSearch=1024` | 4.766 ms | 0.9700 | 6.43 s |
| USearch M32, `expansion_search=1024` | 10.773 ms | 0.9375 | 23.21 s |

At 50,000 rows × 1024 dimensions, NumPy exact measured `1.729 ms` p95 and
FAISS Flat measured `1.475 ms`, both with Recall@10 `1.0`.

This exploratory result supports replacing the current Python sort, but it does
not select an ANN implementation. Random Gaussian vectors and 40 queries do not
represent the geometry of E5 or Vietnamese_Embedding_v2. The ANN gate requires
at least 1,000 real queries, exact Recall@1/5/10, end-to-end RAG metrics, cold
load, build time, peak RSS, and disk size.

Full setup, commands, script hash, scale tables, and decision thresholds are in
[`BENCHMARKS.md`](../BENCHMARKS.md#p212--vector-search-backend-probe).
The backend research and primary-source comparison are in
[`notes/vector_backend_research.md`](../notes/vector_backend_research.md).

## Decision

1. Keep lexical retrieval and dense retrieval as complementary stages. The
   latest Vietnamese IR study also finds BM25 hybrids to be the most stable
   cross-domain strategy; a stronger embedding should not replace lexical
   evidence.
2. Keep the corpus-derived fail-closed guard as an incident regression test,
   with thresholds in `SearchScoringConfig`, not a list of Vietnamese rules.
3. Add `Vietnamese_Embedding_v2` as an explicit benchmark/provisioning
   candidate. The full 1,193-case run is now recorded, but the model is still
   not the default because the current fast baseline has a different latency /
   memory profile and v2 did not win Recall@5 on the first-200 comparison.
4. Treat `Vietnamese_Reranker` as an optional second-stage component. Before
   enabling it, measure rank quality, no-answer precision, CPU latency, and
   score calibration on positive and deliberately unsupported queries.
5. Keep normalized `.npy` vectors as the canonical dense generation. Search
   backend artifacts are derived caches and cannot define embedding identity.
6. Replace the current full Python score sort with deterministic NumPy
   partition top-k before adding a vector-index dependency.
7. Keep FAISS Flat as the primary optional exact candidate. Do not enable HNSW
   until it reaches Recall@10 `>=0.99`, end-to-end nDCG regression `<=0.005`,
   and at least 2× p95 speedup over the winning exact backend.

## Sources

- [AITeamVN Vietnamese Embedding v2 model card](https://huggingface.co/AITeamVN/Vietnamese_Embedding_v2)
- [AITeamVN Vietnamese Reranker model card](https://huggingface.co/AITeamVN/Vietnamese_Reranker)
- [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3)
- [Vietnamese IR study, Findings of EACL 2026](https://aclanthology.org/2026.findings-eacl.110/)
- [Vietnamese IR study PDF with model tables](https://aclanthology.org/2026.findings-eacl.110.pdf)
- [FAISS index catalog](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- [FAISS index-selection guide](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
- [FAISS index-I/O security warning](https://github.com/facebookresearch/faiss/wiki/Index-IO%2C-cloning-and-hyper-parameter-tuning)

The reproducible hybrid-retrieval JSON report is intentionally kept out of git
because eval results are machine-specific. Re-run the model-quality benchmark
with:

```bash
uv run python eval/eval_hybrid_retrieval.py \
  --vault eval/fixtures/real_rag_vault \
  --cases eval/prompts/real_rag_vi.jsonl \
  --variant hybrid --backend aiteamvn_v2 --rrf-k 60 \
  --warm-repeats 1 --encoding-repeats 1 \
  --output eval/results/real-rag-hybrid-aiteamvn-v2.json
```
