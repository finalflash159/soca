# Vector-search backend research

> Research date: 2026-07-28. This note compares storage/search engines only. Embedding-model quality is evaluated separately in [`docs/10-vietnamese-rag-model-selection.md`](../docs/10-vietnamese-rag-model-selection.md).

## Scope

The current dense knowledge index stores normalized `float32` vectors in immutable `.npy` files and performs a matrix-vector dot product followed by a full Python sort. The lifecycle-v2 research asks two separate questions:

1. How should document vectors be versioned, refreshed, verified, and garbage-collected?
1. Which search backend should rank an already-built vector generation?

A vector database does not solve the first question automatically. The planned design therefore keeps a SQLite lifecycle catalog and treats any FAISS, USearch, or HNSW artifact as a derived search index. The normalized `.npy` matrix remains the rebuildable source of truth.

## Candidate assessment

| Backend | Search | Persistence/update notes | Decision | | --- | --- | --- | --- | | NumPy `.npy` | Exact | Existing mmap format; generation rebuild | Default v2 baseline after replacing full sort | | FAISS `IndexFlatIP` | Exact | Reconstruct from `.npy`; all vectors resident in RAM | Primary exact candidate | | FAISS `IndexHNSWFlat` | Approximate | Graph overhead; no remove support | Benchmark gate only | | USearch HNSW | Approximate | Save/load/mmap view and remove API | Benchmark gate only | | hnswlib | Approximate | Soft delete/update; `ef` reset after load | Secondary candidate; PyPI packaging risk | | sqlite-vec | Exact scan | Pre-v1 SQLite extension | Do not adopt yet | | DuckDB VSS | Approximate | Persistent HNSW remains experimental | Reject for production | | LanceDB | Exact/indexed | Own table/version/compaction lifecycle | Revisit for much larger corpora | | Qdrant | Vector database | Local mode or server; local snapshots unavailable | Revisit for service/multi-client mode |

## Primary-source findings

- FAISS documents `IndexFlatIP` as exhaustive exact search using `4 * d` bytes per vector; cosine search uses normalized vectors.
- FAISS HNSW adds graph memory and does not support vector removal.
- FAISS CPU indexes support concurrent read-only searches, while mutation requires external synchronization.
- FAISS warns that `read_index` does not validate input and that a malicious artifact can cause out-of-memory behavior or code execution. SoCa must verify path scope, mode, expected size, and SHA-256 before invoking a native loader.
- `faiss-cpu 1.14.3` publishes a CPython 3.10+ macOS 14 ARM64 wheel, so native packaging is no longer an automatic blocker on the current machine.
- hnswlib's published API supports update and soft delete, but add/query are not mutually thread-safe and `ef` must be restored after load. PyPI `0.8.0` provides only a source distribution.
- DuckDB explicitly labels persistent VSS/HNSW support experimental and warns that WAL recovery can lose or corrupt the index after an unexpected shutdown.
- Qdrant documents that snapshots cannot be created in Python SDK local mode.

## Local exploratory probe

The tracked harness is [`eval/eval_vector_backend.py`](../eval/eval_vector_backend.py). The canonical summary and commands are in [`BENCHMARKS.md`](../BENCHMARKS.md). Raw machine-specific JSON is written below `eval/results/vector_backend/` and is gitignored by project convention.

The 2026-07-28 probe used normalized random Gaussian `float32` vectors, 40 singleton queries, and `k=10`. This is a backend microbenchmark, not a Vietnamese retrieval-quality result.

At 50,000 vectors × 384 dimensions:

- full Python sort: p95 about 14 ms, exact;
- deterministic NumPy partition search: p95 below 1 ms, exact;
- FAISS Flat: p95 about 0.5 ms, exact;
- FAISS HNSW needed `efSearch=1024` to approach 0.98 Recall@10 and was slower than exact search;
- USearch was slower than FAISS HNSW at the tested settings.

At 50,000 vectors × 1024 dimensions, FAISS Flat remained close to 1.5 ms p95 and NumPy exact remained below 2 ms p95.

These results reject “add HNSW now” but do not reject ANN permanently. Random vectors have a different geometry from E5 or Vietnamese_Embedding_v2. The production gate requires at least 1,000 real query vectors, exact ground truth, Recall@1/5/10, end-to-end RAG metrics, peak RSS, build time, and cold-load time.

## Decision

1. Fix the current full Python sort before adding a dependency.
1. Keep exact search as the v2 default while the target corpus remains within the measured latency and memory budget.
1. Keep FAISS Flat as the primary optional exact candidate.
1. Adopt HNSW only if exact p95 or memory exceeds a recorded budget and ANN achieves Recall@10 at least 0.99, end-to-end nDCG regression no worse than 0.005, and at least a 2× latency improvement.
1. Never re-embed documents merely because the search backend changes.
1. Never load a native search artifact before validating its provenance, expected size, and checksum.

## Sources

- [FAISS index catalog](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- [FAISS index-selection guide](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
- [FAISS remove and ID semantics](https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes)
- [FAISS thread safety](https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls)
- [FAISS index-I/O security warning](https://github.com/facebookresearch/faiss/wiki/Index-IO%2C-cloning-and-hyper-parameter-tuning)
- [`faiss-cpu` release artifacts](https://pypi.org/project/faiss-cpu/)
- [USearch API and persistence](https://github.com/unum-cloud/usearch)
- [hnswlib API](https://github.com/nmslib/hnswlib)
- [`hnswlib` release artifacts](https://pypi.org/project/hnswlib/)
- [DuckDB VSS persistence limitations](https://duckdb.org/docs/lts/core_extensions/vss)
- [Qdrant snapshot limitation in local mode](https://qdrant.tech/documentation/tutorials-operations/create-snapshot/)
- [LanceDB Python API and optimize lifecycle](https://lancedb.github.io/lancedb/python/python/)
