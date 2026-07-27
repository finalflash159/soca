"""Reproducible sparse/dense lifecycle probe.

This measures index lifecycle mechanics with a deterministic fake embedder. It
is deliberately not a Vietnamese retrieval-quality benchmark; model/backend
quality belongs in the guarded RAG and vector-backend reports.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from soca.knowledge.indexing.coordinator import IndexCoordinator
from soca.knowledge.indexing.identity import CorpusSpec, EmbeddingFingerprint
from soca.knowledge.markdown_vault import MarkdownVaultKnowledgeSource


class ProbeEmbeddingModel:
    model_id = "probe:lifecycle"
    embedding_fingerprint = EmbeddingFingerprint(
        adapter="probe",
        adapter_version="v1",
        model_id="lifecycle",
        dimension=16,
    )

    def __init__(self) -> None:
        self.document_calls = 0
        self.document_rows = 0

    def embed_documents(self, texts: tuple[str, ...]) -> np.ndarray:
        self.document_calls += 1
        self.document_rows += len(texts)
        rows = []
        for text in texts:
            values = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.float32)
            values = np.resize(values, 16)
            rows.append(values + 1.0)
        return np.asarray(rows, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        del text
        return np.ones(16, dtype=np.float32)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=24)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def run(documents: int) -> dict[str, object]:
    if documents < 1:
        raise ValueError("--documents must be positive")
    with tempfile.TemporaryDirectory(prefix="soca-index-lifecycle-") as directory:
        root = Path(directory)
        wiki = root / "wiki"
        wiki.mkdir()
        for index in range(documents):
            (wiki / f"note-{index:04d}.md").write_text(
                f"# Note {index}\nVietnamese lifecycle benchmark content {index}.",
                encoding="utf-8",
            )
        reader = MarkdownVaultKnowledgeSource(root, include_globs=("wiki/**/*.md",))
        model = ProbeEmbeddingModel()
        coordinator = IndexCoordinator(
            reader,
            spec=CorpusSpec(root),
            index_home=root / "index-home",
            model=model,
        )
        started = time.perf_counter()
        first = coordinator.build_blocking(dense=True)
        full_build_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        warm = coordinator.snapshot()
        warm_snapshot_ms = (time.perf_counter() - started) * 1000
        changed = wiki / "note-0000.md"
        changed.write_text("# Note 0\nUpdated lifecycle benchmark content.", encoding="utf-8")
        started = time.perf_counter()
        second = coordinator.build_blocking(dense=True)
        edit_build_ms = (time.perf_counter() - started) * 1000
        renamed = wiki / "note-0001.md"
        renamed.rename(wiki / "note-renamed.md")
        started = time.perf_counter()
        third = coordinator.build_blocking(dense=True)
        rename_build_ms = (time.perf_counter() - started) * 1000
        return {
            "kind": "index_lifecycle_probe",
            "documents": documents,
            "chunks": len(first.sparse.index.chunks),
            "full_build_ms": round(full_build_ms, 3),
            "warm_snapshot_ms": round(warm_snapshot_ms, 3),
            "edit_build_ms": round(edit_build_ms, 3),
            "rename_build_ms": round(rename_build_ms, 3),
            "edit_embedded_rows": second.dense.embedded_rows if second.dense else None,
            "rename_reused_rows": third.dense.reused_rows if third.dense else None,
            "model_document_calls": model.document_calls,
            "model_document_rows": model.document_rows,
            "warm_dense_state": str(warm.dense_state),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        }


def main() -> int:
    args = _parser().parse_args()
    report = run(args.documents)
    rendered = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
