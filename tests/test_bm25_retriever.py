from __future__ import annotations

from soca.knowledge.index.models import MarkdownChunk
from soca.knowledge.retrievers.bm25 import Bm25ChunkRetriever


def _chunk(chunk_id: str, text: str) -> MarkdownChunk:
    return MarkdownChunk(
        chunk_id=chunk_id,
        document_path=f"wiki/{chunk_id}.md",
        title=chunk_id,
        tags=(),
        text=text,
        line_start=1,
        line_end=1,
    )


def test_bm25_chunk_retriever_returns_real_bm25_scores() -> None:
    retriever = Bm25ChunkRetriever(
        (
            _chunk("bayes", "định lý Bayes cập nhật xác suất bằng bằng chứng"),
            _chunk("onnx", "ONNX Runtime chạy mô hình trên CPU"),
        )
    )

    hits = retriever.rank("xác suất Bayes", limit=2)

    assert hits[0].chunk_id == "bayes"
    assert hits[0].score > hits[1].score
    assert retriever.backend == "bm25"


def test_bm25_chunk_retriever_indexes_note_metadata() -> None:
    retriever = Bm25ChunkRetriever(
        (
            MarkdownChunk(
                chunk_id="attention",
                document_path="wiki/learning/attention.md",
                title="Attention và Transformer",
                tags=("deep-learning", "transformer"),
                text="Các token trao đổi thông tin trong cùng một sequence.",
                line_start=1,
                line_end=1,
            ),
            _chunk("other", "Một đoạn ghi chú không nêu chủ đề."),
        )
    )

    hits = retriever.rank("transformer", limit=2)

    assert hits[0].chunk_id == "attention"
