from __future__ import annotations

from collections.abc import Callable

import pytest

from soca.knowledge.base import KnowledgeDocument
from soca.knowledge.markdown_vault import SearchScoringConfig
from soca.knowledge.retrievers.sparse_document import SparseDocumentRetriever


def _chunk_document(chunk_id: str, text: str) -> KnowledgeDocument:
    return KnowledgeDocument(
        id=chunk_id,
        path="wiki/shared.md",
        title="Shared Note",
        text=text,
        tags=("shared",),
    )


def test_lexical_document_fields_are_precomputed_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = (
        _chunk_document(
            "wiki/shared.md#1",
            "# First section\nneedle alpha body",
        ),
        _chunk_document(
            "wiki/shared.md#2",
            "# Second section\nneedle beta body",
        ),
    )
    retriever = SparseDocumentRetriever(documents, SearchScoringConfig())

    from soca.knowledge import markdown_vault

    original: Callable[[str], tuple[str, ...]] = markdown_vault.tokenize_terms
    tokenized_after_construction: list[str] = []

    def record_tokenization(text: str) -> tuple[str, ...]:
        tokenized_after_construction.append(text)
        return original(text)

    monkeypatch.setattr(markdown_vault, "tokenize_terms", record_tokenization)

    retriever.rank("needle alpha", limit=2)
    retriever.search("needle beta", limit=2)

    precomputed_inputs = {
        value
        for document in documents
        for value in (
            document.title,
            " ".join(document.tags),
            document.path.replace("/", " ").replace("-", " "),
            document.text,
        )
    }
    assert precomputed_inputs.isdisjoint(tokenized_after_construction)


def test_rank_uses_document_id_for_two_chunks_from_the_same_note() -> None:
    documents = (
        _chunk_document(
            "wiki/shared.md#1-2:first",
            "# Shared\nalpha beta first chunk",
        ),
        _chunk_document(
            "wiki/shared.md#3-4:second",
            "# Shared\nalpha beta second chunk",
        ),
    )
    retriever = SparseDocumentRetriever(documents, SearchScoringConfig())

    ranked = retriever.rank("alpha beta", limit=5)

    assert [hit.chunk_id for hit in ranked] == [
        "wiki/shared.md#1-2:first",
        "wiki/shared.md#3-4:second",
    ]
    assert [hit.rank for hit in ranked] == [1, 2]
    assert len({hit.chunk_id for hit in ranked}) == 2
