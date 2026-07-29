from types import SimpleNamespace

from soca.knowledge.base import KnowledgeDocument, KnowledgeHit
from soca.knowledge.context import KnowledgeContextBuilder
from soca.knowledge.hybrid_source import DenseUnavailableError, RetrievalDiagnostics


class FakeKnowledgeSource:
    def __init__(self, hits):
        self.hits = hits
        self.last_query = None
        self.last_limit = None

    def search(self, query: str, limit: int = 5):
        self.last_query = query
        self.last_limit = limit
        return self.hits[:limit]

    def read(self, path: str):
        raise NotImplementedError


def make_hit(
    path: str,
    title: str,
    snippet: str,
    score: float = 1.0,
    *,
    line_start: int | None = None,
    line_end: int | None = None,
):
    return KnowledgeHit(
        document=KnowledgeDocument(
            id=path,
            path=path,
            title=title,
            text=snippet,
            tags=(),
        ),
        score=score,
        snippet=snippet,
        line_start=line_start,
        line_end=line_end,
    )


def test_builds_prompt_context_with_warning_and_citations():
    source = FakeKnowledgeSource(
        [
            make_hit(
                "wiki/dinh-duong/goi-y-bua-an.md",
                "Gợi Ý Bữa Ăn Lành Mạnh",
                "Bữa sáng nên có đạm, chất xơ và tinh bột vừa phải.",
            )
        ]
    )

    builder = KnowledgeContextBuilder(source, max_hits=3, max_chars=1000)
    context = builder.build("bữa sáng lành mạnh")

    assert source.last_query == "bữa sáng lành mạnh"
    assert source.last_limit == 12
    assert "untrusted references" in context.prompt_text
    assert "[K1] wiki/dinh-duong/goi-y-bua-an.md" in context.prompt_text
    assert "Bữa sáng nên có đạm" in context.prompt_text
    assert context.citations[0].path == "wiki/dinh-duong/goi-y-bua-an.md"
    assert context.citations[0].title == "Gợi Ý Bữa Ăn Lành Mạnh"


def test_enforces_hit_limit():
    source = FakeKnowledgeSource(
        [
            make_hit("a.md", "A", "alpha"),
            make_hit("b.md", "B", "beta"),
            make_hit("c.md", "C", "gamma"),
        ]
    )

    builder = KnowledgeContextBuilder(source, max_hits=2, max_chars=1000)
    context = builder.build("test")

    assert [hit.document.path for hit in context.hits] == ["a.md", "b.md"]
    assert len(context.citations) == 2
    assert "[K3]" not in context.prompt_text


def test_context_diversifies_chunks_across_documents() -> None:
    source = FakeKnowledgeSource(
        [
            make_hit("a.md", "A", "first section"),
            make_hit("a.md", "A", "second section"),
            make_hit("b.md", "B", "independent evidence"),
        ]
    )

    context = KnowledgeContextBuilder(source, max_hits=2, max_chars=1000).build("test")

    assert [hit.document.path for hit in context.hits] == ["a.md", "b.md"]


def test_enforces_character_budget():
    source = FakeKnowledgeSource(
        [
            make_hit("a.md", "A", "a" * 300),
            make_hit("b.md", "B", "b" * 300),
        ]
    )

    builder = KnowledgeContextBuilder(source, max_hits=5, max_chars=350)
    context = builder.build("test")

    assert len(context.prompt_text) <= 350
    assert len(context.hits) == 1
    assert context.citations[0].path == "a.md"


def test_empty_query_result_returns_warning_only():
    source = FakeKnowledgeSource([])

    builder = KnowledgeContextBuilder(source, max_hits=3, max_chars=1000)
    context = builder.build("không có gì")

    assert context.query == "không có gì"
    assert context.hits == ()
    assert context.citations == ()
    assert "No local knowledge notes found." in context.prompt_text


def test_build_from_hits_preserves_line_range_in_citations() -> None:
    source = FakeKnowledgeSource([])
    builder = KnowledgeContextBuilder(source, max_hits=3, max_chars=1000)
    hit = make_hit(
        "wiki/a.md",
        "A",
        "alpha",
        line_start=7,
        line_end=11,
    )

    context = builder.build_from_hits("alpha", (hit,))

    assert context.hits == (hit,)
    assert context.citations[0].line_start == 7
    assert context.citations[0].line_end == 11


def test_build_propagates_retrieval_diagnostics_to_context() -> None:
    hit = make_hit(
        "wiki/bayes.md",
        "Định lý Bayes",
        "Bayes cập nhật xác suất bằng bằng chứng.",
        score=0.8,
    )

    class RetrievalSource(FakeKnowledgeSource):
        def retrieve(self, query: str, *, limit: int):
            return SimpleNamespace(
                hits=(hit,),
                diagnostics=SimpleNamespace(
                    overall_state="ready",
                    sparse_top_score=12.0,
                    dense_top_score=0.82,
                    unavailable_reason="",
                ),
            )

    context = KnowledgeContextBuilder(RetrievalSource([])).build("Bayes")

    assert context.retrieval_state == "ready"
    assert context.sparse_top_score == 12.0
    assert context.dense_top_score == 0.82


def test_build_distinguishes_dense_unavailable_from_empty_evidence() -> None:
    class UnavailableSource(FakeKnowledgeSource):
        def retrieve(self, query: str, *, limit: int):
            raise DenseUnavailableError("dense-only index refresh failed")

    context = KnowledgeContextBuilder(UnavailableSource([])).build("Bayes")

    assert context.hits == ()
    assert context.evidence_status == "unavailable"
    assert context.retrieval_state == "unavailable"
    assert context.evidence_reason == "dense-only index refresh failed"


def test_build_distinguishes_dense_failure_diagnostics_from_healthy_empty_index() -> None:
    class FailedDenseSource(FakeKnowledgeSource):
        def retrieve(self, query: str, *, limit: int):
            return SimpleNamespace(
                hits=(),
                diagnostics=RetrievalDiagnostics(
                    sparse_state="absent",
                    dense_state="failed",
                    index_state="ready",
                ),
            )

    context = KnowledgeContextBuilder(FailedDenseSource([])).build("Bayes")

    assert context.evidence_status == "unavailable"
    assert context.retrieval_state == "unavailable"
