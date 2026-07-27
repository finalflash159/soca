from soca.knowledge.base import KnowledgeDocument, KnowledgeHit
from soca.knowledge.context import KnowledgeContextBuilder


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


def make_hit(path: str, title: str, snippet: str, score: float = 1.0):
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
    assert source.last_limit == 3
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
