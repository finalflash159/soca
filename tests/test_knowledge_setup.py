from __future__ import annotations

from pathlib import Path

from soca.core.knowledge_setup import build_knowledge_runtime_setup
from soca.knowledge.cached_source import CachedMarkdownVaultKnowledgeSource
from soca.knowledge.factory import RetrievalConfig
from soca.knowledge.hybrid_source import HybridKnowledgeSource


def test_setup_shares_one_cached_source_across_builder_and_tools(
    tmp_path: Path,
) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "runtime.md").write_text(
        "# Runtime\nPhoGPT runtime note.",
        encoding="utf-8",
    )

    setup = build_knowledge_runtime_setup(tmp_path, knowledge_limit=3)

    assert isinstance(setup.source, CachedMarkdownVaultKnowledgeSource)
    assert setup.builder.source is setup.source
    assert setup.search_tool.source is setup.source
    assert setup.read_tool.source is setup.source
    assert setup.builder.max_hits == 3
    assert setup.search_tool.max_limit == 3
    assert setup.status == "enabled"


def test_setup_can_build_hybrid_source_from_validated_factory_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "runtime.md").write_text("# Runtime\nHybrid note.", encoding="utf-8")

    import soca.knowledge.factory as factory

    monkeypatch.setattr(factory, "_build_model", lambda backend: object())
    setup = build_knowledge_runtime_setup(
        tmp_path,
        knowledge_limit=3,
        retrieval_config=RetrievalConfig(mode="hybrid"),
        index_home=tmp_path / "index",
    )

    assert isinstance(setup.source, HybridKnowledgeSource)
    assert setup.builder.source is setup.source
    assert setup.search_tool.source is setup.source
    assert setup.read_tool.source is setup.source


def test_hybrid_factory_failure_degrades_to_cached_sparse(
    tmp_path: Path,
    monkeypatch,
) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "runtime.md").write_text("# Runtime\nSparse fallback.", encoding="utf-8")

    import soca.knowledge.factory as factory

    def fail_model(backend: str) -> object:
        raise ImportError("optional dense package is unavailable")

    monkeypatch.setattr(factory, "_build_model", fail_model)
    setup = build_knowledge_runtime_setup(
        tmp_path,
        knowledge_limit=3,
        retrieval_config=RetrievalConfig(mode="hybrid"),
        index_home=tmp_path / "index",
    )

    assert isinstance(setup.source, CachedMarkdownVaultKnowledgeSource)
    assert setup.status == "enabled"
