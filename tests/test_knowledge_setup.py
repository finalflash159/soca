from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from soca.core.knowledge_setup import build_knowledge_runtime_setup
from soca.knowledge.factory import RetrievalConfig
from soca.knowledge.hybrid_source import HybridKnowledgeSource


class FakeEmbeddingModel:
    model_id = "fake:knowledge"

    def embed_documents(self, texts: tuple[str, ...]) -> np.ndarray:
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        del text
        return np.asarray([1.0, 0.0], dtype=np.float32)


def test_setup_shares_one_production_hybrid_source_across_builder_and_tools(
    tmp_path: Path,
    monkeypatch,
) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "runtime.md").write_text(
        "# Runtime\nPhoGPT runtime note.",
        encoding="utf-8",
    )

    import soca.knowledge.factory as factory

    monkeypatch.setattr(factory, "_build_model", lambda backend: FakeEmbeddingModel())
    setup = build_knowledge_runtime_setup(tmp_path, knowledge_limit=3)

    assert isinstance(setup.source, HybridKnowledgeSource)
    assert setup.builder.source is setup.source
    assert setup.search_tool.source is setup.source
    assert setup.read_tool.source is setup.source
    assert setup.builder.max_hits == 3
    assert setup.search_tool.max_limit == 3
    assert setup.status.startswith("enabled:hybrid")


def test_setup_can_build_hybrid_source_from_validated_factory_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "runtime.md").write_text("# Runtime\nHybrid note.", encoding="utf-8")

    import soca.knowledge.factory as factory

    monkeypatch.setattr(factory, "_build_model", lambda backend: FakeEmbeddingModel())
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


def test_hybrid_factory_failure_is_explicit_and_does_not_fallback(
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
    with pytest.raises(ImportError, match="optional dense package"):
        build_knowledge_runtime_setup(
            tmp_path,
            knowledge_limit=3,
            retrieval_config=RetrievalConfig(mode="hybrid"),
            index_home=tmp_path / "index",
        )


def test_memory_hybrid_factory_failure_does_not_fallback(tmp_path: Path, monkeypatch) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "archive-note.md").write_text("# Memory\nArchive note.", encoding="utf-8")

    import soca.knowledge.factory as factory
    from soca.core.memory_setup import MemoryRuntimeConfig, build_memory_runtime_setup
    from soca.memory import SessionMemory

    monkeypatch.setattr(
        factory,
        "_build_model",
        lambda backend: (_ for _ in ()).throw(ImportError("dense package unavailable")),
    )
    with pytest.raises(ImportError, match="dense package unavailable"):
        build_memory_runtime_setup(
            tmp_path,
            session=SessionMemory(summary_enabled=False),
            config=MemoryRuntimeConfig(
                retrieval_mode="hybrid",
                dense_backend="aiteamvn_v2",
            ),
            index_home=tmp_path / "index",
        )


def test_knowledge_setup_closes_source_when_runtime_assembly_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import soca.core.knowledge_setup as setup_module

    class Source:
        retrieval_mode = "hybrid"

        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    source = Source()
    monkeypatch.setattr(setup_module, "build_knowledge_source", lambda *args, **kwargs: source)
    monkeypatch.setattr(
        setup_module,
        "KnowledgeCatalog",
        lambda provider: (_ for _ in ()).throw(RuntimeError("catalog failed")),
    )

    with pytest.raises(RuntimeError, match="catalog failed"):
        build_knowledge_runtime_setup(tmp_path, knowledge_limit=3)

    assert source.close_calls == 1


def test_memory_setup_closes_source_when_runtime_assembly_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import soca.core.memory_setup as setup_module
    from soca.memory import SessionMemory

    (tmp_path / "memory").mkdir()

    class Source:
        retrieval_mode = "hybrid"

        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    source = Source()
    monkeypatch.setattr(setup_module, "build_retrieval_source", lambda *args, **kwargs: source)
    monkeypatch.setattr(
        setup_module,
        "CoreMemoryStore",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("core failed")),
    )

    with pytest.raises(RuntimeError, match="core failed"):
        setup_module.build_memory_runtime_setup(
            tmp_path,
            session=SessionMemory(summary_enabled=False),
            config=setup_module.MemoryRuntimeConfig(retrieval_mode="hybrid"),
            index_home=tmp_path / "index",
        )

    assert source.close_calls == 1


def test_factory_closes_source_when_watcher_start_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import soca.knowledge.factory as factory

    class Source:
        retrieval_mode = "hybrid"

        def __init__(self, *args, **kwargs) -> None:
            self.close_calls = 0

        def activate_watcher(self, *, interval_seconds: float) -> None:
            del interval_seconds
            raise RuntimeError("watcher failed")

        def close(self) -> None:
            self.close_calls += 1

    source = Source()
    monkeypatch.setattr(factory, "HybridKnowledgeSource", lambda *args, **kwargs: source)
    monkeypatch.setattr(factory, "_build_model", lambda backend: FakeEmbeddingModel())

    with pytest.raises(RuntimeError, match="watcher failed"):
        factory.build_retrieval_source(
            tmp_path,
            include_globs=("wiki/**/*.md",),
            config=RetrievalConfig(mode="hybrid"),
            index_home=tmp_path / "index",
        )

    assert source.close_calls == 1
