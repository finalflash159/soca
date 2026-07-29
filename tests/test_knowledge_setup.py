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
    (memory / "profile.md").write_text("# Memory\nSparse fallback.", encoding="utf-8")

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
