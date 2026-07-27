from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from eval.embedding_candidates import (
    EVAL_CANDIDATES,
    VietnameseEvalEmbedding,
    build_eval_candidate,
)


def _install_fake_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[dict[str, object]],
) -> None:
    class FakeSentenceTransformer:
        def __init__(self, path: str, *, local_files_only: bool) -> None:
            calls.append({"path": path, "local_files_only": local_files_only})

        def encode(
            self,
            texts: list[str],
            *,
            convert_to_numpy: bool,
            normalize_embeddings: bool,
        ) -> np.ndarray:
            calls.append(
                {
                    "texts": texts,
                    "convert_to_numpy": convert_to_numpy,
                    "normalize_embeddings": normalize_embeddings,
                }
            )
            return np.array([[3.0, 4.0] for _ in texts], dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )


@pytest.mark.parametrize("candidate", tuple(EVAL_CANDIDATES))
def test_eval_candidate_loads_local_only_and_normalizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate: str,
) -> None:
    model_dir = tmp_path / "eval" / candidate
    model_dir.mkdir(parents=True)
    calls: list[dict[str, object]] = []
    _install_fake_sentence_transformers(monkeypatch, calls)

    if candidate == "bkai_phobert_seg":
        monkeypatch.setitem(
            sys.modules,
            "underthesea",
            types.SimpleNamespace(word_tokenize=lambda text, format: f"TOKENIZED({text})"),
        )

    model = VietnameseEvalEmbedding(candidate, model_home=tmp_path)
    vectors = model.embed_documents(("xin chào", "hỏi đáp"))
    query = model.embed_query("câu hỏi")

    assert calls[0] == {"path": str(model_dir), "local_files_only": True}
    assert calls[1]["convert_to_numpy"] is True
    assert calls[1]["normalize_embeddings"] is False
    expected_texts = (
        ["TOKENIZED(xin chào)", "TOKENIZED(hỏi đáp)"]
        if candidate == "bkai_phobert_seg"
        else ["xin chào", "hỏi đáp"]
    )
    assert calls[1]["texts"] == expected_texts
    assert calls[2]["texts"] == (
        ["TOKENIZED(câu hỏi)"] if candidate == "bkai_phobert_seg" else ["câu hỏi"]
    )
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), np.ones(2))
    np.testing.assert_allclose(query, np.array([0.6, 0.8], dtype=np.float32))
    assert not vectors.flags.writeable
    assert not query.flags.writeable


def test_eval_candidate_rejects_unknown_or_missing_models(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown"):
        VietnameseEvalEmbedding("unknown", model_home=tmp_path)
    with pytest.raises(FileNotFoundError, match="not provisioned"):
        build_eval_candidate("aiteamvn_bge_m3")


def test_eval_candidate_rejects_invalid_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "eval" / "aiteamvn_bge_m3"
    model_dir.mkdir(parents=True)

    class FakeSentenceTransformer:
        def __init__(self, path: str, *, local_files_only: bool) -> None:
            pass

        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            return np.array([[0.0, 0.0] for _ in texts], dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    model = VietnameseEvalEmbedding("aiteamvn_bge_m3", model_home=tmp_path)

    with pytest.raises(ValueError, match="zero"):
        model.embed_query("query")
