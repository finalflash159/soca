from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from soca.knowledge.retrievers import dense
from soca.knowledge.retrievers.dense import (
    E5_PASSAGE_PREFIX,
    E5_QUERY_PREFIX,
    FASTEMBED_E5_MODEL,
    FastEmbedModel,
    Model2VecModel,
    default_model_home,
)


@pytest.fixture(autouse=True)
def reset_custom_fastembed_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dense, "_CUSTOM_E5_REGISTERED", False)


def test_default_model_home_uses_absolute_xdg_data_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    assert default_model_home() == tmp_path / "soca" / "models"


def test_default_model_home_rejects_relative_xdg_data_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", "relative/cache")

    with pytest.raises(ValueError, match="absolute"):
        default_model_home()


def test_native_fastembed_uses_task_aware_apis_and_local_only_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    native_model = "BAAI/bge-small-en-v1.5"

    class FakeTextEmbedding:
        @classmethod
        def list_supported_models(cls) -> list[dict[str, str]]:
            return [{"model": native_model}]

        def __init__(self, **kwargs: Any) -> None:
            calls["constructor"] = kwargs

        def passage_embed(self, texts: tuple[str, ...]) -> list[np.ndarray]:
            calls["passages"] = texts
            return [
                np.array([3.0, 4.0], dtype=np.float64),
                np.array([0.0, 2.0], dtype=np.float64),
            ]

        def query_embed(self, text: str) -> list[np.ndarray]:
            calls["query"] = text
            return [np.array([6.0, 8.0], dtype=np.float64)]

        def embed(self, texts: object) -> list[np.ndarray]:
            raise AssertionError(f"native model must use task-aware APIs: {texts!r}")

    import fastembed

    monkeypatch.setattr(fastembed, "TextEmbedding", FakeTextEmbedding)

    model = FastEmbedModel(model_name=native_model, model_home=tmp_path)
    documents = model.embed_documents(("alpha", "beta"))
    query = model.embed_query("question")

    assert calls["constructor"] == {
        "model_name": native_model,
        "cache_dir": str(tmp_path / "fastembed"),
        "local_files_only": True,
    }
    assert calls["passages"] == ("alpha", "beta")
    assert calls["query"] == "question"
    assert model.model_id == f"fastembed:{native_model}"
    np.testing.assert_allclose(np.linalg.norm(documents, axis=1), np.ones(2))
    np.testing.assert_allclose(query, np.array([0.6, 0.8], dtype=np.float32))
    assert documents.dtype == np.float32
    assert not documents.flags.writeable
    assert not query.flags.writeable


def test_custom_e5_registers_once_and_adds_one_task_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {"registrations": []}

    class FakeTextEmbedding:
        @classmethod
        def list_supported_models(cls) -> list[dict[str, str]]:
            return []

        @classmethod
        def add_custom_model(cls, **kwargs: Any) -> None:
            calls["registrations"].append(kwargs)

        def __init__(self, **kwargs: Any) -> None:
            calls["constructor"] = kwargs

        def embed(self, texts: str | tuple[str, ...]) -> list[np.ndarray]:
            calls.setdefault("embedded", []).append(texts)
            count = 1 if isinstance(texts, str) else len(texts)
            return [np.array([3.0, 4.0], dtype=np.float32) for _ in range(count)]

        def passage_embed(self, texts: object) -> list[np.ndarray]:
            raise AssertionError(f"custom E5 must add its explicit prefix: {texts!r}")

        def query_embed(self, text: object) -> list[np.ndarray]:
            raise AssertionError(f"custom E5 must add its explicit prefix: {text!r}")

    import fastembed

    monkeypatch.setattr(fastembed, "TextEmbedding", FakeTextEmbedding)

    model = FastEmbedModel(model_home=tmp_path)
    model.embed_documents(("alpha", "beta"))
    model.embed_query("question")

    assert len(calls["registrations"]) == 1
    registration = calls["registrations"][0]
    assert registration["model"] == FASTEMBED_E5_MODEL
    assert registration["dim"] == 384
    assert registration["model_file"] == "onnx/model.onnx"
    assert calls["embedded"] == [
        (
            f"{E5_PASSAGE_PREFIX}alpha",
            f"{E5_PASSAGE_PREFIX}beta",
        ),
        f"{E5_QUERY_PREFIX}question",
    ]


def test_fastembed_missing_local_model_does_not_enable_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_calls: list[dict[str, Any]] = []

    class FakeTextEmbedding:
        @classmethod
        def list_supported_models(cls) -> list[dict[str, str]]:
            return [{"model": "native/model"}]

        def __init__(self, **kwargs: Any) -> None:
            constructor_calls.append(kwargs)
            raise FileNotFoundError("model is not cached")

    import fastembed

    monkeypatch.setattr(fastembed, "TextEmbedding", FakeTextEmbedding)

    with pytest.raises(FileNotFoundError, match="not cached"):
        FastEmbedModel(model_name="native/model", model_home=tmp_path)

    assert constructor_calls[0]["local_files_only"] is True


def test_model2vec_loads_only_the_provisioned_directory_and_uses_encode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    model_dir = tmp_path / "model2vec" / "org--model"
    model_dir.mkdir(parents=True)

    class FakeStaticModel:
        @classmethod
        def from_pretrained(
            cls,
            path: Path,
            *,
            force_download: bool,
        ) -> FakeStaticModel:
            calls["load"] = (path, force_download)
            return cls()

        def encode(self, texts: list[str]) -> np.ndarray:
            calls.setdefault("encoded", []).append(texts)
            return np.array([[3.0, 4.0] for _ in texts], dtype=np.float64)

    import model2vec

    monkeypatch.setattr(model2vec, "StaticModel", FakeStaticModel)

    model = Model2VecModel(
        model_name="org/model",
        model_home=tmp_path,
    )
    documents = model.embed_documents(("alpha", "beta"))
    query = model.embed_query("question")

    assert calls["load"] == (model_dir, False)
    assert calls["encoded"] == [["alpha", "beta"], ["question"]]
    assert model.model_id == "model2vec:org/model"
    np.testing.assert_allclose(np.linalg.norm(documents, axis=1), np.ones(2))
    np.testing.assert_allclose(query, np.array([0.6, 0.8], dtype=np.float32))
    assert not documents.flags.writeable
    assert not query.flags.writeable


def test_model2vec_missing_local_model_does_not_call_hub_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    class FakeStaticModel:
        @classmethod
        def from_pretrained(cls, path: object, **kwargs: object) -> FakeStaticModel:
            calls.append((path, kwargs))
            return cls()

    import model2vec

    monkeypatch.setattr(model2vec, "StaticModel", FakeStaticModel)

    with pytest.raises(FileNotFoundError, match="not provisioned"):
        Model2VecModel(model_name="org/missing", model_home=tmp_path)

    assert calls == []


@pytest.mark.parametrize(
    "vectors",
    [
        np.array([[0.0, 0.0]], dtype=np.float32),
        np.array([[np.nan, 1.0]], dtype=np.float32),
        np.array([1.0, 2.0], dtype=np.float32),
    ],
)
def test_model2vec_rejects_invalid_backend_output(
    vectors: np.ndarray,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "model2vec" / "org--model"
    model_dir.mkdir(parents=True)

    class FakeStaticModel:
        @classmethod
        def from_pretrained(cls, path: object, **kwargs: object) -> FakeStaticModel:
            return cls()

        def encode(self, texts: list[str]) -> np.ndarray:
            return vectors

    import model2vec

    monkeypatch.setattr(model2vec, "StaticModel", FakeStaticModel)
    model = Model2VecModel(model_name="org/model", model_home=tmp_path)

    with pytest.raises(ValueError):
        model.embed_query("question")


@pytest.mark.parametrize(
    ("backend", "expected_model_id"),
    [
        ("fastembed", "fake:fastembed"),
        ("model2vec", "fake:model2vec"),
    ],
)
def test_download_command_selects_an_explicit_backend_and_runs_a_smoke_query(
    backend: str,
    expected_model_id: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import download_embedding

    calls: list[tuple[str, bool, str]] = []

    class FakeModel:
        def __init__(self, name: str, *, allow_download: bool) -> None:
            self.model_id = f"fake:{name}"
            self.name = name
            self.allow_download = allow_download

        def embed_query(self, text: str) -> np.ndarray:
            calls.append((self.name, self.allow_download, text))
            return np.ones(7, dtype=np.float32)

    monkeypatch.setattr(
        download_embedding,
        "FastEmbedModel",
        lambda *, allow_download: FakeModel(
            "fastembed",
            allow_download=allow_download,
        ),
    )
    monkeypatch.setattr(
        download_embedding,
        "Model2VecModel",
        lambda *, allow_download: FakeModel(
            "model2vec",
            allow_download=allow_download,
        ),
    )
    monkeypatch.setattr(sys, "argv", ["download_embedding.py", backend])

    assert download_embedding.main() == 0

    assert calls == [
        (
            backend,
            True,
            "kiểm tra mô hình tìm kiếm",
        )
    ]
    assert capsys.readouterr().out.strip() == f"{expected_model_id}: dimension=7"


def test_download_command_rejects_an_unknown_backend() -> None:
    from scripts.download_embedding import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["unknown"])


@pytest.mark.real_model
@pytest.mark.skipif(
    os.environ.get("SOCA_RUN_REAL_MODEL_TESTS") != "1",
    reason="set SOCA_RUN_REAL_MODEL_TESTS=1 after provisioning both embedding models",
)
@pytest.mark.parametrize("model_factory", [FastEmbedModel, Model2VecModel])
def test_provisioned_real_embedding_model_returns_a_normalized_vector(
    model_factory: type[FastEmbedModel] | type[Model2VecModel],
) -> None:
    vector = model_factory().embed_query("kiểm tra mô hình tìm kiếm")

    assert vector.ndim == 1
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()
    assert np.linalg.norm(vector) == pytest.approx(1.0)
