from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from soca.knowledge.indexing.identity import EmbeddingFingerprint, sha256_file
from soca.knowledge.indexing.vector import stable_exact_top_k
from soca.knowledge.retriever import RankedHit
from soca.model_paths import default_model_root

FASTEMBED_E5_MODEL = "intfloat/multilingual-e5-small"
MODEL2VEC_MODEL = "minishlab/potion-multilingual-128M"
AITEAMVN_V2_MODEL = "AITeamVN/Vietnamese_Embedding_v2"
AITEAMVN_V2_REVISION = "18b44161e041bf1d3a333ab5144b5b7b93f914d2"
AITEAMVN_V2_MODEL_SHA256 = "2fa082ead5ade68225327b913339bbd5aa1e14bcd7888ff9b09d69752a8d1cee"
AITEAMVN_V2_TOKENIZER_SHA256 = "b74659c780d49afad7a7b9799868f75cbd3014fb6c34956e85a793028d38094a"
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "
_CUSTOM_E5_REGISTERED = False


def default_model_home() -> Path:
    return default_model_root()


def _normalize_rows(vectors: np.ndarray, *, expected_rows: int) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != expected_rows or array.shape[1] < 1:
        raise ValueError("embedding matrix has an invalid shape")
    if not np.isfinite(array).all():
        raise ValueError("embedding matrix must contain finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("embedding vectors must have non-zero norm")
    normalized = np.ascontiguousarray(array / norms, dtype=np.float32)
    normalized.setflags(write=False)
    return normalized


def _normalize_query(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1 or array.size < 1 or not np.isfinite(array).all():
        raise ValueError("query embedding is invalid")
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("query embedding must have non-zero finite norm")
    normalized = np.ascontiguousarray(array / norm, dtype=np.float32)
    normalized.setflags(write=False)
    return normalized


class EmbeddingModel(Protocol):
    @property
    def model_id(self) -> str: ...

    def embed_documents(self, texts: tuple[str, ...]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def production_embedding_fingerprint() -> EmbeddingFingerprint:
    return EmbeddingFingerprint(
        adapter="sentence_transformers",
        adapter_version="runtime-v1",
        model_id=AITEAMVN_V2_MODEL,
        model_revision=AITEAMVN_V2_REVISION,
        artifact_digest=AITEAMVN_V2_MODEL_SHA256,
        tokenizer_digest=AITEAMVN_V2_TOKENIZER_SHA256,
        dimension=1024,
        pooling="mean",
        normalize=True,
        max_length=8192,
    )


class VietnameseEmbeddingV2Model:
    def __init__(self, *, model_home: Path | None = None) -> None:
        root = (model_home or default_model_home()) / "knowledge" / "aiteamvn_v2"
        manifest_path = root / ".soca-model.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"production embedding manifest is missing: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": 1,
            "repo_id": AITEAMVN_V2_MODEL,
            "revision": AITEAMVN_V2_REVISION,
            "model_sha256": AITEAMVN_V2_MODEL_SHA256,
            "tokenizer_sha256": AITEAMVN_V2_TOKENIZER_SHA256,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("production embedding manifest does not match the pinned model")
        if sha256_file(root / "model.safetensors") != AITEAMVN_V2_MODEL_SHA256:
            raise ValueError("production embedding model checksum mismatch")
        if sha256_file(root / "tokenizer.json") != AITEAMVN_V2_TOKENIZER_SHA256:
            raise ValueError("production embedding tokenizer checksum mismatch")
        from sentence_transformers import SentenceTransformer

        self._root = root
        self._model = SentenceTransformer(
            str(root),
            device="cpu",
            local_files_only=True,
            trust_remote_code=False,
        )

    @property
    def model_id(self) -> str:
        return f"sentence_transformers:{AITEAMVN_V2_MODEL}"

    @property
    def embedding_fingerprint(self) -> EmbeddingFingerprint:
        return production_embedding_fingerprint()

    def embed_documents(self, texts: tuple[str, ...]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return _normalize_rows(np.asarray(vectors), expected_rows=len(texts))

    def embed_query(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("query must not be empty")
        return self.embed_documents((text,))[0]


class FastEmbedModel:
    def __init__(
        self,
        model_name: str = FASTEMBED_E5_MODEL,
        *,
        model_home: Path | None = None,
        allow_download: bool = False,
    ) -> None:
        global _CUSTOM_E5_REGISTERED

        from fastembed import TextEmbedding
        from fastembed.common.model_description import ModelSource, PoolingType

        supported = {
            item["model"]
            for item in TextEmbedding.list_supported_models()
            if isinstance(item, dict) and isinstance(item.get("model"), str)
        }
        if model_name not in supported:
            if model_name != FASTEMBED_E5_MODEL:
                raise ValueError(f"unsupported FastEmbed model: {model_name}")
            TextEmbedding.add_custom_model(
                model=model_name,
                pooling=PoolingType.MEAN,
                normalization=True,
                sources=ModelSource(hf=model_name),
                dim=384,
                model_file="onnx/model.onnx",
            )
            _CUSTOM_E5_REGISTERED = True
        self._custom_e5 = model_name == FASTEMBED_E5_MODEL and _CUSTOM_E5_REGISTERED

        self._model_name = model_name
        cache_dir = (model_home or default_model_home()) / "fastembed"
        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir),
            local_files_only=not allow_download,
        )

    @property
    def model_id(self) -> str:
        return f"fastembed:{self._model_name}"

    @property
    def embedding_fingerprint(self) -> EmbeddingFingerprint:
        return EmbeddingFingerprint(
            adapter="fastembed",
            adapter_version="runtime-v1",
            model_id=self._model_name,
            dimension=384 if self._model_name == FASTEMBED_E5_MODEL else 0,
            query_prefix=E5_QUERY_PREFIX if self._custom_e5 else "",
            passage_prefix=E5_PASSAGE_PREFIX if self._custom_e5 else "",
            pooling="mean",
            normalize=True,
            max_length=512,
        )

    def embed_documents(self, texts: tuple[str, ...]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if self._custom_e5:
            inputs = tuple(f"{E5_PASSAGE_PREFIX}{text}" for text in texts)
            output = self._model.embed(inputs)
        else:
            output = self._model.passage_embed(texts)
        vectors = np.asarray(list(output), dtype=np.float32)
        return _normalize_rows(vectors, expected_rows=len(texts))

    def embed_query(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("query must not be empty")
        output = (
            self._model.embed(f"{E5_QUERY_PREFIX}{text}")
            if self._custom_e5
            else self._model.query_embed(text)
        )
        vectors = list(output)
        if len(vectors) != 1:
            raise ValueError("embedding backend returned an unexpected query count")
        return _normalize_query(np.asarray(vectors[0], dtype=np.float32))


class Model2VecModel:
    def __init__(
        self,
        model_name: str = MODEL2VEC_MODEL,
        *,
        model_home: Path | None = None,
        allow_download: bool = False,
    ) -> None:
        from model2vec import StaticModel

        self._model_name = model_name
        model_path = (
            (model_home or default_model_home()) / "model2vec" / model_name.replace("/", "--")
        )
        if allow_download:
            downloaded = StaticModel.from_pretrained(model_name, force_download=False)
            model_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            downloaded.save_pretrained(model_path)
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model2Vec model is not provisioned at {model_path}")
        self._model = StaticModel.from_pretrained(
            model_path,
            force_download=False,
        )

    @property
    def model_id(self) -> str:
        return f"model2vec:{self._model_name}"

    @property
    def embedding_fingerprint(self) -> EmbeddingFingerprint:
        dimension = getattr(self._model, "dim", 0)
        return EmbeddingFingerprint(
            adapter="model2vec",
            adapter_version="runtime-v1",
            model_id=self._model_name,
            dimension=int(dimension) if isinstance(dimension, int) else 0,
            pooling="static",
            normalize=True,
        )

    def embed_documents(self, texts: tuple[str, ...]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = np.asarray(self._model.encode(list(texts)), dtype=np.float32)
        return _normalize_rows(vectors, expected_rows=len(texts))

    def embed_query(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("query must not be empty")
        vectors = np.asarray(self._model.encode([text]), dtype=np.float32)
        normalized = _normalize_rows(vectors, expected_rows=1)
        return normalized[0]


@dataclass(frozen=True)
class DenseIndex:
    model_id: str
    source_digest: str
    chunk_ids: tuple[str, ...]
    vectors: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("dense model id must not be empty")
        if (
            not isinstance(self.source_digest, str)
            or len(self.source_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.source_digest)
        ):
            raise ValueError("dense source digest must be a lowercase SHA-256 digest")
        if not isinstance(self.chunk_ids, (tuple, list)) or any(
            not isinstance(chunk_id, str) or not chunk_id.strip() for chunk_id in self.chunk_ids
        ):
            raise ValueError("dense chunk ids must be non-empty strings")
        frozen_chunk_ids = tuple(self.chunk_ids)
        if len(frozen_chunk_ids) != len(set(frozen_chunk_ids)):
            raise ValueError("dense chunk ids must be unique")
        frozen = _normalize_rows(self.vectors, expected_rows=len(frozen_chunk_ids))
        object.__setattr__(self, "chunk_ids", frozen_chunk_ids)
        object.__setattr__(self, "vectors", frozen)

    @property
    def dimension(self) -> int:
        return self.vectors.shape[1]


@dataclass(frozen=True)
class DenseRanking:
    hits: tuple[RankedHit, ...]
    max_score: float | None


class DenseRetriever:
    def __init__(self, index: DenseIndex, model: EmbeddingModel) -> None:
        if index.model_id != model.model_id:
            raise ValueError("dense index model does not match embedding model")
        self.index = index
        self.model = model

    @property
    def available(self) -> bool:
        return bool(self.index.chunk_ids)

    def rank_with_score(self, query: str, *, limit: int) -> DenseRanking:
        if limit < 1:
            raise ValueError("limit must be positive")
        if not self.available or not query.strip():
            return DenseRanking((), None)

        query_vector = _normalize_query(self.model.embed_query(query))
        if query_vector.shape[0] != self.index.dimension:
            raise ValueError("query embedding dimension does not match dense index")
        scores = self.index.vectors @ query_vector
        order = stable_exact_top_k(scores, self.index.chunk_ids, limit=limit)
        hits = tuple(
            RankedHit(
                chunk_id=self.index.chunk_ids[index],
                rank=rank,
                score=float(scores[index]),
            )
            for rank, index in enumerate(order, start=1)
        )
        return DenseRanking(
            hits=hits,
            max_score=float(np.max(scores)),
        )

    def rank(self, query: str, *, limit: int) -> list[RankedHit]:
        return list(self.rank_with_score(query, limit=limit).hits)
