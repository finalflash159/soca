from __future__ import annotations

from pathlib import Path

import numpy as np

from soca.knowledge.retrievers.dense import EmbeddingModel, default_model_home

EVAL_CANDIDATES = {
    "aiteamvn_bge_m3": "AITeamVN/Vietnamese_Embedding",
    "aiteamvn_v2": "AITeamVN/Vietnamese_Embedding_v2",
    "bkai_phobert_seg": "bkai-foundation-models/vietnamese-bi-encoder",
}

# The v2 weights are also mirrored under this ID.  Keep the model's canonical
# ID above, but let the provisioning helper use an already-cached mirror when
# the canonical Hugging Face snapshot is incomplete.
EVAL_CANDIDATE_FALLBACKS = {
    "aiteamvn_v2": ("thanhtantran/Vietnamese_Embedding_v2",),
}


def _normalize(vectors: np.ndarray, *, rows: int) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != rows or array.shape[1] < 1:
        raise ValueError("candidate returned an invalid embedding shape")
    if not np.isfinite(array).all():
        raise ValueError("candidate returned non-finite embeddings")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("candidate returned a zero embedding")
    normalized = np.ascontiguousarray(array / norms, dtype=np.float32)
    normalized.setflags(write=False)
    return normalized


class VietnameseEvalEmbedding:
    def __init__(
        self,
        candidate: str,
        *,
        model_home: Path | None = None,
    ) -> None:
        if candidate not in EVAL_CANDIDATES:
            raise ValueError("unknown Vietnamese eval candidate")
        self._candidate = candidate
        self._remote_id = EVAL_CANDIDATES[candidate]
        path = (model_home or default_model_home()) / "eval" / candidate
        if not path.is_dir():
            raise FileNotFoundError(f"eval model is not provisioned at {path}")
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(str(path), local_files_only=True)

    @property
    def model_id(self) -> str:
        return f"eval:{self._remote_id}"

    def _prepare(self, texts: tuple[str, ...]) -> list[str]:
        if self._candidate != "bkai_phobert_seg":
            return list(texts)
        from underthesea import word_tokenize

        return [word_tokenize(text, format="text") for text in texts]

    def embed_documents(self, texts: tuple[str, ...]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self._model.encode(
            self._prepare(texts),
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return _normalize(vectors, rows=len(texts))

    def embed_query(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("query must not be empty")
        return self.embed_documents((text,))[0]


def build_eval_candidate(candidate: str) -> EmbeddingModel:
    return VietnameseEvalEmbedding(candidate)
