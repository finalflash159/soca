from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from soca.knowledge.indexing.identity import EmbeddingFingerprint


@dataclass(frozen=True)
class ModelSpec:
    key: str
    adapter: str
    model_id: str
    dimension: int
    source: str
    license: str
    cache_subdirectory: str


MODEL_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="fastembed-e5-small",
        adapter="fastembed",
        model_id="intfloat/multilingual-e5-small",
        dimension=384,
        source="https://huggingface.co/intfloat/multilingual-e5-small",
        license="mit",
        cache_subdirectory="fastembed",
    ),
)


def model_spec(key: str) -> ModelSpec:
    for item in MODEL_REGISTRY:
        if item.key == key:
            return item
    raise KeyError(f"unknown knowledge model: {key}")


def model_status(key: str, *, model_home: Path | None = None) -> dict[str, object]:
    from soca.knowledge.retrievers.dense import FastEmbedModel

    spec = model_spec(key)
    try:
        model = FastEmbedModel(model_name=spec.model_id, model_home=model_home, allow_download=False)
    except (ImportError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        return {
            "key": spec.key,
            "model_id": spec.model_id,
            "adapter": spec.adapter,
            "dimension": spec.dimension,
            "state": "missing",
            "error": str(exc),
        }
    fingerprint = getattr(model, "embedding_fingerprint", None)
    return {
        "key": spec.key,
        "model_id": spec.model_id,
        "adapter": spec.adapter,
        "dimension": spec.dimension,
        "state": "installed",
        "fingerprint": fingerprint.value if isinstance(fingerprint, EmbeddingFingerprint) else None,
    }


def load_model(key: str, *, model_home: Path | None = None, allow_download: bool = False):
    from soca.knowledge.retrievers.dense import FastEmbedModel

    spec = model_spec(key)
    return FastEmbedModel(
        model_name=spec.model_id,
        model_home=model_home,
        allow_download=allow_download,
    )
