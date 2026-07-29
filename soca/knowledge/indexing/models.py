from __future__ import annotations

import json
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from soca.knowledge.indexing.identity import EmbeddingFingerprint, sha256_file
from soca.knowledge.retrievers.dense import (
    AITEAMVN_V2_MODEL_SHA256,
    AITEAMVN_V2_TOKENIZER_SHA256,
    default_model_home,
    production_embedding_fingerprint,
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    adapter: str
    model_id: str
    dimension: int
    source: str
    license: str
    cache_subdirectory: str
    revision: str


MODEL_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="aiteamvn-v2",
        adapter="sentence_transformers",
        model_id="AITeamVN/Vietnamese_Embedding_v2",
        dimension=1024,
        source="https://huggingface.co/AITeamVN/Vietnamese_Embedding_v2",
        license="apache-2.0",
        cache_subdirectory="knowledge/aiteamvn_v2",
        revision="18b44161e041bf1d3a333ab5144b5b7b93f914d2",
    ),
)


def model_spec(key: str) -> ModelSpec:
    for item in MODEL_REGISTRY:
        if item.key == key:
            return item
    raise KeyError(f"unknown knowledge model: {key}")


def model_status(key: str, *, model_home: Path | None = None) -> dict[str, object]:
    spec = model_spec(key)
    root = (model_home or default_model_home()) / spec.cache_subdirectory
    try:
        _verify_model_files(root, spec)
    except (FileNotFoundError, OSError, ValueError) as exc:
        return {
            "key": spec.key,
            "model_id": spec.model_id,
            "adapter": spec.adapter,
            "dimension": spec.dimension,
            "state": "missing",
            "error": str(exc),
        }
    return {
        "key": spec.key,
        "model_id": spec.model_id,
        "adapter": spec.adapter,
        "dimension": spec.dimension,
        "state": "installed",
        "revision": spec.revision,
        "path": str(root),
    }


def model_fingerprint(key: str) -> EmbeddingFingerprint:
    model_spec(key)
    return production_embedding_fingerprint()


def model_is_provisioned(key: str, *, model_home: Path | None = None) -> bool:
    spec = model_spec(key)
    root = (model_home or default_model_home()) / spec.cache_subdirectory
    try:
        _verify_model_layout(root, spec)
    except (FileNotFoundError, OSError, ValueError):
        return False
    return True


def load_model(key: str, *, model_home: Path | None = None, allow_download: bool = False):
    from soca.knowledge.retrievers.dense import VietnameseEmbeddingV2Model

    model_spec(key)
    if allow_download:
        raise ValueError("production model provisioning must use the pinned installer")
    return VietnameseEmbeddingV2Model(model_home=model_home)


def _verify_model_layout(root: Path, spec: ModelSpec) -> None:
    manifest_path = root / ".soca-model.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(f"model manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "key": spec.key,
        "repo_id": spec.model_id,
        "revision": spec.revision,
        "model_sha256": AITEAMVN_V2_MODEL_SHA256,
        "tokenizer_sha256": AITEAMVN_V2_TOKENIZER_SHA256,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("model manifest does not match the production lock")
    for name in ("model.safetensors", "tokenizer.json"):
        path = root / name
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"model artifact must be a regular non-symlink file: {path}")


def _verify_model_files(root: Path, spec: ModelSpec) -> None:
    _verify_model_layout(root, spec)
    if sha256_file(root / "model.safetensors") != AITEAMVN_V2_MODEL_SHA256:
        raise ValueError("model checksum mismatch")
    if sha256_file(root / "tokenizer.json") != AITEAMVN_V2_TOKENIZER_SHA256:
        raise ValueError("tokenizer checksum mismatch")


def _make_private(root: Path) -> None:
    paths = (root, *root.rglob("*"))
    modes: list[tuple[Path, int]] = []
    for path in paths:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"model cache must not contain symlinks: {path}")
        if stat.S_ISDIR(metadata.st_mode):
            mode = 0o700
        elif stat.S_ISREG(metadata.st_mode):
            mode = 0o600
        else:
            raise ValueError(f"model cache contains an unsupported file type: {path}")
        modes.append((path, mode))
    for path, mode in modes:
        if os.chmod in os.supports_follow_symlinks:
            os.chmod(path, mode, follow_symlinks=False)
        else:
            path.chmod(mode)


def install_model(key: str, *, model_home: Path | None = None) -> Path:
    spec = model_spec(key)
    home = model_home or default_model_home()
    target = home / spec.cache_subdirectory
    if target.exists():
        _verify_model_files(target, spec)
        _make_private(target)
        return target
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    legacy = home / "eval" / "aiteamvn_v2"
    temporary = target.parent / f".{target.name}.{uuid4().hex}.tmp"
    moved_legacy = False
    try:
        if legacy.is_dir():
            if sha256_file(legacy / "model.safetensors") != AITEAMVN_V2_MODEL_SHA256:
                raise ValueError("legacy model checksum mismatch")
            if sha256_file(legacy / "tokenizer.json") != AITEAMVN_V2_TOKENIZER_SHA256:
                raise ValueError("legacy tokenizer checksum mismatch")
            os.replace(legacy, temporary)
            moved_legacy = True
        else:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=spec.model_id,
                revision=spec.revision,
                local_dir=temporary,
                ignore_patterns=["*.h5", "*.msgpack", "onnx/**", "openvino/**", "tf_model.*"],
            )
        manifest = {
            "schema_version": 1,
            "key": spec.key,
            "repo_id": spec.model_id,
            "revision": spec.revision,
            "model_sha256": AITEAMVN_V2_MODEL_SHA256,
            "tokenizer_sha256": AITEAMVN_V2_TOKENIZER_SHA256,
        }
        (temporary / ".soca-model.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _verify_model_files(temporary, spec)
        _make_private(temporary)
        os.replace(temporary, target)
        moved_legacy = False
    except Exception:
        if moved_legacy and temporary.exists() and not legacy.exists():
            os.replace(temporary, legacy)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target
