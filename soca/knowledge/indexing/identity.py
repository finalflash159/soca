from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CorpusKind = Literal["knowledge", "memory"]


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CorpusSpec:
    vault_path: Path
    kind: CorpusKind = "knowledge"
    include_globs: tuple[str, ...] = ("wiki/**/*.md",)
    exclude_dirs: tuple[str, ...] = (".obsidian", ".trash", "private")
    exclude_files: tuple[str, ...] = ("index.md", "log.md")
    max_file_bytes: int = 256 * 1024
    policy_version: str = "path-policy-v1"

    def __post_init__(self) -> None:
        if self.kind not in {"knowledge", "memory"}:
            raise ValueError("corpus kind must be knowledge or memory")
        if not self.include_globs or any(
            not isinstance(item, str) or not item or item.startswith("/") or "\\" in item
            for item in self.include_globs
        ):
            raise ValueError("include globs must be non-empty relative POSIX strings")
        if any(not isinstance(item, str) or not item for item in self.exclude_dirs):
            raise ValueError("exclude dirs must contain non-empty strings")
        if any(not isinstance(item, str) or not item for item in self.exclude_files):
            raise ValueError("exclude files must contain non-empty strings")
        if (
            isinstance(self.max_file_bytes, bool)
            or not isinstance(self.max_file_bytes, int)
            or self.max_file_bytes < 1
        ):
            raise ValueError("max_file_bytes must be positive")
        if not self.policy_version.strip():
            raise ValueError("policy_version must not be empty")

    @property
    def resolved_vault_path(self) -> str:
        return str(self.vault_path.expanduser().resolve())

    @property
    def policy(self) -> dict[str, object]:
        return {
            "include_globs": list(self.include_globs),
            "exclude_dirs": list(self.exclude_dirs),
            "exclude_files": list(self.exclude_files),
            "max_file_bytes": self.max_file_bytes,
            "policy_version": self.policy_version,
        }

    @property
    def corpus_identity(self) -> CorpusIdentity:
        return CorpusIdentity.from_spec(self)


@dataclass(frozen=True)
class CorpusIdentity:
    value: str
    vault_path: str
    kind: CorpusKind
    policy_json: str

    @classmethod
    def from_spec(cls, spec: CorpusSpec) -> CorpusIdentity:
        payload = {
            "schema_namespace": "soca-index-v2",
            "resolved_vault_path": spec.resolved_vault_path,
            "corpus_kind": spec.kind,
            **spec.policy,
        }
        return cls(
            value=sha256_text(canonical_json(payload)),
            vault_path=spec.resolved_vault_path,
            kind=spec.kind,
            policy_json=canonical_json(spec.policy),
        )

    @property
    def prefix(self) -> str:
        return self.value[:16]


@dataclass(frozen=True)
class ChunkerFingerprint:
    parser_version: str = "markdown-parser-v1"
    algorithm_version: str = "chunker-v1"
    target_tokens: int = 320
    overlap_lines: int = 2
    token_strategy: str = "unicode-word-v1"
    heading_policy: str = "section-heading-v1"
    normalization_policy: str = "markdown-text-preserve-v1"

    def __post_init__(self) -> None:
        if self.target_tokens < 32 or self.overlap_lines < 0:
            raise ValueError("invalid chunker fingerprint configuration")

    @property
    def payload(self) -> dict[str, object]:
        return {
            "parser_version": self.parser_version,
            "algorithm_version": self.algorithm_version,
            "target_tokens": self.target_tokens,
            "overlap_lines": self.overlap_lines,
            "token_strategy": self.token_strategy,
            "heading_policy": self.heading_policy,
            "normalization_policy": self.normalization_policy,
        }

    @property
    def value(self) -> str:
        return sha256_text(canonical_json(self.payload))


@dataclass(frozen=True)
class EmbeddingFingerprint:
    adapter: str
    adapter_version: str
    model_id: str
    model_revision: str = "unknown"
    artifact_digest: str = "unknown"
    tokenizer_digest: str = "unknown"
    dimension: int = 0
    query_prefix: str = ""
    passage_prefix: str = ""
    pooling: str = "unknown"
    normalize: bool = True
    max_length: int = 0
    truncation: bool = True

    def __post_init__(self) -> None:
        if not self.adapter.strip() or not self.model_id.strip():
            raise ValueError("embedding adapter and model_id are required")
        if isinstance(self.dimension, bool) or self.dimension < 0:
            raise ValueError("embedding dimension must be non-negative")

    @property
    def payload(self) -> dict[str, object]:
        return {
            "adapter": self.adapter,
            "adapter_version": self.adapter_version,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "artifact_digest": self.artifact_digest,
            "tokenizer_digest": self.tokenizer_digest,
            "dimension": self.dimension,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
            "pooling": self.pooling,
            "normalize": self.normalize,
            "max_length": self.max_length,
            "truncation": self.truncation,
        }

    @property
    def value(self) -> str:
        return sha256_text(canonical_json(self.payload))


@dataclass(frozen=True)
class SearchBackendFingerprint:
    kind: str = "numpy_exact"
    library_version: str = "numpy"
    metric: str = "inner_product"
    dtype: str = "float32"
    normalized: bool = True
    top_k_algorithm: str = "argpartition_boundary_ties_v1"
    parameters: Mapping[str, object] | None = None

    @property
    def payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "library_version": self.library_version,
            "metric": self.metric,
            "dtype": self.dtype,
            "normalized": self.normalized,
            "top_k_algorithm": self.top_k_algorithm,
            "parameters": dict(self.parameters or {}),
        }

    @property
    def value(self) -> str:
        return sha256_text(canonical_json(self.payload))


def embedding_fingerprint_for(model: object) -> EmbeddingFingerprint:
    configured = getattr(model, "embedding_fingerprint", None)
    if isinstance(configured, EmbeddingFingerprint):
        return configured
    if isinstance(configured, Mapping):
        payload = dict(configured)
        return EmbeddingFingerprint(**payload)
    model_id = str(getattr(model, "model_id", "unknown:model"))
    adapter, _, name = model_id.partition(":")
    dimension = getattr(model, "dimension", 0)
    return EmbeddingFingerprint(
        adapter=adapter or "unknown",
        adapter_version="runtime-unknown",
        model_id=name or model_id,
        dimension=int(dimension) if isinstance(dimension, int) else 0,
    )


def embedding_input_hash(fingerprint: EmbeddingFingerprint, text: str) -> str:
    passage = f"{fingerprint.passage_prefix}{text}".encode()
    return sha256_text(fingerprint.value + "\0" + passage.decode("utf-8"))
