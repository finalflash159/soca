"""Shared text normalization for the code-switch recording and scoring scripts.

Both `english_indices` (used at recording time) and alignment (used at
scoring time) must tokenize identically, or English-word indices will point
at the wrong token and silently corrupt CS-WER.
"""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

# English terms to score separately. Matched by normalized token, case-insensitive.
EN_TERMS: frozenset[str] = frozenset(
    {
        "github", "repo", "pytorch", "tensorflow", "docker", "compose", "log", "postgresql",
        "onnxruntime", "onnx", "typescript", "component", "api", "json", "xml",
        "readme", "root", "kubernetes", "local", "cache", "redis", "nginx", "proxy",
        "port", "merge", "pull", "request", "branch", "main", "pytest", "test",
        "fail", "embedding", "model", "batch", "size", "inference", "transformer",
        "layer", "build", "image", "push", "registry", "config", "file", "index",
        "database", "query", "function", "memory", "leak", "commit", "history",
        "setup", "ci", "cd", "actions", "latency", "millisecond", "load", "gpu",
        "cpu", "parse", "response", "unit", "class", "refactor", "module",
        "authentication", "token", "refresh", "websocket", "disconnect", "optimize",
        "fastapi", "flask", "backend", "gigabyte", "migration", "level", "debug",
        "info", "dependency", "conflict", "version", "export", "gguf", "llama",
        "cpp", "python", "quantize", "int", "vector", "faiss", "stream", "client",
        "chunk", "train", "deploy", "convert",
    }
)


def normalize(text: str) -> str:
    """NFC + lowercase + strip punctuation + collapse whitespace.

    NFC is required: some sources return Vietnamese diacritics in NFD form,
    which breaks string comparison silently (seen before in answer_validation).
    """
    text = unicodedata.normalize("NFC", text).lower()
    stripped = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    return " ".join(stripped.split())


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def english_indices(reference: str) -> list[int]:
    """Token positions of English words in the reference sentence."""
    return [i for i, word in enumerate(tokens(reference)) if word in EN_TERMS]


def manifest_fingerprint(path: Path) -> str:
    """SHA-256 of the manifest file, so prediction/score artifacts can record
    exactly which recording session they were run against."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
