from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from soca.knowledge.retrievers.dense import default_model_home

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = REPO_ROOT / "eval" / "retrieval_models.lock.json"


def _load_hf_token() -> None:
    if os.environ.get("HF_TOKEN"):
        return
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "HF_TOKEN":
            os.environ["HF_TOKEN"] = value.strip().strip("\"'")
            return


def _load_lock(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("retrieval model lock must use schema_version 1")
    models = payload.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("retrieval model lock requires models")
    for key, model in models.items():
        if not isinstance(model, dict):
            raise ValueError(f"{key}: model entry must be an object")
        revision = model.get("revision")
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            raise ValueError(f"{key}: revision must be an immutable commit")
    return models


def _make_private(root: Path) -> None:
    for directory in (root, *root.rglob("*")):
        if directory.is_dir():
            directory.chmod(0o700)
        elif directory.is_file():
            directory.chmod(0o600)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provision(key: str, model: dict[str, Any], *, model_home: Path) -> Path:
    repo_id = model["repo_id"]
    revision = model["revision"]
    if not isinstance(repo_id, str) or not isinstance(revision, str):
        raise ValueError(f"{key}: invalid model identity")
    target = model_home / "eval" / key
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    print(f"[{key}] {repo_id}@{revision}", flush=True)
    dependencies = model.get("remote_code_dependencies", {})
    if not isinstance(dependencies, dict):
        raise ValueError(f"{key}: remote_code_dependencies must be an object")
    for dependency_id, dependency_revision in dependencies.items():
        if not isinstance(dependency_id, str) or not isinstance(dependency_revision, str):
            raise ValueError(f"{key}: invalid remote code dependency")
        dependency_path = Path(
            snapshot_download(repo_id=dependency_id, revision="main")
        )
        if dependency_path.name != dependency_revision:
            raise RuntimeError(
                f"{key}: remote code {dependency_id} resolved to "
                f"{dependency_path.name}, expected {dependency_revision}"
            )
    required_hashes = model.get("required_file_sha256", {})
    if not isinstance(required_hashes, dict):
        raise ValueError(f"{key}: required_file_sha256 must be an object")
    ignore_patterns = [
        "*.h5",
        "*.msgpack",
        "onnx/**",
        "openvino/**",
        "tf_model.*",
    ]
    if not any(filename.endswith(".bin") for filename in required_hashes):
        ignore_patterns.append("*.bin")
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=target,
        ignore_patterns=tuple(ignore_patterns),
    )
    for filename, expected_sha256 in required_hashes.items():
        if not isinstance(filename, str) or not isinstance(expected_sha256, str):
            raise ValueError(f"{key}: invalid required file checksum")
        actual_sha256 = _sha256(target / filename)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"{key}: checksum mismatch for {filename}: "
                f"{actual_sha256} != {expected_sha256}"
            )
    manifest = {
        "schema_version": 1,
        "key": key,
        "repo_id": repo_id,
        "revision": revision,
        "role": model.get("role"),
        "license": model.get("license"),
        "trust_remote_code": model.get("trust_remote_code") is True,
    }
    (target / ".soca-model.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _make_private(target)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Provision checksum-pinned retrieval benchmark models."
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    models = _load_lock(args.lock)
    selected = set(args.model)
    if not args.all and not selected:
        raise SystemExit("pass --all or at least one --model")
    unknown = selected - set(models)
    if unknown:
        raise SystemExit("unknown models: " + ", ".join(sorted(unknown)))
    _load_hf_token()
    for key, model in models.items():
        if args.all or key in selected:
            print(provision(key, model, model_home=default_model_home()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
