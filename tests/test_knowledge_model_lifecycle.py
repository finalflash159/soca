from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from soca.knowledge.indexing.models import (
    _make_private,
    model_fingerprint,
    model_is_provisioned,
    model_spec,
)
from soca.knowledge.retrievers.dense import (
    AITEAMVN_V2_MODEL,
    AITEAMVN_V2_MODEL_SHA256,
    AITEAMVN_V2_TOKENIZER_SHA256,
)


def _write_model_layout(model_home: Path) -> Path:
    spec = model_spec("aiteamvn-v2")
    root = model_home / spec.cache_subdirectory
    root.mkdir(parents=True)
    (root / ".soca-model.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "key": spec.key,
                "repo_id": spec.model_id,
                "revision": spec.revision,
                "model_sha256": AITEAMVN_V2_MODEL_SHA256,
                "tokenizer_sha256": AITEAMVN_V2_TOKENIZER_SHA256,
            }
        ),
        encoding="utf-8",
    )
    (root / "model.safetensors").write_bytes(b"not read by lightweight status")
    (root / "tokenizer.json").write_bytes(b"not read by lightweight status")
    return root


def test_model_readiness_and_fingerprint_do_not_load_or_hash_weights(tmp_path: Path) -> None:
    _write_model_layout(tmp_path)

    assert model_is_provisioned("aiteamvn-v2", model_home=tmp_path) is True
    assert model_fingerprint("aiteamvn-v2").dimension == 1024


def test_active_retrieval_lock_matches_runtime_production_identity() -> None:
    lock_path = Path(__file__).resolve().parents[1] / "eval" / "retrieval_models.lock.json"
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    locked = payload["models"]["aiteamvn_v2"]

    assert locked["repo_id"] == AITEAMVN_V2_MODEL
    assert locked["revision"] == model_spec("aiteamvn-v2").revision
    assert locked["required_file_sha256"] == {
        "model.safetensors": AITEAMVN_V2_MODEL_SHA256,
        "tokenizer.json": AITEAMVN_V2_TOKENIZER_SHA256,
    }


def test_private_permission_walk_rejects_symlinks_without_touching_target(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    outside.chmod(0o644)
    (root / "escape").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        _make_private(root)

    assert stat.S_IMODE(outside.stat().st_mode) == 0o644
