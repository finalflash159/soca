from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from eval.eval_vn_mteb import (
    DATASET_REVISION,
    EXPECTED_EVALUABLE_QUERY_COUNT,
    EXPECTED_EXCLUDED_QRELS,
    EXPECTED_UPSTREAM_QUERY_COUNT,
    PRODUCTION_CANDIDATE,
    sanitize_result,
    validate_provisioned_dataset,
)


def test_dataset_validation_requires_pinned_revision_and_file_hashes(tmp_path: Path) -> None:
    data_root = tmp_path / "public"
    source_root = data_root / "arguana-vn"
    file_path = source_root / "queries" / "test.parquet"
    file_path.parent.mkdir(parents=True)
    file_path.write_bytes(b"query-data")
    digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
    lock = tmp_path / "sources.lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": {
                    "arguana-vn": {
                        "source": "GreenNode/arguana-vn",
                        "revision": DATASET_REVISION,
                        "destination": "public/arguana-vn",
                        "license_verified": True,
                        "files": ["queries/test.parquet"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    provision = tmp_path / "provision.json"
    provision.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": {
                    "arguana-vn": {
                        "revision": DATASET_REVISION,
                        "files": {
                            "queries/test.parquet": {
                                "bytes": len(b"query-data"),
                                "sha256": digest,
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    identity = validate_provisioned_dataset(
        source_lock=lock,
        provision_manifest=provision,
        data_root=data_root,
    )

    assert identity["revision"] == DATASET_REVISION
    assert identity["file_count"] == 1

    file_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        validate_provisioned_dataset(
            source_lock=lock,
            provision_manifest=provision,
            data_root=data_root,
        )


def test_public_result_omits_query_and_document_receipts() -> None:
    public = sanitize_result(
        {
            "status": "ok",
            "dataset": "arguana-vn",
            "candidate": PRODUCTION_CANDIDATE,
            "query_count": 1295,
            "metrics": {"recall_at_5": 0.75},
            "measurements": [
                {
                    "query_id": "private-ish-id",
                    "retrieved_documents": ["doc"],
                    "latency_ms": 1.0,
                }
            ],
        }
    )

    assert "measurements" not in public
    assert public["metrics"] == {"recall_at_5": 0.75}


def test_arguana_inventory_distinguishes_upstream_from_evaluable_qrels() -> None:
    assert EXPECTED_UPSTREAM_QUERY_COUNT == 1295
    assert EXPECTED_EVALUABLE_QUERY_COUNT == 1290
    assert EXPECTED_EXCLUDED_QRELS == 5
    assert EXPECTED_UPSTREAM_QUERY_COUNT - EXPECTED_EXCLUDED_QRELS == (
        EXPECTED_EVALUABLE_QUERY_COUNT
    )
