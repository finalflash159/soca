"""Run the production retriever on a pinned public VN-MTEB retrieval slice."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from eval.result_io import make_eval_artifact_metadata, write_json_artifact
from eval.retrieval_bakeoff import (
    DEFAULT_DATA_ROOT,
    RankerFactory,
    _load_dataset,
    evaluate_candidate,
    parse_candidate,
    production_chunks,
)
from soca.knowledge.retrievers.dense import production_embedding_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "arguana-vn"
DATASET_REPO = "GreenNode/arguana-vn"
DATASET_REVISION = "137122e4a56c03399d31bce35a045f0034242a4c"
EXPECTED_UPSTREAM_QUERY_COUNT = 1295
EXPECTED_EVALUABLE_QUERY_COUNT = 1290
EXPECTED_EXCLUDED_QRELS = 5
PRODUCTION_CANDIDATE = "hybrid_linear:bm25:aiteamvn_v2:0.75"
DEFAULT_SOURCE_LOCK = REPO_ROOT / "data" / "benchmarks" / "retrieval" / "sources.lock.json"
DEFAULT_PROVISION_MANIFEST = (
    REPO_ROOT / "data" / "benchmarks" / "retrieval" / "provisioned-manifest.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "eval" / "results" / "vn_mteb_arguana.json"
DEFAULT_RAW_OUTPUT = REPO_ROOT / "artifacts" / "local" / "vn_mteb_arguana_raw.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def validate_provisioned_dataset(
    *,
    source_lock: Path,
    provision_manifest: Path,
    data_root: Path,
) -> dict[str, Any]:
    lock = _mapping(json.loads(source_lock.read_text(encoding="utf-8")), label="source lock")
    lock_sources = _mapping(lock.get("sources"), label="source lock sources")
    source = _mapping(lock_sources.get(DATASET_NAME), label=DATASET_NAME)
    if source.get("source") != DATASET_REPO or source.get("revision") != DATASET_REVISION:
        raise ValueError("VN-MTEB source does not match the pinned repository revision")
    if source.get("license_verified") is not True:
        raise ValueError("VN-MTEB dataset license is not verified")
    expected_files = source.get("files")
    if (
        not isinstance(expected_files, list)
        or not expected_files
        or not all(isinstance(item, str) and item for item in expected_files)
    ):
        raise ValueError("VN-MTEB source lock has no files")
    provision = _mapping(
        json.loads(provision_manifest.read_text(encoding="utf-8")),
        label="provision manifest",
    )
    provision_sources = _mapping(provision.get("sources"), label="provision sources")
    receipt = _mapping(provision_sources.get(DATASET_NAME), label="provision receipt")
    if receipt.get("revision") != DATASET_REVISION:
        raise ValueError("provisioned VN-MTEB revision mismatch")
    receipt_files = _mapping(receipt.get("files"), label="provisioned files")
    source_root = data_root / DATASET_NAME
    total_bytes = 0
    for relative in expected_files:
        assert isinstance(relative, str)
        expected = _mapping(receipt_files.get(relative), label=f"receipt {relative}")
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if expected.get("bytes") != path.stat().st_size or expected.get("sha256") != _sha256(path):
            raise ValueError(f"VN-MTEB checksum mismatch: {relative}")
        total_bytes += path.stat().st_size
    return {
        "repo": DATASET_REPO,
        "revision": DATASET_REVISION,
        "license_verified": True,
        "file_count": len(expected_files),
        "total_bytes": total_bytes,
        "source_lock_sha256": _sha256(source_lock),
        "provision_manifest_sha256": _sha256(provision_manifest),
        "paths": [str(source_root / relative) for relative in expected_files],
    }


def sanitize_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "measurements"}


def run_benchmark(
    *,
    data_root: Path,
    source_lock: Path,
    provision_manifest: Path,
    raw_output: Path,
    output: Path,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    dataset_identity = validate_provisioned_dataset(
        source_lock=source_lock,
        provision_manifest=provision_manifest,
        data_root=data_root,
    )
    dataset = _load_dataset(DATASET_NAME, data_root)
    chunks, document_paths = production_chunks(dataset)
    spec = parse_candidate(PRODUCTION_CANDIDATE)
    result = evaluate_candidate(
        dataset,
        spec,
        chunks=chunks,
        document_paths=document_paths,
        ranker_factory=RankerFactory(chunks, batch_size=batch_size),
        query_limit=None,
        seed=seed,
    )
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    write_json_artifact(raw_output, result)
    reasons: list[str] = []
    if result.get("status") != "ok":
        reasons.append("candidate_failed")
    if result.get("query_count") != EXPECTED_EVALUABLE_QUERY_COUNT:
        reasons.append("incomplete_query_set")
    if result.get("excluded_upstream_qrels") != EXPECTED_EXCLUDED_QRELS:
        reasons.append("upstream_qrel_inventory_changed")
    data_files = tuple(Path(path) for path in dataset_identity.pop("paths"))
    report = {
        "schema_version": "soca-vn-mteb-retrieval-v1",
        "artifact": make_eval_artifact_metadata(
            suite="vn_mteb_arguana_retrieval",
            run_type="benchmark",
            data_files=(source_lock, provision_manifest, *data_files),
            config={
                "dataset": DATASET_NAME,
                "dataset_revision": DATASET_REVISION,
                "candidate": PRODUCTION_CANDIDATE,
                "seed": seed,
                "batch_size": batch_size,
                "query_limit": None,
                "model": asdict(production_embedding_fingerprint()),
            },
            ignored_untracked_paths=(raw_output, output),
        ).to_dict(),
        "dataset": dataset_identity,
        "result": sanitize_result(result),
        "gate": {
            "passed": not reasons,
            "reasons": reasons,
            "upstream_query_count": EXPECTED_UPSTREAM_QUERY_COUNT,
            "expected_evaluable_query_count": EXPECTED_EVALUABLE_QUERY_COUNT,
            "expected_excluded_upstream_qrels": EXPECTED_EXCLUDED_QRELS,
        },
    }
    write_json_artifact(output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_SOURCE_LOCK)
    parser.add_argument("--provision-manifest", type=Path, default=DEFAULT_PROVISION_MANIFEST)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20_260_814)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_benchmark(
            data_root=args.data_root,
            source_lock=args.source_lock,
            provision_manifest=args.provision_manifest,
            raw_output=args.raw_output,
            output=args.output,
            seed=args.seed,
            batch_size=args.batch_size,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark boundary is typed below
        report = {
            "schema_version": "soca-vn-mteb-retrieval-v1",
            "status": "blocked",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "gate": {"passed": False, "reasons": ["benchmark_failed"]},
        }
        write_json_artifact(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(not report["gate"]["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
