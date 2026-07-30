from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eval.baseline_cases import (
    RemediationCase,
    assert_no_family_leakage,
    assert_quality_eligible,
    load_cases,
)
from eval.result_io import make_eval_artifact_metadata, write_json_artifact

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = (
    REPO_ROOT / "eval" / "prompts" / "remediation_workflow_vi.jsonl",
    REPO_ROOT / "eval" / "prompts" / "remediation_rag_vi.jsonl",
)
DEFAULT_CORPUS_ROOT = REPO_ROOT / "eval" / "fixtures" / "real_rag_vault"
DEFAULT_AUDIT_INVENTORY = REPO_ROOT / "eval" / "remediation_audit_inventory.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_corpus(corpus_root: Path) -> tuple[Path, ...]:
    manifest_path = corpus_root / "SOURCE_MANIFEST.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("sources")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{manifest_path}: missing non-empty sources")
    for entry in entries:
        relative_path = entry.get("path") if isinstance(entry, dict) else None
        expected_hash = entry.get("sha256") if isinstance(entry, dict) else None
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            raise ValueError(f"{manifest_path}: invalid source entry")
        source_path = corpus_root / relative_path
        if not source_path.is_file() or _sha256(source_path) != expected_hash:
            raise ValueError(f"{manifest_path}: source hash mismatch for {relative_path}")
    files = tuple(sorted(path for path in corpus_root.rglob("*") if path.is_file()))
    return files


def validate_audit_inventory(
    inventory_path: Path,
    cases: Sequence[RemediationCase],
) -> dict[str, Any]:
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "soca-remediation-audit-inventory-v1":
        raise ValueError(f"{inventory_path}: unsupported schema")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{inventory_path}: items must be a non-empty list")
    case_ids = {case.case_id for case in cases}
    item_ids: set[str] = set()
    referenced_cases: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"{inventory_path}: audit item must be an object")
        item_id = item.get("id")
        status = item.get("status")
        gate = item.get("gate")
        evidence = item.get("evidence")
        item_case_ids = item.get("case_ids")
        current_behavior = item.get("current_behavior")
        if (
            not isinstance(item_id, str)
            or not item_id
            or item_id in item_ids
            or status not in {"characterized", "expected_failure"}
            or not isinstance(gate, str)
            or not gate
            or not isinstance(evidence, list)
            or not evidence
            or not all(isinstance(value, str) and value for value in evidence)
            or not isinstance(item_case_ids, list)
            or not all(isinstance(value, str) and value for value in item_case_ids)
            or not isinstance(current_behavior, str)
            or not current_behavior
        ):
            raise ValueError(f"{inventory_path}: invalid audit item")
        unknown_cases = set(item_case_ids) - case_ids
        if unknown_cases:
            raise ValueError(
                f"{inventory_path}: unknown case ids for {item_id}: "
                + ", ".join(sorted(unknown_cases))
            )
        item_ids.add(item_id)
        referenced_cases.update(item_case_ids)
    case_audit_items = {item for case in cases for item in case.audit_items}
    unknown_audit_items = case_audit_items - item_ids
    if unknown_audit_items:
        raise ValueError(
            f"{inventory_path}: cases reference unknown audit items: "
            + ", ".join(sorted(unknown_audit_items))
        )
    return {
        "path": str(inventory_path),
        "sha256": _sha256(inventory_path),
        "item_count": len(items),
        "characterized_count": sum(item["status"] == "characterized" for item in items),
        "expected_failure_count": sum(item["status"] == "expected_failure" for item in items),
        "runtime_case_count": len(referenced_cases),
    }


def build_dataset_manifest(
    paths: Sequence[Path],
    *,
    corpus_roots: Sequence[Path] | None = None,
    audit_inventory: Path = DEFAULT_AUDIT_INVENTORY,
    ignored_untracked_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one dataset is required")
    all_cases: list[RemediationCase] = []
    datasets: list[dict[str, Any]] = []
    for path in paths:
        cases = load_cases(path, quality_suite=True)
        assert_quality_eligible(cases)
        datasets.append(
            {
                "path": str(path),
                "case_count": len(cases),
                "categories": dict(sorted(Counter(case.category for case in cases).items())),
                "splits": dict(sorted(Counter(case.split for case in cases).items())),
                "suite_kinds": dict(sorted(Counter(case.suite_kind for case in cases).items())),
                "family_count": len({case.family for case in cases}),
                "dataset_classes": sorted({case.dataset_class for case in cases}),
            }
        )
        all_cases.extend(cases)

    case_ids = [case.case_id for case in all_cases]
    duplicates = sorted(case_id for case_id, count in Counter(case_ids).items() if count > 1)
    if duplicates:
        raise ValueError("duplicate case ids across datasets: " + ", ".join(duplicates))
    assert_no_family_leakage(tuple(all_cases))
    audit_description = validate_audit_inventory(audit_inventory, all_cases)

    resolved_corpus_roots = tuple(corpus_roots or (DEFAULT_CORPUS_ROOT,))
    corpus_files: list[Path] = []
    corpus_descriptions: list[dict[str, Any]] = []
    for corpus_root in resolved_corpus_roots:
        files = validate_corpus(corpus_root)
        corpus_files.extend(files)
        corpus_descriptions.append(
            {
                "root": str(corpus_root),
                "manifest": str(corpus_root / "SOURCE_MANIFEST.json"),
                "file_count": len(files),
            }
        )

    metadata = make_eval_artifact_metadata(
        suite="remediation_baseline",
        data_files=tuple(paths) + (audit_inventory,) + tuple(corpus_files),
        config={
            "quality_suite": True,
            "dataset_count": len(paths),
            "corpus_count": len(resolved_corpus_roots),
            "suite_kinds": dict(
                sorted(Counter(case.suite_kind for case in all_cases).items())
            ),
        },
        ignored_untracked_paths=ignored_untracked_paths,
    )
    return {
        "schema_version": "soca-remediation-dataset-manifest-v1",
        "suite": "remediation_baseline",
        "artifact": metadata.to_dict(),
        "case_count": len(all_cases),
        "suite_kinds": dict(sorted(Counter(case.suite_kind for case in all_cases).items())),
        "family_count": len({case.family for case in all_cases}),
        "datasets": datasets,
        "corpora": corpus_descriptions,
        "audit_inventory": audit_description,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=Path, dest="datasets")
    parser.add_argument("--corpus", action="append", type=Path, dest="corpora")
    parser.add_argument("--audit-inventory", type=Path, default=DEFAULT_AUDIT_INVENTORY)
    parser.add_argument("--ignore-source-path", action="append", type=Path, default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "eval" / "results" / "remediation_baseline" / "manifest.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = tuple(args.datasets or DEFAULT_DATASETS)
    manifest = build_dataset_manifest(
        paths,
        corpus_roots=tuple(args.corpora or (DEFAULT_CORPUS_ROOT,)),
        audit_inventory=args.audit_inventory,
        ignored_untracked_paths=tuple(args.ignore_source_path),
    )
    write_json_artifact(args.output, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
