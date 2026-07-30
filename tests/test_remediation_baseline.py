from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.baseline_cases import (
    assert_no_family_leakage,
    assert_quality_eligible,
    load_cases,
)
from eval.remediation_eval import (
    DEFAULT_AUDIT_INVENTORY,
    build_dataset_manifest,
    validate_audit_inventory,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_CASES = REPO_ROOT / "eval" / "prompts" / "remediation_workflow_vi.jsonl"
RAG_CASES = REPO_ROOT / "eval" / "prompts" / "remediation_rag_vi.jsonl"


def test_remediation_cases_are_unique_and_not_demo_derived() -> None:
    workflow = load_cases(WORKFLOW_CASES, quality_suite=True)
    rag = load_cases(RAG_CASES, quality_suite=True)

    assert len(workflow) >= 8
    assert len(rag) >= 6
    assert {case.split for case in workflow} == {"test", "challenge"}
    assert {case.split for case in rag} == {"test", "challenge"}
    assert {case.suite_kind for case in (*workflow, *rag)} == {
        "regression",
        "capability",
    }
    assert_no_family_leakage((*workflow, *rag))
    assert_quality_eligible(workflow)
    assert_quality_eligible(rag)


def test_loader_rejects_demo_cases_in_quality_suite(tmp_path: Path) -> None:
    path = tmp_path / "demo.jsonl"
    path.write_text(
        '{"id":"demo","dataset_class":"demo_smoke","split":"test",'
        '"suite_kind":"regression","family":"demo",'
        '"category":"smoke","turns":["x"],"expected_terminal":"safe_failure",'
        '"audit_items":["P0-1"],"provenance":"demo_smoke"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid remediation case"):
        load_cases(path, quality_suite=True)


def test_loader_rejects_explicit_demo_derivative(tmp_path: Path) -> None:
    path = tmp_path / "derived.jsonl"
    path.write_text(
        '{"id":"derived","dataset_class":"sanitized_benchmark","split":"test",'
        '"suite_kind":"regression","family":"derived",'
        '"category":"bad","turns":["x"],"expected_terminal":"safe_failure",'
        '"audit_items":["P0-1"],'
        '"provenance":"derived_from_demo/knowledge_demo_vault"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="derived from demo"):
        load_cases(path, quality_suite=True)


@pytest.mark.parametrize("field", ["expected_sources", "expected_tools", "expected_citations"])
def test_loader_rejects_scalar_expectation_fields(tmp_path: Path, field: str) -> None:
    payload = {
        "id": "bad-shape",
        "suite_kind": "regression",
        "dataset_class": "sanitized_benchmark",
        "split": "test",
        "family": "malformed",
        "category": "malformed",
        "turns": ["x"],
        "expected_terminal": "safe_failure",
        "audit_items": ["P0-1"],
        "provenance": "independently authored",
        field: "knowledge",
    }
    path = tmp_path / "bad-shape.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be JSON lists"):
        load_cases(path, quality_suite=True)


def test_loader_rejects_family_leakage_across_splits(tmp_path: Path) -> None:
    common = {
        "suite_kind": "capability",
        "dataset_class": "public_screening",
        "family": "same-paraphrase-family",
        "category": "retrieval",
        "turns": ["x"],
        "expected_terminal": "achieved",
        "audit_items": ["P0-1"],
        "provenance": "public source",
    }
    path = tmp_path / "leak.jsonl"
    path.write_text(
        json.dumps({"id": "a", "split": "train", **common}) + "\n"
        + json.dumps({"id": "b", "split": "test", **common})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="paraphrase families"):
        assert_no_family_leakage(load_cases(path, quality_suite=True))


def test_audit_inventory_is_machine_readable_and_references_known_cases() -> None:
    cases = (
        *load_cases(WORKFLOW_CASES, quality_suite=True),
        *load_cases(RAG_CASES, quality_suite=True),
    )

    summary = validate_audit_inventory(DEFAULT_AUDIT_INVENTORY, cases)

    assert summary["item_count"] >= 50
    assert summary["expected_failure_count"] > 0
    assert summary["runtime_case_count"] == len(cases)


def test_manifest_contains_provenance_and_dataset_breakdown() -> None:
    manifest = build_dataset_manifest((WORKFLOW_CASES, RAG_CASES))

    assert manifest["schema_version"] == "soca-remediation-dataset-manifest-v1"
    assert manifest["case_count"] == 14
    assert manifest["artifact"]["suite"] == "remediation_baseline"
    assert manifest["suite_kinds"] == {"capability": 10, "regression": 4}
    assert manifest["family_count"] == 13
    assert all(dataset["dataset_classes"] for dataset in manifest["datasets"])
    assert all("sha256" in item for item in manifest["artifact"]["data_files"])
    assert manifest["corpora"][0]["file_count"] > 0
    assert manifest["audit_inventory"]["item_count"] >= 50
    assert manifest["artifact"]["source"]["state_digest"]
