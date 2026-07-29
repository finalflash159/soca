from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.baseline_cases import assert_quality_eligible, load_cases
from eval.remediation_eval import build_dataset_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_CASES = REPO_ROOT / "eval" / "prompts" / "remediation_workflow_vi.jsonl"
RAG_CASES = REPO_ROOT / "eval" / "prompts" / "remediation_rag_vi.jsonl"


def test_remediation_cases_are_unique_and_not_demo_derived() -> None:
    workflow = load_cases(WORKFLOW_CASES, quality_suite=True)
    rag = load_cases(RAG_CASES, quality_suite=True)

    assert len(workflow) >= 4
    assert len(rag) >= 3
    assert {case.split for case in workflow} == {"test", "challenge"}
    assert {case.split for case in rag} == {"test", "challenge"}
    assert_quality_eligible(workflow)
    assert_quality_eligible(rag)


def test_loader_rejects_demo_cases_in_quality_suite(tmp_path: Path) -> None:
    path = tmp_path / "demo.jsonl"
    path.write_text(
        '{"id":"demo","dataset_class":"demo_smoke","split":"test",'
        '"category":"smoke","turns":["x"],"expected_terminal":"safe_failure",'
        '"provenance":"demo_smoke"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid remediation case"):
        load_cases(path, quality_suite=True)


def test_loader_rejects_explicit_demo_derivative(tmp_path: Path) -> None:
    path = tmp_path / "derived.jsonl"
    path.write_text(
        '{"id":"derived","dataset_class":"sanitized_benchmark","split":"test",'
        '"category":"bad","turns":["x"],"expected_terminal":"safe_failure",'
        '"provenance":"derived_from_demo/knowledge_demo_vault"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="derived from demo"):
        load_cases(path, quality_suite=True)


@pytest.mark.parametrize("field", ["expected_sources", "expected_tools", "expected_citations"])
def test_loader_rejects_scalar_expectation_fields(tmp_path: Path, field: str) -> None:
    payload = {
        "id": "bad-shape",
        "dataset_class": "sanitized_benchmark",
        "split": "test",
        "category": "malformed",
        "turns": ["x"],
        "expected_terminal": "safe_failure",
        "provenance": "independently authored",
        field: "knowledge",
    }
    path = tmp_path / "bad-shape.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be JSON lists"):
        load_cases(path, quality_suite=True)


def test_manifest_contains_provenance_and_dataset_breakdown() -> None:
    manifest = build_dataset_manifest((WORKFLOW_CASES, RAG_CASES))

    assert manifest["schema_version"] == "soca-remediation-dataset-manifest-v1"
    assert manifest["case_count"] == 7
    assert manifest["artifact"]["suite"] == "remediation_baseline"
    assert all(dataset["dataset_classes"] for dataset in manifest["datasets"])
    assert all("sha256" in item for item in manifest["artifact"]["data_files"])
    assert manifest["corpora"][0]["file_count"] > 0
    assert manifest["artifact"]["source"]["state_digest"]
