from __future__ import annotations

import hashlib
import json
from pathlib import Path

from eval.eval_hybrid_retrieval import load_cases
from soca.knowledge.markdown_vault import MarkdownVaultKnowledgeSource

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "eval" / "fixtures" / "knowledge_demo_vault"
QRELS = ROOT / "eval" / "prompts" / "knowledge_demo_vi.jsonl"


def _payloads() -> dict[str, dict[str, object]]:
    payloads: dict[str, dict[str, object]] = {}
    for line in QRELS.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        payloads[str(payload["id"])] = payload
    return payloads


def test_demo_fixture_manifest_and_slice_contract_are_traceable() -> None:
    manifest = json.loads((FIXTURE / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["fixture"] == "knowledge_demo_v1"
    assert manifest["case_count"] == 8
    assert manifest["slices"] == {"learning_notes": 4, "life_vault": 4}

    for source in manifest["sources"]:
        path = FIXTURE / source["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]

    cases = load_cases(QRELS)
    assert len(cases) == 8
    assert {case.slice_name for case in cases} == {"learning_notes", "life_vault"}
    assert all(path.startswith("wiki/") for case in cases for path in case.relevant_paths)


def test_demo_fixture_retrieves_expected_note_at_rank_one() -> None:
    source = MarkdownVaultKnowledgeSource(FIXTURE, include_globs=("wiki/**/*.md",))
    payloads = _payloads()

    for case in load_cases(QRELS):
        hits = source.search(case.query, limit=3)
        assert hits, case.case_id
        assert hits[0].document.path in case.relevant_paths, case.case_id

        expected_contains = payloads[case.case_id].get("expected_contains", ())
        text = " ".join(hits[0].document.text.split())
        assert all(expected in text for expected in expected_contains), case.case_id


def test_demo_fixture_marks_synthetic_finance_and_health_safety_boundaries() -> None:
    finance = " ".join(
        (FIXTURE / "wiki/life/finance/food-budget-2026-07.md").read_text(encoding="utf-8").split()
    )
    health = " ".join(
        (FIXTURE / "wiki/life/health/safety-boundaries.md").read_text(encoding="utf-8").split()
    )

    assert "synthetic_demo" in finance
    assert "không phải sổ tài chính cá nhân" in finance
    assert "không thay thế bác sĩ" in health
    assert "không được chẩn đoán" in health
