from __future__ import annotations

import hashlib
import json
from pathlib import Path

from eval.eval_hybrid_retrieval import load_cases

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "eval" / "fixtures" / "real_rag_vault"


def test_real_fixture_manifest_hashes_and_slices() -> None:
    manifest = json.loads((FIXTURE / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["case_count"] == 1193
    for source in manifest["sources"]:
        path = FIXTURE / source["path"]
        assert path.is_file()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == source["sha256"]

    cases = load_cases(ROOT / "eval" / "prompts" / "real_rag_vi.jsonl")
    slices = {case.slice_name for case in cases}
    assert slices == {"learning_notes", "life_vault_project"}
    assert len(cases) == manifest["case_count"]
    assert all(path.startswith("wiki/") for case in cases for path in case.relevant_paths)
