from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "eval" / "fixtures" / "real_rag_vault"


def test_public_retrieval_fixture_has_attributed_reference_slice() -> None:
    manifest = json.loads((FIXTURE / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    sources = manifest["sources"]

    assert manifest["case_count"] == 1193
    public_sources = [item for item in sources if item["path"].startswith("wiki/xquad_vi/")]
    assert len(public_sources) == 48

    for item in public_sources:
        path = FIXTURE / item["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
