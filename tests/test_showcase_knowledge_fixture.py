from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "eval" / "fixtures" / "knowledge_vault"


def test_showcase_fixture_has_attributed_public_reference_slice() -> None:
    manifest = json.loads((FIXTURE / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    sources = manifest["sources"]

    assert manifest["dataset"] == "XQuAD Vietnamese"
    assert manifest["license"] == "CC BY-SA 4.0"
    assert manifest["document_count"] == 48
    assert len(sources) == manifest["document_count"]
    assert all(item["path"].startswith("wiki/xquad_vi/") for item in sources)

    for item in sources:
        path = FIXTURE / item["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]

    provenance = (FIXTURE / "wiki" / "sources" / "xquad-vietnamese.md").read_text(
        encoding="utf-8"
    )
    assert "XQuAD Vietnamese" in provenance
    assert "CC BY-SA 4.0" in provenance
