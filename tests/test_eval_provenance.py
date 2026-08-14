from __future__ import annotations

from pathlib import Path

from eval.provenance import file_set_identity, package_versions, run_provenance


def test_file_set_identity_changes_with_content_and_records_size(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"abc")
    second.write_bytes(b"defg")

    before = file_set_identity((first, second))
    second.write_bytes(b"changed")
    after = file_set_identity((first, second))

    assert before["file_count"] == 2
    assert before["total_bytes"] == 7
    assert before["content_sha256"] != after["content_sha256"]


def test_run_provenance_includes_runtime_environment() -> None:
    provenance = run_provenance(suite="fixture")

    assert provenance["suite"] == "fixture"
    assert provenance["environment"]["python"]
    assert provenance["environment"]["machine"]
    assert package_versions(("numpy",))["numpy"]
