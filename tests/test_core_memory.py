from pathlib import Path

import pytest

from soca.memory import CoreMemoryStore


def write_core(root: Path, value: str = "Người dùng thích tiếng Việt") -> None:
    memory = root / "memory"
    memory.mkdir(parents=True, exist_ok=True)
    (memory / "core.json").write_text(
        '{"schema_version":1,"items":[{"id":"language","value":'
        f'"{value}","approved_at":"2026-01-01T00:00:00Z",'
        '"sensitivity":"normal","updated_at":"2026-01-01T00:00:00Z",'
        '"provenance":"user"}]}',
        encoding="utf-8",
    )


def test_reads_approved_core_memory(tmp_path: Path) -> None:
    write_core(tmp_path)

    assert CoreMemoryStore(tmp_path).read_core() == "- [language] Người dùng thích tiếng Việt"


def test_missing_core_is_empty(tmp_path: Path) -> None:
    assert CoreMemoryStore(tmp_path).read_core() == ""


def test_core_memory_enforces_character_budget(tmp_path: Path) -> None:
    write_core(tmp_path, "a" * 200)

    core_text = CoreMemoryStore(tmp_path, max_chars=50).read_core()

    assert len(core_text) <= 50
    assert core_text.endswith("...")


def test_core_memory_rejects_invalid_schema(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "core.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="Expecting property name"):
        CoreMemoryStore(tmp_path).items()


def test_core_memory_rejects_symlink(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = tmp_path / "outside.json"
    target.write_text('{"schema_version":1,"items":[]}', encoding="utf-8")
    (memory / "core.json").symlink_to(target)

    with pytest.raises(ValueError, match="regular file"):
        CoreMemoryStore(tmp_path).items()
