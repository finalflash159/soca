from pathlib import Path

import pytest

from soca.memory import MarkdownLongTermMemory


def test_reads_manual_profile_memory(tmp_path: Path):
    profile_path = tmp_path / "memory" / "profile.md"
    profile_path.parent.mkdir()
    profile_path.write_text(
        "# Profile Memory\n\n- Người dùng thích tiếng Việt.\n- Ưu tiên giải thích code kỹ.",
        encoding="utf-8",
    )

    memory = MarkdownLongTermMemory(tmp_path)

    profile = memory.read_profile()
    assert "Người dùng thích tiếng Việt" in profile
    assert "Ưu tiên giải thích code kỹ" in profile


def test_missing_profile_returns_empty_string(tmp_path: Path):
    memory = MarkdownLongTermMemory(tmp_path)

    assert memory.read_profile() == ""


def test_profile_memory_enforces_character_budget(tmp_path: Path):
    profile_path = tmp_path / "memory" / "profile.md"
    profile_path.parent.mkdir()
    profile_path.write_text("a" * 200, encoding="utf-8")

    memory = MarkdownLongTermMemory(tmp_path, max_chars=50)

    profile = memory.read_profile()
    assert len(profile) <= 50
    assert profile.endswith("...")


def test_normalizes_relative_profile_path_inside_vault(tmp_path: Path):
    memory = MarkdownLongTermMemory(tmp_path, profile_path="memory/../profile.md")

    assert memory.read_profile() == ""


def test_profile_path_must_stay_inside_vault(tmp_path: Path):
    memory = MarkdownLongTermMemory(tmp_path, profile_path="../profile.md")

    with pytest.raises(ValueError, match="outside"):
        memory.read_profile()


def test_rejects_absolute_profile_path(tmp_path: Path):
    with pytest.raises(ValueError, match="profile_path must be relative"):
        MarkdownLongTermMemory(tmp_path, profile_path=tmp_path / "profile.md")


def test_rejects_non_markdown_profile_path(tmp_path: Path):
    with pytest.raises(ValueError, match="markdown"):
        MarkdownLongTermMemory(tmp_path, profile_path="memory/profile.txt")
