from __future__ import annotations

from pathlib import Path

from soca.knowledge.index.persistence import default_index_home
from soca.knowledge.vault import init_knowledge_vault


def test_init_knowledge_vault_creates_runtime_layout(tmp_path: Path) -> None:
    vault = tmp_path / "Knowledge"

    result = init_knowledge_vault(vault)

    assert result.root == vault.resolve()
    assert (vault / "wiki" / "index.md").is_file()
    assert (vault / "memory" / "core.json").is_file()
    assert (vault / ".soca" / ".gitignore").is_file()
    assert default_index_home(vault) == vault.resolve() / ".soca" / "knowledge_index"


def test_init_knowledge_vault_does_not_overwrite_existing_notes(tmp_path: Path) -> None:
    vault = tmp_path / "Knowledge"
    vault.mkdir()
    wiki = vault / "WIKI.md"
    wiki.write_text("user content\n", encoding="utf-8")

    result = init_knowledge_vault(vault)

    assert wiki.read_text(encoding="utf-8") == "user content\n"
    assert wiki in result.skipped_files


def test_init_knowledge_vault_reports_permission_warnings_without_aborting(
    monkeypatch, tmp_path: Path
) -> None:
    def deny_chmod(self: Path, mode: int) -> None:
        raise PermissionError(f"chmod denied for {self}")

    monkeypatch.setattr(Path, "chmod", deny_chmod)

    vault = tmp_path / "Knowledge"
    result = init_knowledge_vault(vault)

    assert (vault / "wiki" / "index.md").is_file()
    assert result.permission_warnings
