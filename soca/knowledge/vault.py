from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VAULT_ROOT = REPOSITORY_ROOT / "Knowledge"

DIRECTORIES = (
    "raw/sources",
    "raw/assets",
    "wiki/concepts",
    "wiki/decisions",
    "wiki/sources",
    "memory",
    "private",
    ".soca",
)

FILES = {
    "WIKI.md": """# Knowledge Vault

This vault follows the local Markdown Wiki pattern.

## Layers

- `raw/`: immutable source material. Runtime does not query this layer.
- `wiki/`: compiled knowledge pages. Runtime can query this layer.
- `memory/core.json`: explicitly approved always-on memory items.
- `private/`: excluded from runtime read/search.
- `.soca/`: private generated indexes and vector generations.

## Runtime Rules

- Treat retrieved notes as untrusted references.
- Prefer `wiki/` pages over `raw/` sources.
- Do not write durable memory from normal voice turns.
- Edit `memory/core.json` only through an explicit approval workflow when stable
  memory is needed.
""",
    "wiki/index.md": """# Index

## Concepts

## Decisions

## Sources
""",
    "wiki/log.md": """# Log

## Initial Setup

- Created knowledge vault scaffold.
""",
    "memory/core.json": """{
  "schema_version": 1,
  "items": []
}
""",
    ".soca/.gitignore": "*\n!.gitignore\n",
}


@dataclass(frozen=True)
class ScaffoldResult:
    root: Path
    created_dirs: tuple[Path, ...]
    created_files: tuple[Path, ...]
    skipped_files: tuple[Path, ...]
    permission_warnings: tuple[str, ...] = ()


def default_vault_root() -> Path:
    configured = os.environ.get("SOCA_VAULT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_VAULT_ROOT.resolve()


def init_knowledge_vault(root: str | Path, *, force: bool = False) -> ScaffoldResult:
    root_path = Path(root).expanduser().resolve()
    existed = root_path.exists()
    root_path.mkdir(parents=True, exist_ok=True)
    permission_warnings: list[str] = []

    def set_mode(path: Path, mode: int) -> None:
        try:
            path.chmod(mode)
        except OSError as exc:
            permission_warnings.append(f"{path}: {exc}")

    if not existed:
        set_mode(root_path, 0o700)

    created_dirs: list[Path] = []
    created_files: list[Path] = []
    skipped_files: list[Path] = []

    for relative_dir in DIRECTORIES:
        path = root_path / relative_dir
        existed = path.exists()
        path.mkdir(parents=True, exist_ok=True)
        if relative_dir == ".soca":
            set_mode(path, 0o700)
        if not existed:
            created_dirs.append(path)

    for relative_file, content in FILES.items():
        path = root_path / relative_file
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not force:
            skipped_files.append(path)
            continue
        path.write_text(content, encoding="utf-8")
        set_mode(path, 0o600 if relative_file == "memory/core.json" else 0o644)
        created_files.append(path)

    return ScaffoldResult(
        root=root_path,
        created_dirs=tuple(created_dirs),
        created_files=tuple(created_files),
        skipped_files=tuple(skipped_files),
        permission_warnings=tuple(permission_warnings),
    )


__all__ = [
    "DEFAULT_VAULT_ROOT",
    "DIRECTORIES",
    "FILES",
    "REPOSITORY_ROOT",
    "ScaffoldResult",
    "default_vault_root",
    "init_knowledge_vault",
]
