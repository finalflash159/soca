from __future__ import annotations

from pathlib import Path

DEFAULT_PROFILE_PATH = Path("memory/profile.md")


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


class MarkdownLongTermMemory:
    """Read-only long-term profile memory from a local Markdown vault."""

    def __init__(
        self,
        vault_root: str | Path,
        profile_path: str | Path = DEFAULT_PROFILE_PATH,
        max_chars: int = 1000,
    ) -> None:
        self.root = Path(vault_root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Memory vault not found: {self.root}")

        self.profile_path = Path(profile_path)
        if self.profile_path.is_absolute():
            raise ValueError("profile_path must be relative to vault_root")
        if self.profile_path.suffix.lower() != ".md":
            raise ValueError("profile_path must point to a markdown file")
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than 0")

        self.max_chars = max_chars

    def read_profile(self) -> str:
        path = self._resolve_profile_path()
        if not path.exists():
            return ""
        if not path.is_file():
            return ""

        return _truncate(path.read_text(encoding="utf-8"), self.max_chars)

    def _resolve_profile_path(self) -> Path:
        candidate = (self.root / self.profile_path).resolve()

        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("profile_path is outside of the memory vault") from exc

        return candidate
