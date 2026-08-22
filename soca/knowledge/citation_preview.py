"""Bounded, vault-confined verification for a structured citation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from soca.knowledge.markdown_vault import (
    MarkdownVaultFileTooLargeError,
    MarkdownVaultKnowledgeSource,
)

CitationPreviewStatus = Literal["current", "changed", "unverified", "missing", "unavailable"]
MAX_CITATION_PREVIEW_LINES = 120


def citation_fingerprint(text: str) -> str:
    """Return the persisted evidence revision for one decoded Markdown source."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CitationPreview:
    """A current, bounded view of a citation without exposing a filesystem path."""

    status: CitationPreviewStatus
    title: str | None
    line_start: int | None
    line_end: int | None
    passage: str | None
    fingerprint: str | None
    error_code: str | None = None

    def as_protocol(
        self,
        *,
        request_id: str,
        path: str,
        source: str = "knowledge",
    ) -> dict[str, str | int | None]:
        return {
            "event": "citation_preview",
            "request_id": request_id,
            "path": path,
            "source": source,
            "status": self.status,
            "title": self.title,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "passage": self.passage,
            "fingerprint": self.fingerprint,
            "error_code": self.error_code,
        }


def _unavailable(code: str) -> CitationPreview:
    return CitationPreview(
        status="unavailable",
        title=None,
        line_start=None,
        line_end=None,
        passage=None,
        fingerprint=None,
        error_code=code,
    )


def preview_vault_citation(
    vault_root: Path,
    *,
    path: str,
    line_start: int | None,
    line_end: int | None,
    expected_fingerprint: str | None,
) -> CitationPreview:
    """Read exactly the cited Markdown lines and classify their current revision.

    Citation paths are always relative to the configured knowledge vault.  The
    same vault reader that retrieval uses rejects paths outside that root,
    excluded directories and non-Markdown files.  The preview is deliberately
    bounded by the reader's file-size policy and by line count: an evidence
    check must not turn an arbitrary persisted path into an unbounded engine
    read.
    """

    if (
        isinstance(line_start, bool)
        or isinstance(line_end, bool)
        or not isinstance(line_start, int)
        or not isinstance(line_end, int)
        or line_start < 1
        or line_end < line_start
    ):
        return _unavailable("citation_location_unavailable")
    if line_end - line_start + 1 > MAX_CITATION_PREVIEW_LINES:
        return _unavailable("citation_range_too_large")
    if not vault_root.is_dir():
        return _unavailable("knowledge_vault_unavailable")

    try:
        source = MarkdownVaultKnowledgeSource(vault_root)
        document = source.read(path)
    except FileNotFoundError:
        return CitationPreview(
            status="missing",
            title=None,
            line_start=line_start,
            line_end=line_end,
            passage=None,
            fingerprint=None,
            error_code="source_missing",
        )
    except MarkdownVaultFileTooLargeError:
        return _unavailable("source_too_large")
    except PermissionError:
        return _unavailable("source_permission_denied")
    except (OSError, UnicodeError, ValueError):
        return _unavailable("source_unavailable")

    lines = document.text.splitlines()
    fingerprint = citation_fingerprint(document.text)
    status: CitationPreviewStatus
    if expected_fingerprint is None:
        status = "unverified"
    else:
        status = "current" if expected_fingerprint == fingerprint else "changed"

    if line_end > len(lines):
        return CitationPreview(
            status="changed" if status == "current" else status,
            title=document.title,
            line_start=line_start,
            line_end=line_end,
            passage=None,
            fingerprint=fingerprint,
            error_code="citation_location_unavailable",
        )

    passage = "\n".join(lines[line_start - 1 : line_end])
    return CitationPreview(
        status=status,
        title=document.title,
        line_start=line_start,
        line_end=line_end,
        passage=passage,
        fingerprint=fingerprint,
    )
