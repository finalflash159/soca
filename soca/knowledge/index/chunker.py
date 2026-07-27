from __future__ import annotations

import hashlib
import re

from soca.knowledge.base import KnowledgeDocument
from soca.knowledge.index.models import MarkdownChunk

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
HEADING_RE = re.compile(r"^#{1,6}\s")


def _token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def _section_ranges(lines: list[str]) -> tuple[tuple[int, int], ...]:
    if not lines:
        return ()
    starts = [index for index, line in enumerate(lines) if HEADING_RE.match(line)]
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    return tuple(
        (start, starts[index + 1] - 1 if index + 1 < len(starts) else len(lines) - 1)
        for index, start in enumerate(starts)
    )


def _window_ranges(
    lines: list[str],
    start: int,
    end: int,
    *,
    target_tokens: int,
    overlap_lines: int,
) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    cursor = start
    while cursor <= end:
        window_end = cursor
        used_tokens = 0
        while window_end <= end:
            line_tokens = max(1, _token_count(lines[window_end]))
            if window_end > cursor and used_tokens + line_tokens > target_tokens:
                break
            used_tokens += line_tokens
            window_end += 1
        final_end = max(cursor, window_end - 1)
        ranges.append((cursor, final_end))
        if final_end >= end:
            break
        cursor = max(cursor + 1, final_end - overlap_lines + 1)
    return tuple(ranges)


def _chunk_id(path: str, line_start: int, line_end: int, text: str) -> str:
    payload = f"{path}\0{line_start}\0{line_end}\0{text}".encode()
    suffix = hashlib.sha256(payload).hexdigest()[:16]
    return f"{path}#{line_start}-{line_end}:{suffix}"


def chunk_markdown(
    document: KnowledgeDocument,
    *,
    target_tokens: int = 320,
    overlap_lines: int = 2,
) -> tuple[MarkdownChunk, ...]:
    if target_tokens < 32:
        raise ValueError("target_tokens must be at least 32")
    if overlap_lines < 0:
        raise ValueError("overlap_lines must be non-negative")

    lines = document.text.splitlines()
    chunks: list[MarkdownChunk] = []
    for section_start, section_end in _section_ranges(lines):
        for start, end in _window_ranges(
            lines,
            section_start,
            section_end,
            target_tokens=target_tokens,
            overlap_lines=overlap_lines,
        ):
            text = "\n".join(lines[start : end + 1]).strip()
            if not text:
                continue
            line_start = start + 1
            line_end = end + 1
            chunks.append(
                MarkdownChunk(
                    chunk_id=_chunk_id(document.path, line_start, line_end, text),
                    document_path=document.path,
                    title=document.title,
                    tags=document.tags,
                    text=text,
                    line_start=line_start,
                    line_end=line_end,
                )
            )
    return tuple(chunks)
