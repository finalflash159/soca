from __future__ import annotations

import re

_SENTENCE_BOUNDARY_RE = re.compile(
    r"(?P<punct>[.!?\u2026]+)(?P<trailing>\s+|$)|(?P<newline>\n+)"
)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MARKDOWN_EMPHASIS_RE = re.compile(r"(\*\*|__)(.*?)\1|(?<!\w)(\*|_)([^*_]+?)\3(?!\w)")
_MARKDOWN_CODE_RE = re.compile(r"`([^`]+)`")
_MARKDOWN_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+")
_MARKDOWN_LIST_BULLET_RE = re.compile(r"(?m)^\s*[-*+]\s+")
_MARKDOWN_RULE_RE = re.compile(r"(?m)^\s*[-*_]{3,}\s*$")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")


def is_numbered_list_marker(text: str, punctuation_start: int, punctuation: str) -> bool:
    """Return True for markers like ``1.`` at the start of a line.

    TTS chunking should not emit ``1.`` as a standalone spoken chunk when the
    model starts a numbered list after a colon/newline.
    """
    if punctuation != ".":
        return False

    digit_start = punctuation_start
    while digit_start > 0 and text[digit_start - 1].isdigit():
        digit_start -= 1
    if digit_start == punctuation_start:
        return False

    prefix = text[:digit_start]
    if not prefix.strip():
        return True

    last_newline = max(prefix.rfind("\n"), prefix.rfind("\r"))
    return last_newline >= 0 and not prefix[last_newline + 1 :].strip()


def split_sentences(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []

    parts: list[str] = []
    start = 0

    for match in _SENTENCE_BOUNDARY_RE.finditer(cleaned):
        punctuation = match.group("punct")
        if punctuation is not None:
            if is_numbered_list_marker(cleaned, match.start("punct"), punctuation):
                continue
            boundary_end = match.end("punct")
        else:
            boundary_end = match.start("newline")

        part = cleaned[start:boundary_end].strip()
        if part:
            parts.append(part)
        start = match.end()

    tail = cleaned[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def chunk_text_for_tts(text: str, min_chars: int = 24) -> list[str]:
    """Split text into TTS-friendly chunks without tiny leading fragments."""
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    pending = ""

    for sentence in sentences:
        candidate = f"{pending} {sentence}".strip() if pending else sentence
        if len(candidate) < min_chars:
            pending = candidate
            continue

        chunks.append(candidate)
        pending = ""

    if pending:
        if chunks:
            chunks[-1] = f"{chunks[-1]} {pending}".strip()
        else:
            chunks.append(pending)

    return chunks


def normalize_text_for_tts(text: str) -> str:
    """Convert lightweight assistant markdown into speech-friendly text.

    UI/logs keep the original text. This function is only for text sent into TTS
    engines so models do not literally read markdown markers such as ``**``.
    """
    cleaned = text.strip()
    if not cleaned:
        return ""

    cleaned = _MARKDOWN_RULE_RE.sub("", cleaned)
    cleaned = _MARKDOWN_HEADING_RE.sub("", cleaned)
    cleaned = _MARKDOWN_LIST_BULLET_RE.sub("", cleaned)
    cleaned = _MARKDOWN_LINK_RE.sub(r"\1", cleaned)
    cleaned = _MARKDOWN_CODE_RE.sub(r"\1", cleaned)
    cleaned = _MARKDOWN_EMPHASIS_RE.sub(
        lambda match: (match.group(2) or match.group(4) or "").strip(),
        cleaned,
    )
    cleaned = cleaned.replace("\\*", "*")
    cleaned = _MULTISPACE_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
