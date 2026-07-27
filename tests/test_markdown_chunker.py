from __future__ import annotations

from dataclasses import replace

import pytest

from soca.knowledge.base import KnowledgeDocument
from soca.knowledge.index.chunker import chunk_markdown


def _document(
    text: str,
    *,
    path: str = "wiki/note.md",
) -> KnowledgeDocument:
    return KnowledgeDocument(
        id=path,
        path=path,
        title="Note",
        text=text,
        tags=("test",),
    )


def test_chunker_keeps_heading_sections_separate_with_exact_line_ranges() -> None:
    document = _document("# Main\nMain body.\n## Details\nDetail body.\n### Final\nFinal body.")

    chunks = chunk_markdown(document, target_tokens=64, overlap_lines=1)

    assert [(chunk.line_start, chunk.line_end) for chunk in chunks] == [
        (1, 2),
        (3, 4),
        (5, 6),
    ]
    assert [chunk.text for chunk in chunks] == [
        "# Main\nMain body.",
        "## Details\nDetail body.",
        "### Final\nFinal body.",
    ]


def test_oversized_section_is_windowed_with_line_overlap() -> None:
    payload_lines = [
        f"line {index} alpha beta gamma delta epsilon zeta eta" for index in range(1, 10)
    ]
    document = _document("\n".join(["# Large section", *payload_lines]))

    chunks = chunk_markdown(document, target_tokens=32, overlap_lines=2)

    assert len(chunks) > 1
    assert chunks[0].line_start == 1
    assert chunks[-1].line_end == len(payload_lines) + 1
    assert all(
        current.line_start <= previous.line_end
        for previous, current in zip(chunks, chunks[1:], strict=False)
    )
    assert all(
        current.line_start > previous.line_start
        for previous, current in zip(chunks, chunks[1:], strict=False)
    )


def test_single_line_larger_than_target_still_makes_progress() -> None:
    oversized_line = " ".join(f"token-{index}" for index in range(80))
    document = _document(f"# Large\n{oversized_line}")

    chunks = chunk_markdown(document, target_tokens=32, overlap_lines=2)

    assert [(chunk.line_start, chunk.line_end) for chunk in chunks] == [(1, 1), (2, 2)]
    assert chunks[1].text == oversized_line


def test_chunk_text_matches_its_one_based_source_line_range() -> None:
    source_lines = ["# Main", "alpha", "beta", "## Other", "gamma", "delta"]
    document = _document("\n".join(source_lines))

    chunks = chunk_markdown(document, target_tokens=32, overlap_lines=1)

    for chunk in chunks:
        expected = "\n".join(source_lines[chunk.line_start - 1 : chunk.line_end]).strip()
        assert chunk.text == expected


def test_chunk_ids_are_deterministic_and_content_addressed() -> None:
    document = _document("# Main\nalpha beta")

    first = chunk_markdown(document)
    second = chunk_markdown(document)
    changed_text = chunk_markdown(replace(document, text="# Main\nalpha changed"))
    changed_path = chunk_markdown(replace(document, id="wiki/other.md", path="wiki/other.md"))

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert first[0].chunk_id != changed_text[0].chunk_id
    assert first[0].chunk_id != changed_path[0].chunk_id


def test_empty_markdown_has_no_chunks() -> None:
    assert chunk_markdown(_document("")) == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_tokens": 31},
        {"overlap_lines": -1},
    ],
)
def test_chunker_rejects_invalid_configuration(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        chunk_markdown(_document("alpha"), **kwargs)
