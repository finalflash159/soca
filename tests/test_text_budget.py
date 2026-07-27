from __future__ import annotations

import pytest

from soca.core.text_budget import truncate


@pytest.mark.parametrize(
    ("text", "max_chars", "expected"),
    [
        ("  xin chào  ", 20, "xin chào"),
        ("xin chào", 0, ""),
        ("xin chào", -1, ""),
        ("abcdef", 1, "a"),
        ("abcdef", 2, "ab"),
        ("abcdef", 3, "abc"),
        ("abcdef", 4, "a..."),
        ("abcdef", 5, "ab..."),
    ],
)
def test_truncate_has_the_existing_character_budget_semantics(
    text: str,
    max_chars: int,
    expected: str,
) -> None:
    assert truncate(text, max_chars) == expected


def test_truncate_never_exceeds_the_budget() -> None:
    result = truncate("  một đoạn văn dài hơn budget  ", 10)

    assert result == "một đoạ..."
    assert len(result) <= 10
