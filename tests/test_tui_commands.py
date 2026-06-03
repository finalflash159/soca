from __future__ import annotations

from soca.app.tui.commands import filter_slash_commands, slash_help_text


def names(text: str) -> list[str]:
    return [command.name for command in filter_slash_commands(text)]


def test_slash_filter_shows_all_commands_for_bare_slash() -> None:
    result = names("/")

    assert "/chat" in result
    assert "/status" in result
    assert "/usage" in result
    assert "/copy" in result


def test_slash_filter_supports_regex() -> None:
    assert names("/m.*y") == ["/memory"]


def test_slash_filter_falls_back_to_substring_for_invalid_regex() -> None:
    assert names("/mem[") == []
    assert names("/mem") == ["/memory"]


def test_slash_help_text_lists_commands() -> None:
    text = slash_help_text()

    assert "/chat" in text
    assert "/voice" in text
    assert "/copy" in text
