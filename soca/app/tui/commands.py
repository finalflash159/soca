from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SlashCommand:
    name: str
    description: str


SLASH_COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand("/status", "chuyển sang status/readiness view"),
    SlashCommand("/chat", "chuyển sang chat view"),
    SlashCommand("/voice", "chuyển sang voice monitor shell"),
    SlashCommand("/listen", "bắt đầu realtime voice loop nếu đang dừng"),
    SlashCommand("/stop", "dừng realtime voice loop sau turn hiện tại"),
    SlashCommand("/trace", "bật/tắt trace"),
    SlashCommand("/usage", "xem token/latency của session"),
    SlashCommand("/memory", "xem session memory trong RAM"),
    SlashCommand("/copy", "copy toàn bộ transcript vào clipboard nội bộ"),
    SlashCommand("/clear", "xóa timeline + session memory"),
    SlashCommand("/help", "xem danh sách lệnh"),
    SlashCommand("/exit", "thoát TUI"),
)


def filter_slash_commands(text: str) -> tuple[SlashCommand, ...]:
    """Return slash commands matching the current composer text.

    Matching is regex-first so `/m.*y` can match `/memory`. If the user is still
    typing an invalid regex, fall back to case-insensitive substring matching.
    """
    if not text.startswith("/"):
        return ()

    query = text[1:].strip()
    if not query:
        return SLASH_COMMANDS

    haystacks = {command: command.name[1:].lower() for command in SLASH_COMMANDS}

    try:
        pattern = re.compile(query, flags=re.IGNORECASE)
    except re.error:
        lowered = query.lower()
        return tuple(
            command
            for command, haystack in haystacks.items()
            if lowered in haystack
        )

    return tuple(
        command
        for command, haystack in haystacks.items()
        if pattern.search(haystack)
    )


def slash_help_text() -> str:
    lines = ["Slash commands:"]
    for command in SLASH_COMMANDS:
        lines.append(f"{command.name:<26} -> {command.description}")
    lines.extend(
        [
            "",
            "Ở status mode, câu chat thường sẽ không gửi vào runtime. Gõ /chat trước.",
            "Ở voice mode, loop tự chạy; dùng /stop để dừng và /listen để chạy lại.",
        ]
    )
    return "\n".join(lines)


__all__ = ["SLASH_COMMANDS", "SlashCommand", "filter_slash_commands", "slash_help_text"]
