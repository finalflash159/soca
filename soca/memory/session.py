from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from soca.core.text_budget import truncate
from soca.memory.base import MemoryRole, MemoryTurn

RECENT_CONVERSATION_HEADER = "Recent conversation:"
VALID_ROLES = {"user", "assistant"}


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


class SessionMemory:
    """RAM-only recent conversation memory with deterministic character budgets."""

    def __init__(
        self,
        turns: Iterable[MemoryTurn] | None = None,
        max_turns: int = 6,
        max_chars: int = 1600,
        max_turn_chars: int = 500,
    ) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than 0")
        if max_chars <= len(RECENT_CONVERSATION_HEADER):
            raise ValueError("max_chars must leave room for the session memory header")
        if max_turn_chars <= 0:
            raise ValueError("max_turn_chars must be greater than 0")

        self.max_turns = max_turns
        self.max_chars = max_chars
        self.max_turn_chars = max_turn_chars
        self._turns: deque[MemoryTurn] = deque()

        if turns is not None:
            for turn in turns:
                self.append(turn.role, turn.text)

    @property
    def turns(self) -> tuple[MemoryTurn, ...]:
        return tuple(self._turns)

    def append(self, role: MemoryRole, text: str) -> None:
        if role not in VALID_ROLES:
            raise ValueError(f"Unsupported memory role: {role}")

        normalized = _normalize_text(text)
        if not normalized:
            return

        self._turns.append(
            MemoryTurn(
                role=role,
                text=truncate(normalized, self.max_turn_chars),
            )
        )
        self._trim_turn_count()

    def clear(self) -> None:
        self._turns.clear()

    def render(self) -> str:
        if not self._turns:
            return ""

        selected_lines: list[str] = []
        used_chars = len(RECENT_CONVERSATION_HEADER)

        for turn in reversed(self._turns):
            line = self._format_turn(turn)
            line_cost = len(line) + 1
            remaining = self.max_chars - used_chars - 1

            if line_cost > remaining:
                if selected_lines:
                    continue

                truncated = truncate(line, max(0, remaining))
                if truncated:
                    selected_lines.append(truncated)
                    used_chars += len(truncated) + 1
                continue

            selected_lines.append(line)
            used_chars += line_cost

        if not selected_lines:
            return ""

        return "\n".join([RECENT_CONVERSATION_HEADER, *reversed(selected_lines)])

    def _trim_turn_count(self) -> None:
        while len(self._turns) > self.max_turns:
            self._turns.popleft()

    def _format_turn(self, turn: MemoryTurn) -> str:
        label = "User" if turn.role == "user" else "Assistant"
        return f"{label}: {turn.text}"
