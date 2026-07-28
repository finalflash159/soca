from __future__ import annotations

from collections.abc import Iterable

from soca.core.text_budget import truncate
from soca.memory.base import MemoryRole, MemoryTurn
from soca.memory.working import WorkingMemory, WorkingMemoryPolicy

RECENT_CONVERSATION_HEADER = "Recent conversation:"
VALID_ROLES = {"user", "assistant"}


class SessionMemory:
    """Compatibility adapter over typed working-memory conversation turns.

    ``append(user)`` opens a turn and the following delivered ``append(assistant)``
    completes it.  The legacy flat ``turns`` view remains only for display and
    older integrations; compaction/state ownership lives in ``working``.
    """

    def __init__(
        self,
        turns: Iterable[MemoryTurn] | None = None,
        max_turns: int = 6,
        max_chars: int = 1600,
        max_turn_chars: int = 500,
        *,
        thread_id: str = "default",
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
        self.working = WorkingMemory(thread_id=thread_id, policy=WorkingMemoryPolicy())
        self._pending_sequences: list[int] = []
        if turns is not None:
            for turn in turns:
                self.append(turn.role, turn.text)

    @property
    def turns(self) -> tuple[MemoryTurn, ...]:
        flattened: list[MemoryTurn] = []
        for turn in self.working.snapshot.turns:
            flattened.append(MemoryTurn("user", turn.user_text))
            if turn.assistant_text:
                flattened.append(MemoryTurn("assistant", turn.assistant_text))
        return tuple(flattened)

    def append(self, role: MemoryRole, text: str) -> None:
        if role not in VALID_ROLES:
            raise ValueError(f"Unsupported memory role: {role}")
        normalized = " ".join(text.strip().split())
        if not normalized:
            return
        bounded = truncate(normalized, self.max_turn_chars)
        if role == "user":
            turn = self.working.begin_turn(bounded)
            self._pending_sequences.append(turn.sequence)
            return
        if not self._pending_sequences:
            # An assistant response without a user request has no trustworthy
            # turn ownership, therefore it cannot enter working memory.
            return
        sequence = self._pending_sequences.pop(0)
        self.working.finish_turn(sequence, bounded)
        # The approved default has no local summary model yet.  It preserves
        # prompt bounds through deterministic trim-only, never keyword/regex
        # extraction masquerading as a summary.
        if self.working.snapshot.token_count >= self.working.policy.high_watermark_tokens:
            self.working.trim_only()

    def clear(self) -> None:
        self.working = WorkingMemory(thread_id=self.working.thread_id, policy=self.working.policy)
        self._pending_sequences.clear()

    def render(self) -> str:
        raw = self.working.render()
        if len(raw) <= self.max_chars:
            return raw
        # Retain the newest bounded conversation block for legacy character
        # callers. WorkingMemory remains the source of truth and is token-bounded.
        lines = raw.splitlines()
        selected: list[str] = []
        used = len(RECENT_CONVERSATION_HEADER)
        for line in reversed(lines):
            if line in {RECENT_CONVERSATION_HEADER, "Earlier conversation summary:"}:
                continue
            cost = len(line) + 1
            if used + cost > self.max_chars:
                continue
            selected.append(line)
            used += cost
        if not selected:
            return ""
        return "\n".join([RECENT_CONVERSATION_HEADER, *reversed(selected)])
