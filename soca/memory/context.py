from __future__ import annotations

from dataclasses import dataclass

from soca.core.text_budget import truncate
from soca.memory.base import LongTermMemorySource, SessionMemorySource


@dataclass(frozen=True)
class MemoryContext:
    profile_text: str
    session_text: str
    prompt_text: str


class MemoryContextBuilder:
    def __init__(
        self,
        long_term: LongTermMemorySource | None = None,
        session: SessionMemorySource | None = None,
        max_chars: int = 2200,
        profile_chars: int = 800,
    ) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than 0")
        if profile_chars <= 0:
            raise ValueError("profile_chars must be greater than 0")

        self.long_term = long_term
        self.session = session
        self.max_chars = max_chars
        self.profile_chars = profile_chars

    def build(self) -> MemoryContext:
        profile_text = ""
        session_text = ""
        parts: list[str] = []

        if self.long_term is not None:
            profile_text = truncate(self.long_term.read_profile(), self.profile_chars)
            if profile_text:
                parts.append(f"Long-term memory:\n{profile_text}")

        if self.session is not None:
            session_text = self.session.render().strip()
            if session_text:
                parts.append(session_text)

        prompt_text = truncate("\n\n".join(parts), self.max_chars)
        return MemoryContext(
            profile_text=profile_text,
            session_text=session_text,
            prompt_text=prompt_text,
        )
