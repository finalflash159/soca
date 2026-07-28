from __future__ import annotations

from dataclasses import dataclass

from soca.core.text_budget import truncate
from soca.knowledge import KnowledgeCitation
from soca.memory.base import (
    LongTermMemorySource,
    MemoryProfileResult,
    QueryAwareLongTermMemorySource,
    SessionMemorySource,
)


@dataclass(frozen=True)
class MemoryContext:
    profile_text: str
    session_text: str
    prompt_text: str
    hits: tuple[object, ...] = ()
    citations: tuple[KnowledgeCitation, ...] = ()
    mode: str = "blob"
    degraded_reason: str = ""


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

    def build(self, query: str | None = None, *, include_archive: bool = False) -> MemoryContext:
        profile = MemoryProfileResult(text="")
        session_text = ""
        parts: list[str] = []

        if self.long_term is not None:
            if (
                include_archive
                and query is not None
                and isinstance(self.long_term, QueryAwareLongTermMemorySource)
            ):
                profile = self.long_term.retrieve_profile(query)
            else:
                profile = MemoryProfileResult(text=self.long_term.read_profile())
            profile_text = truncate(profile.text, self.profile_chars)
            if profile_text:
                parts.append(f"Long-term memory:\n{profile_text}")
        else:
            profile_text = ""

        if self.session is not None:
            session_text = self.session.render().strip()
            if session_text:
                parts.append(session_text)

        prompt_text = truncate("\n\n".join(parts), self.max_chars)
        citations = tuple(
            KnowledgeCitation(
                path=str(getattr(getattr(hit, "document", None), "path", "")),
                title=str(getattr(getattr(hit, "document", None), "title", "")),
                line_start=getattr(hit, "line_start", None),
                line_end=getattr(hit, "line_end", None),
                source="memory",
            )
            for hit in profile.hits
            if str(getattr(getattr(hit, "document", None), "path", ""))
        )
        return MemoryContext(
            profile_text=profile_text,
            session_text=session_text,
            prompt_text=prompt_text,
            hits=profile.hits,
            citations=citations,
            mode=profile.mode,
            degraded_reason=profile.degraded_reason,
        )
