"""Assemble already-selected memory blocks without performing retrieval."""

from __future__ import annotations

from soca.core.text_budget import truncate
from soca.memory.access import MemoryAccessPlan
from soca.memory.context import MemoryContext


class PromptContextAssembler:
    def __init__(self, *, max_chars: int = 64_000) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than 0")
        self.max_chars = max_chars

    def assemble(
        self,
        core_working: MemoryContext,
        archive: MemoryContext | None,
        *,
        plan: MemoryAccessPlan,
    ) -> MemoryContext:
        if plan.archive_mode != "none" and archive is None:
            raise ValueError("archive retrieval plan requires an archive context")
        if plan.archive_mode == "none" and archive is not None:
            raise ValueError("archive context requires an explicit archive mode")

        parts: list[str] = []
        if plan.include_core and plan.include_working:
            if core_working.prompt_text:
                parts.append(core_working.prompt_text)
        else:
            if plan.include_core and core_working.core_text:
                parts.append("Core memory:\n" + core_working.core_text)
            if plan.include_working and core_working.session_text:
                parts.append(core_working.session_text)
        if plan.archive_mode != "none" and archive is not None and archive.archive_text:
            parts.append(archive.archive_text)
        prompt_text = truncate("\n\n".join(parts), self.max_chars)
        evidence_context = archive if archive is not None else core_working
        return MemoryContext(
            profile_text=core_working.core_text if plan.include_core else "",
            session_text=core_working.session_text if plan.include_working else "",
            prompt_text=prompt_text,
            hits=(core_working.hits + archive.hits) if archive is not None else core_working.hits,
            citations=(core_working.citations + archive.citations)
            if archive is not None
            else core_working.citations,
            mode=evidence_context.mode,
            degraded_reason=evidence_context.degraded_reason,
            evidence_status=evidence_context.evidence_status,
            evidence_reason=evidence_context.evidence_reason,
            rejected_hit_count=evidence_context.rejected_hit_count,
            top_relevance=evidence_context.top_relevance,
            relevance_margin=evidence_context.relevance_margin,
            core_text=core_working.core_text if plan.include_core else "",
            archive_text=archive.archive_text if plan.archive_mode != "none" and archive is not None else "",
        )


__all__ = ["PromptContextAssembler"]
