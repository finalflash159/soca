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

        core_text = core_working.prompt_text
        archive_text = ""
        if plan.archive_mode != "none" and archive is not None:
            archive_text = truncate(archive.archive_text, self.max_chars)
            core_budget = max(0, self.max_chars - len(archive_text) - 2)
            core_text = truncate(core_text, core_budget)

        parts: list[str] = []
        if plan.include_core and plan.include_working:
            if core_text:
                parts.append(core_text)
        else:
            if plan.include_core and core_working.core_text:
                parts.append("Core memory:\n" + core_working.core_text)
            if plan.include_working and core_working.session_text:
                parts.append(core_working.session_text)
        if archive_text:
            parts.append(archive_text)
        prompt_text = truncate("\n\n".join(parts), self.max_chars)
        archive_hits_source = archive.hits if archive is not None else ()
        archive_citations_source = archive.citations if archive is not None else ()
        archive_paths = {
            str(getattr(getattr(hit, "document", None), "path", ""))
            for hit in archive_hits_source
        }
        retained_archive_paths = {path for path in archive_paths if path and path in archive_text}
        archive_hits = tuple(
            hit
            for hit in archive_hits_source
            if str(getattr(getattr(hit, "document", None), "path", "")) in retained_archive_paths
        )
        archive_citations = tuple(
            citation
            for citation in archive_citations_source
            if citation.path in retained_archive_paths
        )
        evidence_context = archive if archive is not None and archive_text else core_working
        return MemoryContext(
            memory_text=core_working.core_text if plan.include_core else "",
            session_text=core_working.session_text if plan.include_working else "",
            prompt_text=prompt_text,
            hits=core_working.hits + archive_hits,
            citations=core_working.citations + archive_citations,
            mode=evidence_context.mode,
            degraded_reason=evidence_context.degraded_reason,
            evidence_status=evidence_context.evidence_status,
            evidence_reason=evidence_context.evidence_reason,
            rejected_hit_count=evidence_context.rejected_hit_count,
            top_relevance=evidence_context.top_relevance,
            relevance_margin=evidence_context.relevance_margin,
            score_separation=evidence_context.score_separation,
            query_coverage=evidence_context.query_coverage,
            sparse_top_score=evidence_context.sparse_top_score,
            dense_top_score=evidence_context.dense_top_score,
            retrieval_state=evidence_context.retrieval_state,
            retrieval_reason=evidence_context.retrieval_reason,
            core_text=core_working.core_text if plan.include_core else "",
            archive_text=archive_text,
        )


__all__ = ["PromptContextAssembler"]
