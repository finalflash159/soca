from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from soca.memory import SessionMemory

from .context import ASRContextSourceRecord

if TYPE_CHECKING:
    from soca.knowledge.catalog import KnowledgeCatalogSnapshot


class KnowledgeCatalogSource(Protocol):
    def snapshot(self) -> KnowledgeCatalogSnapshot: ...


def runtime_context_records(
    knowledge_catalog: KnowledgeCatalogSource | None,
    session_memory: SessionMemory | None,
) -> tuple[ASRContextSourceRecord, ...]:
    records: list[ASRContextSourceRecord] = []
    if knowledge_catalog is not None:
        snapshot = knowledge_catalog.snapshot()
        for document in snapshot.documents:
            records.append(
                ASRContextSourceRecord(
                    value=document.title,
                    provenance=f"vault:{snapshot.revision}:{document.path}:title",
                    priority=30,
                )
            )
            records.extend(
                ASRContextSourceRecord(
                    value=tag,
                    provenance=f"vault:{snapshot.revision}:{document.path}:tag",
                    priority=20,
                )
                for tag in document.tags
            )
            records.extend(
                ASRContextSourceRecord(
                    value=heading.text,
                    provenance=f"vault:{snapshot.revision}:{document.path}:heading:{heading.line}",
                    priority=10,
                )
                for heading in document.headings
            )
    if session_memory is not None:
        records.extend(
            ASRContextSourceRecord(
                value=turn.text,
                provenance=f"session:{session_memory.working.thread_id}:{index}:{turn.role}",
                priority=40,
            )
            for index, turn in enumerate(session_memory.turns)
        )
    return tuple(records)


__all__ = ["KnowledgeCatalogSource", "runtime_context_records"]
