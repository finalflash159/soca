from __future__ import annotations

from soca.knowledge import KnowledgeCitation, KnowledgeDocument, KnowledgeHit
from soca.memory import (
    CoreMemoryStore,
    MemoryAccessPlan,
    MemoryContext,
    MemoryContextBuilder,
    MemoryProfileResult,
    PromptContextAssembler,
    SessionMemory,
)


class _MemorySource:
    def __init__(self) -> None:
        self.retrieve_calls = 0

    def read_profile(self) -> str:
        return "người dùng thích câu trả lời rõ ràng"

    def retrieve_profile(self, query: str) -> MemoryProfileResult:
        self.retrieve_calls += 1
        return MemoryProfileResult(
            text=f"archive evidence for: {query}",
            mode="retrieved",
            evidence_status="supported",
            evidence_reason="test_hit",
        )


def test_normal_context_does_not_search_archive_and_archive_is_explicit() -> None:
    source = _MemorySource()
    session = SessionMemory(summary_enabled=False)
    session.append("user", "xin chào")
    session.append("assistant", "chào bạn")
    builder = MemoryContextBuilder(long_term=source, session=session)

    core = builder.build("TTS", include_archive=False)

    assert source.retrieve_calls == 0
    assert "người dùng thích" in core.prompt_text
    assert "User: xin chào" in core.prompt_text

    archive = builder.build(
        "TTS",
        include_archive=True,
        include_core=False,
        include_working=False,
    )
    assert source.retrieve_calls == 1
    assert "No local memory notes found" in archive.archive_text
    assert "User: xin chào" not in archive.prompt_text

    combined = PromptContextAssembler().assemble(
        core,
        archive,
        plan=MemoryAccessPlan(
            archive_mode="semantic",
            archive_query="TTS",
            reason="test_archive_selection",
        ),
    )
    assert combined.prompt_text.count("User: xin chào") == 1
    assert combined.prompt_text.count("No local memory notes found") == 1


def test_assembler_rejects_archive_without_explicit_plan() -> None:
    source = _MemorySource()
    builder = MemoryContextBuilder(long_term=source)
    core = builder.build("TTS", include_archive=False)
    archive = builder.build("TTS", include_archive=True, include_core=False, include_working=False)

    try:
        PromptContextAssembler().assemble(core, archive, plan=MemoryAccessPlan())
    except ValueError as exc:
        assert "explicit archive mode" in str(exc)
    else:
        raise AssertionError("archive must require an explicit access plan")


def test_core_store_is_separate_from_archive_profile(tmp_path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "profile.md").write_text("old archive profile", encoding="utf-8")
    (memory_dir / "core.json").write_text(
        '{"schema_version":1,"items":[{"id":"language","value":"Tiếng Việt",'
        '"approved_at":"2026-01-01T00:00:00Z","sensitivity":"normal",'
        '"updated_at":"2026-01-01T00:00:00Z","provenance":"user"}]}',
        encoding="utf-8",
    )

    core = CoreMemoryStore(tmp_path)
    assert core.read_profile() == "- [language] Tiếng Việt"


def test_invalid_core_degrades_without_blocking_working_memory(tmp_path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "core.json").write_text("{not-json", encoding="utf-8")
    session = SessionMemory(summary_enabled=False)
    session.append("user", "working vẫn dùng được")

    context = MemoryContextBuilder(
        core=CoreMemoryStore(tmp_path),
        session=session,
    ).build("xin chào", include_archive=False)

    assert context.degraded_reason == "core_invalid"
    assert "working vẫn dùng được" in context.prompt_text


def test_assembler_reserves_space_for_selected_archive_evidence() -> None:
    document = KnowledgeDocument("memory-a", "memory/a.md", "A", "TTS choice")
    hit = KnowledgeHit(document, 1.0, "TTS choice", 1, 1)
    citation = KnowledgeCitation("memory/a.md", "A", 1, 1, "memory")
    core = MemoryContextBuilder(session=SessionMemory()).build()
    archive = MemoryContext(
        profile_text="",
        session_text="",
        prompt_text="[M1] memory/a.md\nMemory: TTS choice",
        hits=(hit,),
        citations=(citation,),
        mode="retrieved",
        evidence_status="supported",
        evidence_reason="test_hit",
        archive_text="[M1] memory/a.md\nMemory: TTS choice",
    )
    core = MemoryContext(
        profile_text="",
        session_text="Recent conversation:\n" + "x" * 180,
        prompt_text="Recent conversation:\n" + "x" * 180,
        core_text="",
    )

    combined = PromptContextAssembler(max_chars=220).assemble(
        core,
        archive,
        plan=MemoryAccessPlan(
            archive_mode="semantic",
            archive_query="TTS",
            reason="test_archive_priority",
        ),
    )

    assert len(combined.prompt_text) <= 220
    assert "memory/a.md" in combined.prompt_text
    assert combined.citations == (citation,)
    assert combined.hits == (hit,)
