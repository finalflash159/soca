from __future__ import annotations

from soca.memory import (
    MemoryAccessPlan,
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
