from types import SimpleNamespace
from typing import Any, cast

import pytest

from soca.memory import (
    MemoryCapacityError,
    MemoryContextBuilder,
    MemoryTurn,
    SessionMemory,
    WorkingMemoryPolicy,
    WorkingSummaryArtifact,
)
from soca.memory.summary import LocalSummaryWorkerProcess


class _ImmediateSummaryWorker:
    def __init__(self) -> None:
        self.job = None
        self.start_count = 0
        self.spec = SimpleNamespace(key="test-summary")

    @property
    def status(self):
        return SimpleNamespace(state="running" if self.job is not None else "idle")

    def start(self, job) -> bool:
        self.job = job
        self.start_count += 1
        return True

    def poll(self):
        if self.job is None:
            return None
        job = self.job
        self.job = None
        return {
            "ok": True,
            "artifact": WorkingSummaryArtifact(
                version=1,
                generation=job.generation,
                source_through_sequence=job.frozen_turns[-1].sequence,
                summary="Trạng thái đã compact.",
                decisions=("Dùng summary local.",),
            ).to_dict(),
        }

    def cancel(self) -> bool:
        running = self.job is not None
        self.job = None
        return running


class _FailOnceSummaryWorker(_ImmediateSummaryWorker):
    def __init__(self) -> None:
        super().__init__()
        self._failed_once = False

    def poll(self):
        if self.job is None:
            return None
        if not self._failed_once:
            self._failed_once = True
            self.job = None
            return {"ok": False, "error": "transient worker failure"}
        return super().poll()


class _AlwaysFailSummaryWorker(_ImmediateSummaryWorker):
    def poll(self):
        if self.job is None:
            return None
        self.job = None
        return {"ok": False, "error": "summary worker unavailable"}


def test_session_memory_renders_recent_conversation():
    memory = SessionMemory()
    memory.append("user", "Tôi muốn ăn sáng lành mạnh.")
    memory.append("assistant", "Bạn có thể ăn yến mạch với chuối.")

    rendered = memory.render()

    assert rendered == "\n".join(
        [
            "Recent conversation:",
            "User: Tôi muốn ăn sáng lành mạnh.",
            "Assistant: Bạn có thể ăn yến mạch với chuối.",
        ]
    )


def test_session_memory_ignores_empty_turns():
    memory = SessionMemory()
    memory.append("user", "   ")

    assert memory.turns == ()
    assert memory.render() == ""


def test_session_memory_counts_complete_user_assistant_pairs_as_turns():
    memory = SessionMemory(max_turns=2)
    memory.append("user", "một")
    memory.append("assistant", "hai")
    memory.append("user", "ba")
    memory.append("assistant", "bốn")
    memory.append("user", "năm")

    assert memory.turns == (
        MemoryTurn(role="user", text="một"),
        MemoryTurn(role="assistant", text="hai"),
        MemoryTurn(role="user", text="ba"),
        MemoryTurn(role="assistant", text="bốn"),
        MemoryTurn(role="user", text="năm"),
    )


def test_session_memory_enforces_render_character_budget_and_keeps_newest():
    memory = SessionMemory(max_turns=5, max_chars=80, max_turn_chars=80)
    memory.append("user", "câu cũ rất dài " * 4)
    memory.append("assistant", "trả lời cũ rất dài " * 4)
    memory.append("user", "câu mới")

    rendered = memory.render()

    assert len(rendered) <= 80
    assert "câu mới" in rendered
    assert "câu cũ" not in rendered


def test_session_memory_character_budget_keeps_structured_earlier_state():
    memory = SessionMemory(max_turns=5, max_chars=220, max_turn_chars=80)
    for index in range(6):
        turn = memory.working.begin_turn(f"user {index}")
        memory.working.finish_turn(turn.sequence, f"assistant {index}")
    job = memory.working.prepare_compaction(force=True)
    assert job is not None
    artifact = WorkingSummaryArtifact(
        version=1,
        generation=job.generation,
        source_through_sequence=job.frozen_turns[-1].sequence,
        summary="Giữ trạng thái cũ.",
        decisions=("Dùng TTS local.",),
    )
    assert memory.working.publish_summary(job, artifact)

    rendered = memory.render()

    assert len(rendered) <= 220
    assert "Earlier conversation state:" in rendered
    assert "Dùng TTS local." in rendered
    assert "user 5" in rendered


def test_session_memory_truncates_individual_turns():
    memory = SessionMemory(max_turn_chars=20)
    memory.append("user", "question")
    memory.append("assistant", "a" * 100)

    assert memory.turns[1].text == "a" * 17 + "..."


def test_session_memory_rejects_unknown_role():
    memory = SessionMemory()

    with pytest.raises(ValueError, match="Unsupported memory role"):
        memory.append("tool", "not supported")  # type: ignore[arg-type]


def test_session_memory_clear_removes_turns():
    memory = SessionMemory()
    memory.append("user", "Xin chào")

    memory.clear()

    assert memory.turns == ()
    assert memory.render() == ""


def test_session_memory_stats_expose_policy_and_prompt_sections() -> None:
    memory = SessionMemory(summary_enabled=False)
    memory.append("user", "Ghi nhớ quyết định dùng TTS local.")
    memory.append("assistant", "Đã ghi nhận.")

    stats = memory.stats()

    assert stats.current_tokens > 0
    assert stats.rendered_tokens > 0
    assert stats.hard_limit_tokens == 16_384
    assert stats.high_watermark_tokens == 15_000
    assert stats.target_tokens == 12_000
    assert stats.summary_tokens == 0
    assert stats.recent_tokens > 0
    assert stats.turn_count == 1
    assert stats.complete_turn_count == 1
    assert stats.pending_compaction is False
    assert stats.worker_state == "disabled"


def test_session_memory_automatically_runs_selected_summary_worker() -> None:
    fake_worker = _ImmediateSummaryWorker()
    worker = cast(LocalSummaryWorkerProcess, cast(Any, fake_worker))
    memory = SessionMemory(
        max_chars=4000,
        max_turn_chars=500,
        summary_worker=worker,
    )
    for index in range(48):
        memory.append("user", f"quyết định {index} " + "nội dung " * 55)
        memory.append("assistant", f"đã ghi nhận {index} " + "phản hồi " * 55)

    rendered = memory.render()

    assert fake_worker.start_count == 1
    assert memory.summary_model_key == "test-summary"
    assert memory.summary_worker_state == "idle"
    assert memory.working.snapshot.pending_compaction is False
    assert memory.working.snapshot.summary is not None
    assert "Dùng summary local." in rendered


def test_session_memory_retries_auto_compaction_after_a_transient_summary_failure() -> None:
    worker = _FailOnceSummaryWorker()
    memory = SessionMemory(
        max_chars=100_000,
        max_turn_chars=10_000,
        working_policy=WorkingMemoryPolicy(
            hard_limit_tokens=32_768,
            high_watermark_tokens=15_000,
            target_tokens=12_000,
            summary_budget_tokens=2_048,
            recent_budget_tokens=512,
            mode="background_summary",
        ),
        summary_worker=cast(LocalSummaryWorkerProcess, cast(Any, worker)),
    )
    turn_text = "x" * 6_000  # approximately 1,500 tokens per message
    for _ in range(5):
        memory.append("user", turn_text)
        memory.append("assistant", turn_text)

    assert worker.start_count == 1
    assert memory.compaction_status().status == "failed"
    token_count_after_fallback = memory.stats().current_tokens

    # The failure remains visible for the UI and the source turns remain intact.
    # Subsequent growth can cross 15K and schedule a new async summary job.
    assert memory.compaction_status().status == "failed"
    assert memory.stats().current_tokens == token_count_after_fallback
    for _ in range(3):
        memory.append("user", turn_text)
        memory.append("assistant", turn_text)
        if worker.start_count == 2:
            break

    assert worker.start_count == 2
    assert memory.compaction_status().status == "published"


def test_background_summary_failure_never_trims_source_history() -> None:
    worker = _AlwaysFailSummaryWorker()
    policy = WorkingMemoryPolicy(
        hard_limit_tokens=1_000,
        high_watermark_tokens=80,
        target_tokens=60,
        summary_budget_tokens=20,
        recent_budget_tokens=10,
        mode="background_summary",
    )
    memory = SessionMemory(
        max_chars=100_000,
        max_turn_chars=10_000,
        working_policy=policy,
        summary_worker=cast(LocalSummaryWorkerProcess, cast(Any, worker)),
    )
    for index in range(5):
        memory.append("user", f"source user {index} " + "x" * 80)
        memory.append("assistant", f"source answer {index} " + "y" * 80)

    assert memory.compaction_status().status == "failed"
    assert "source user 0" in memory.render()

    with pytest.raises(MemoryCapacityError):
        memory.append("user", "z" * 10_000)


def test_memory_context_builder_combines_core_and_session():
    class FakeLongTermMemory:
        def read_core(self) -> str:
            return "- Người dùng thích câu trả lời ngắn gọn."

    session = SessionMemory()
    session.append("user", "Tôi muốn ăn sáng lành mạnh.")

    context = MemoryContextBuilder(
        core=FakeLongTermMemory(),
        session=session,
        max_chars=300,
    ).build()

    assert "Core memory:" in context.prompt_text
    assert "Người dùng thích câu trả lời ngắn gọn" in context.prompt_text
    assert "Recent conversation:" in context.prompt_text
    assert "Tôi muốn ăn sáng lành mạnh" in context.prompt_text
