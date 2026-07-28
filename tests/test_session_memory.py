import pytest

from soca.memory import MemoryContextBuilder, MemoryTurn, SessionMemory


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


def test_memory_context_builder_combines_profile_and_session():
    class FakeLongTermMemory:
        def read_profile(self) -> str:
            return "- Người dùng thích câu trả lời ngắn gọn."

    session = SessionMemory()
    session.append("user", "Tôi muốn ăn sáng lành mạnh.")

    context = MemoryContextBuilder(
        long_term=FakeLongTermMemory(),
        session=session,
        max_chars=300,
    ).build()

    assert "Long-term memory:" in context.prompt_text
    assert "Người dùng thích câu trả lời ngắn gọn" in context.prompt_text
    assert "Recent conversation:" in context.prompt_text
    assert "Tôi muốn ăn sáng lành mạnh" in context.prompt_text
