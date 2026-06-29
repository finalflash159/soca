from __future__ import annotations

from rich.console import Console

from soca.app.console import print_followup, print_streaming_event
from soca.core.streaming import StreamingEvent


def _render(renderable_call) -> str:
    console = Console(record=True, width=80)
    renderable_call(console)
    return console.export_text()


def test_print_followup_uses_followup_label_not_fallback() -> None:
    text = _render(lambda c: print_followup(c, "Mình chưa nghe rõ, bạn nói lại nha."))
    assert "Follow-up:" in text
    assert "Fallback" not in text


def test_print_streaming_event_renders_repair_as_followup() -> None:
    event = StreamingEvent(
        type="repair",
        text="Alo alo, có ai ngoài đó hông ta?",
        metadata={"repair_kind": "no_input", "repair_action": "reprompt"},
    )
    text = _render(lambda c: print_streaming_event(c, event))
    assert "Follow-up:" in text
    assert "no_input" in text
