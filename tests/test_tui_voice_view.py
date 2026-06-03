from __future__ import annotations

import pytest

pytest.importorskip("textual")

import soca.app.tui.voice_view as vv
from soca.app.tui.voice_view import VOICE_STATES, VoiceTurnView


def _render_to_text(view: VoiceTurnView, *, profile="baseline", memory_on=True, frame=0) -> str:
    from rich.console import Console

    console = Console(width=80, no_color=True)
    with console.capture() as capture:
        console.print(vv._render_status(view, profile, memory_on, frame))
    return capture.get()


def test_render_status_does_not_raise_for_any_state() -> None:
    for state in VOICE_STATES:
        view = VoiceTurnView(state=state, turn_index=1, elapsed_s=1.42, note="x")
        text = _render_to_text(view)
        assert "listening" in text and "processing" in text and "speaking" in text


def test_speaking_shows_music_notes_and_open_beak() -> None:
    speaking = _render_to_text(VoiceTurnView(state="speaking"))
    idle = _render_to_text(VoiceTurnView(state="listening"))

    # Music notes + open-beak bird only while speaking.
    assert "♪" in speaking
    assert "(o>" in speaking
    assert "♪" not in idle
    assert "(o>" not in idle


def test_music_frame_cycles_distinct_frames() -> None:
    frames = {
        _render_to_text(VoiceTurnView(state="speaking"), frame=i).split("\n")[0]
        for i in range(len(vv._MUSIC_FRAMES))
    }
    # Different frames must render differently so the notes look animated.
    assert len(frames) > 1


def test_turn_view_is_immutable_dataclass() -> None:
    from dataclasses import replace

    base = VoiceTurnView()
    updated = replace(base, state="speaking")
    assert base.state == "idle"
    assert updated.state == "speaking"
