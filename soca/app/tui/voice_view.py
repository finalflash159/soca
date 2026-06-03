"""Compact live voice-status bar (plan §13.5, simplified).

Voice mode keeps the same two-part layout as chat (timeline + inspector): the
conversation streams into the timeline (ASR = your line, SoCa = reply) and the
technical details live in the Inspector. This module only adds a thin **live
status strip** on top: the state machine (listening/processing/speaking) plus
animated music notes while SoCa speaks.

Rendering is a pure function of an immutable :class:`VoiceTurnView` snapshot so
state transitions stay unit-testable without a running Textual app.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Group, RenderableType
from rich.text import Text
from textual.widgets import Static

from soca.app.tui import theme
from soca.app.tui.branding import DISPLAY_NAME

# Logical pipeline states surfaced to the user.
VOICE_STATES = ("loading", "idle", "listening", "processing", "speaking", "error")

_AMBER = "#d29922"  # processing/loading accent (local to voice; not in shared palette)

# state -> (glyph, label, style)
_STATE_BADGE: dict[str, tuple[str, str, str]] = {
    "loading": ("◐", "LOADING", _AMBER),
    "idle": ("○", "idle", theme.MUTED),
    "listening": ("●", "LISTENING", theme.ACCENT),
    "processing": ("●", "PROCESSING", _AMBER),
    "speaking": ("●", "SPEAKING", theme.GREEN),
    "error": ("●", "ERROR", theme.RED),
}

# Cycled while speaking so the notes look alive.
_MUSIC_FRAMES = ("♪      ", "♪ ♫    ", "♪ ♫ ♪  ", "  ♫ ♪  ")
# Cycled while loading so the spinner looks alive.
_SPINNER = ("◐", "◓", "◑", "◒")


@dataclass(frozen=True)
class VoiceTurnView:
    """Immutable snapshot driving the live status bar."""

    state: str = "idle"
    turn_index: int | None = None
    elapsed_s: float | None = None
    note: str = ""


class VoiceStatusBar(Static):
    """Thin live strip above the conversation: state machine + music notes."""

    def render_status(
        self,
        view: VoiceTurnView,
        *,
        profile: str,
        memory_on: bool,
        music_frame: int = 0,
    ) -> None:
        self.update(_render_status(view, profile, memory_on, music_frame))


# --- pure renderers -----------------------------------------------------------


def _bird(state: str) -> str:
    # A small constant bird; it "opens its beak" only while speaking.
    return "(o>" if state == "speaking" else "(.>"


def _badge(state: str, frame: int = 0) -> Text:
    glyph, label, style = _STATE_BADGE.get(state, _STATE_BADGE["idle"])
    if state == "loading":
        glyph = _SPINNER[frame % len(_SPINNER)]
    return Text(f"{glyph} {label}", style=theme.st(f"bold {style}"))


def _music(state: str, frame: int) -> str:
    if state != "speaking":
        return ""
    return _MUSIC_FRAMES[frame % len(_MUSIC_FRAMES)]


def _render_status(
    view: VoiceTurnView,
    profile: str,
    memory_on: bool,
    music_frame: int,
) -> RenderableType:
    line1 = Text()
    line1.append(f"{_bird(view.state)}  ", style=theme.st(f"bold {theme.GREEN}"))
    line1.append(f"{DISPLAY_NAME}   ", style=theme.st(theme.TITLE))
    line1.append_text(_badge(view.state, music_frame))
    music = _music(view.state, music_frame)
    if music:
        line1.append(f"   {music}", style=theme.st(theme.GREEN))
    if view.turn_index is not None:
        turn = f"   turn {view.turn_index}"
        if view.elapsed_s is not None:
            turn += f" · {view.elapsed_s:.2f}s"
        line1.append(turn, style=theme.st(theme.MUTED))

    line2 = Text()
    for name in ("listening", "processing", "speaking"):
        active = view.state == name
        glyph = "●" if active else "○"
        style = _STATE_BADGE[name][2] if active else theme.MUTED
        line2.append(f"{glyph} {name}   ", style=theme.st(f"bold {style}" if active else style))
    line2.append("     ")
    line2.append(f"{profile} · ", style=theme.st(theme.MUTED))
    line2.append("mem●" if memory_on else "mem○", style=theme.st(theme.MUTED))
    if view.note:
        note_style = theme.RED if view.state == "error" else theme.MUTED
        line2.append(f"   {view.note}", style=theme.st(note_style))

    return Group(line1, line2)


__all__ = [
    "VOICE_STATES",
    "VoiceStatusBar",
    "VoiceTurnView",
]
