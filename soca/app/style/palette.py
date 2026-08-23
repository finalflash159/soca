"""SoCa design tokens — the "dawn" palette (bình minh).

SoCa sings at first light: one warm-gold accent over a pre-dawn indigo sky.
This module is the single source of truth for every color and glyph used by
BOTH the console renderers and the Textual TUI, so the two surfaces never
drift apart. It must stay importable without the `ui` extra — no textual
imports here, plain constants only.

Honors NO_COLOR (https://no-color.org) through :func:`st`.
"""

from __future__ import annotations

import os

NO_COLOR = bool(os.environ.get("NO_COLOR"))

# --- core surfaces -------------------------------------------------------------
BG = "#14131c"  # pre-dawn indigo-black
SURFACE = "#1b1a26"  # strips, panels, composer
BORDER = "#2a2838"  # hairlines and panel borders
TEXT = "#e8e3d8"  # warm off-white body text
MUTED = "#767287"  # secondary / inactive (indigo-tinted gray)

# --- the one accent ------------------------------------------------------------
ACCENT = "#e6c07b"  # dawn gold: SoCa's voice, focus, active mode
ACCENT_BRIGHT = "#f2d9a4"  # gradient ramp top (first light)
ACCENT_DEEP = "#b98a4e"  # gradient ramp base (horizon)

# ALT is a *tinted gray plus*, not a second accent: user markers and key hints.
ALT = "#a49ac9"

# --- semantics (desaturated so they sit inside the dawn family) ----------------
GOOD = "#8ac79a"
WARN = "#e09652"
BAD = "#e08398"

# Composite styles used verbatim across surfaces.
TITLE = f"bold {TEXT}"
SOCA_MARK_STYLE = f"bold {ACCENT}"
USER_MARK_STYLE = f"bold {ALT}"


class ICON:
    """Shared glyph set (unicode, deliberately no emoji)."""

    BIRD = "(o>"
    BIRD_CLOSED = "(.>"
    USER = "❯"
    POINTER = "❯"
    DOT = "·"
    OK = "✓"
    ERR = "✗"
    BAR = "▌"
    RULE = "─"
    STATE_ON = "●"
    STATE_OFF = "○"
    STATE_HALF = "◐"
    NOTE_A = "♪"
    NOTE_B = "♫"


# Cycled while SoCa speaks so the notes look alive.
MUSIC_FRAMES = ("♪      ", "♪ ♫    ", "♪ ♫ ♪  ", "  ♫ ♪  ")
# Quarter-moon spinner for loading states.
SPINNER_FRAMES = ("◐", "◓", "◑", "◒")


def st(style: str) -> str:
    """Drop styling when NO_COLOR is set, otherwise pass the style through."""
    return "" if NO_COLOR else style


__all__ = [
    "ACCENT",
    "ACCENT_BRIGHT",
    "ACCENT_DEEP",
    "ALT",
    "BAD",
    "BG",
    "BORDER",
    "GOOD",
    "ICON",
    "MUSIC_FRAMES",
    "MUTED",
    "NO_COLOR",
    "SOCA_MARK_STYLE",
    "SPINNER_FRAMES",
    "SURFACE",
    "TEXT",
    "TITLE",
    "USER_MARK_STYLE",
    "WARN",
    "st",
]
