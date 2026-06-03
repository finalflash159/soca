"""Shared calm-dark palette for the SoCa TUI content styling.

Panel/CSS theming lives in styles.tcss; this module is for the inline Rich
styles we apply to timeline/inspector/voice content. Honors NO_COLOR.
"""

from __future__ import annotations

import os

NO_COLOR = bool(os.environ.get("NO_COLOR"))

# Calm dark palette (GitHub-dark inspired). Used across widgets so every panel
# matches instead of drifting into bright cyan/blue Rich defaults.
TEXT = "#c9d1d9"  # default body text
MUTED = "#6e7681"  # secondary / inactive
BORDER = "#30363d"  # muted panel/table border
ACCENT = "#58a6ff"  # field labels, active mode (blue)
GREEN = "#3fb950"  # SoCa / assistant / ok
RED = "#f85149"  # error
TITLE = "bold #c9d1d9"  # soft panel/table title


def st(style: str) -> str:
    """Drop styling when NO_COLOR is set, otherwise pass the style through."""
    return "" if NO_COLOR else style


__all__ = [
    "ACCENT",
    "BORDER",
    "GREEN",
    "MUTED",
    "NO_COLOR",
    "RED",
    "TEXT",
    "TITLE",
    "st",
]
