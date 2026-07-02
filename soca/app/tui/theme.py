"""Textual theme for the SoCa TUI, built from the shared dawn palette.

styles.tcss consumes the theme variables ($primary, $surface, ...) plus the
extra $soca-* variables registered here, so palette.py stays the single
source of truth for colors across console and TUI.
"""

from __future__ import annotations

from textual.theme import Theme

from soca.app.style import palette

THEME_NAME = "soca-dawn"

SOCA_DAWN = Theme(
    name=THEME_NAME,
    primary=palette.ACCENT,
    secondary=palette.ALT,
    accent=palette.ACCENT,
    background=palette.BG,
    surface=palette.SURFACE,
    panel=palette.SURFACE,
    foreground=palette.TEXT,
    success=palette.GOOD,
    warning=palette.WARN,
    error=palette.BAD,
    dark=True,
    variables={
        "soca-border": palette.BORDER,
        "soca-muted": palette.MUTED,
        "soca-alt": palette.ALT,
        # Focused-input border: accent softened so focus reads calm, not loud.
        "soca-focus": f"{palette.ACCENT} 60%",
        "footer-background": palette.BG,
        "footer-key-foreground": palette.ALT,
        "footer-description-foreground": palette.MUTED,
    },
)

# Compatibility aliases for the transitional Textual widgets (voice_view.py,
# widgets.py) that style inline Rich content from this module. They now draw
# from the shared dawn palette; GREEN/RED keep their old names but map to the
# semantic tokens.
NO_COLOR = palette.NO_COLOR
TEXT = palette.TEXT
MUTED = palette.MUTED
BORDER = palette.BORDER
ACCENT = palette.ACCENT
GREEN = palette.GOOD
RED = palette.BAD
TITLE = palette.TITLE
st = palette.st

__all__ = [
    "ACCENT",
    "BORDER",
    "GREEN",
    "MUTED",
    "NO_COLOR",
    "RED",
    "SOCA_DAWN",
    "TEXT",
    "THEME_NAME",
    "TITLE",
    "st",
]
