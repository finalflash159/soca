"""Color interpolation helpers for gradient text (splash bird, wordmark).

Ported from torlink's `lerpHex`/ramp approach: interpolate a small set of
stops across the characters of a glyph block so the mark catches the light
top-left to bottom-right, like dawn hitting the bird.
"""

from __future__ import annotations

from rich.text import Text

from soca.app.style.palette import ACCENT, ACCENT_BRIGHT, ACCENT_DEEP, NO_COLOR

# Default dawn ramp: first light -> gold -> horizon.
DAWN_RAMP: tuple[str, ...] = (ACCENT_BRIGHT, ACCENT, ACCENT_DEEP)


def _rgb(hex_color: str) -> tuple[int, int, int]:
    value = int(hex_color.lstrip("#"), 16)
    return (value >> 16) & 255, (value >> 8) & 255, value & 255


def lerp_hex(a: str, b: str, t: float) -> str:
    """Linear-interpolate two hex colors at ``t`` in [0, 1] -> new hex."""
    t = min(1.0, max(0.0, t))
    ar, ag, ab = _rgb(a)
    br, bg, bb = _rgb(b)
    mix = (round(x + (y - x) * t) for x, y in ((ar, br), (ag, bg), (ab, bb)))
    return "#" + "".join(f"{c:02x}" for c in mix)


def ramp_color(t: float, stops: tuple[str, ...] = DAWN_RAMP) -> str:
    """Sample a multi-stop ramp at ``t`` in [0, 1]."""
    if len(stops) == 1:
        return stops[0]
    t = min(1.0, max(0.0, t))
    span = len(stops) - 1
    idx = min(int(t * span), span - 1)
    local_t = t * span - idx
    return lerp_hex(stops[idx], stops[idx + 1], local_t)


def gradient_block(block: str, stops: tuple[str, ...] = DAWN_RAMP, *, bold: bool = True) -> Text:
    """Style a multi-line glyph block with a diagonal gradient, one char at a time.

    The blend factor is the average of the x and y positions, so color flows
    from the top-left stop to the bottom-right stop.
    """
    lines = block.splitlines() or [""]
    result = Text()
    rows = max(1, len(lines) - 1)
    for row, line in enumerate(lines):
        chars = list(line)
        cols = max(1, len(chars) - 1)
        for col, ch in enumerate(chars):
            if ch == " " or NO_COLOR:
                result.append(ch)
                continue
            t = (col / cols + row / rows) / 2
            style = ramp_color(t, stops)
            result.append(ch, style=f"bold {style}" if bold else style)
        if row < len(lines) - 1:
            result.append("\n")
    return result


__all__ = ["DAWN_RAMP", "gradient_block", "lerp_hex", "ramp_color"]
