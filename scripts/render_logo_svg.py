from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from xml.sax.saxutils import escape

from soca.app.style.gradient import ramp_color
from soca.app.style.palette import BG

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGO_TSX = REPO_ROOT / "ui" / "src" / "components" / "Logo.tsx"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "assets" / "soca-mark.svg"

WORDMARK = "SoCa"
FONT_SIZE = 34.0
# A monospace cell is the font's advance width, ~0.6em, and a terminal row is ~1.0em.
# Anything wider pulls the glyphs apart and the bird stops reading as one shape.
CELL_WIDTH = FONT_SIZE * 0.6
CELL_HEIGHT = FONT_SIZE * 0.84
PADDING = 26.0
WORDMARK_GAP = 12.0
MONOSPACE = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'DejaVu Sans Mono', monospace"


def read_bird_lines(source: Path = LOGO_TSX) -> tuple[str, ...]:
    """Read the mark out of the component that draws it on screen.

    Copying the glyphs into this file would create a second source of truth that
    silently rots the first time somebody redraws the bird in the TUI. Parsing the
    array instead means a change there either shows up here or fails loudly.
    """
    text = source.read_text(encoding="utf-8")
    match = re.search(r"BIRD_LINES[^=]*=\s*\[(.*?)\]", text, re.DOTALL)
    if match is None:
        raise ValueError(f"BIRD_LINES not found in {source}")
    lines = [ast.literal_eval(literal) for literal in re.findall(r'"[^"]*"', match.group(1))]
    if not lines:
        raise ValueError(f"BIRD_LINES in {source} is empty")
    return tuple(lines)


def _glyph(char: str, x: float, y: float, color: str) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" '
        f'text-anchor="middle">{escape(char)}</text>'
    )


def _bird_glyphs(lines: tuple[str, ...], origin_y: float) -> list[str]:
    """Diagonal ramp, matching the Bird component: t = (col/cols + row/rows) / 2."""
    rows = max(1, len(lines) - 1)
    glyphs: list[str] = []
    for row, line in enumerate(lines):
        chars = list(line)
        cols = max(1, len(chars) - 1)
        for col, char in enumerate(chars):
            if char == " ":
                continue
            t = (col / cols + row / rows) / 2
            x = PADDING + (col + 0.5) * CELL_WIDTH
            y = origin_y + (row + 0.8) * CELL_HEIGHT
            glyphs.append(_glyph(char, x, y, ramp_color(t)))
    return glyphs


def _wordmark_glyphs(width: float, baseline_y: float) -> list[str]:
    """Left-to-right ramp, matching the Wordmark component: t = i / (len - 1)."""
    last = max(1, len(WORDMARK) - 1)
    start_x = (width - len(WORDMARK) * CELL_WIDTH) / 2
    return [
        _glyph(char, start_x + (index + 0.5) * CELL_WIDTH, baseline_y, ramp_color(index / last))
        for index, char in enumerate(WORDMARK)
    ]


def render(lines: tuple[str, ...]) -> str:
    columns = max(len(line) for line in lines)
    width = columns * CELL_WIDTH + 2 * PADDING
    bird_height = len(lines) * CELL_HEIGHT
    height = PADDING + bird_height + WORDMARK_GAP + CELL_HEIGHT + PADDING

    glyphs = _bird_glyphs(lines, PADDING)
    glyphs += _wordmark_glyphs(width, PADDING + bird_height + WORDMARK_GAP + CELL_HEIGHT * 0.8)

    body = "\n    ".join(glyphs)
    # The dark plate is the TUI's own background, so the mark reads identically in
    # GitHub's light and dark themes instead of vanishing into one of them.
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="SoCa">\n'
        f'  <rect width="{width:.0f}" height="{height:.0f}" rx="12" fill="{BG}"/>\n'
        f'  <g font-family="{MONOSPACE}" font-size="{FONT_SIZE:.0f}" font-weight="bold" '
        f'xml:space="preserve">\n    {body}\n  </g>\n</svg>\n'
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render the SoCa splash mark to SVG, reading the glyphs and the dawn ramp "
        "from the same sources the TUI uses."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(read_bird_lines()), encoding="utf-8")
    try:
        shown: Path | str = args.output.relative_to(REPO_ROOT)
    except ValueError:
        shown = args.output
    print(f"wrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
