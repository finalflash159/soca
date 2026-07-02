"""SoCa shared visual design system (palette, glyphs, gradients).

Import tokens from here so console and TUI stay on one palette:

    from soca.app.style import palette
    from soca.app.style.gradient import gradient_block
"""

from soca.app.style import gradient, palette

__all__ = ["gradient", "palette"]
