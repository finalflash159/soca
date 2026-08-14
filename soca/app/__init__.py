"""Application surfaces.

Both entry points are resolved lazily. Importing any module in this package used
to pull ``voice_loop`` and, through it, the ASR stack — transformers, torchaudio
and torch — so a metadata command such as ``soca status`` paid ~2.4 s to print a
table it could build from config files alone.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from soca.app.text_chat import run_text_chat
    from soca.app.voice_loop import run_voice_loop

__all__ = ["run_text_chat", "run_voice_loop"]

_LAZY_EXPORTS = {
    "run_text_chat": "soca.app.text_chat",
    "run_voice_loop": "soca.app.voice_loop",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(__all__)
