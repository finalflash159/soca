"""Backward-compatible shortcut for downloading the default PhoGPT GGUF."""

from __future__ import annotations

import sys

from scripts.download_llm import main
from soca.llm.registry import DEFAULT_LLM_MODEL_KEY

if __name__ == "__main__":
    raise SystemExit(main(["--model", DEFAULT_LLM_MODEL_KEY, *sys.argv[1:]]))
