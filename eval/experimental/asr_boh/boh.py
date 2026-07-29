"""Research-only Vietnamese Bag-of-Hallucinations matcher.

This module is intentionally outside ``soca.asr``. It is used only by the
ASR ablation/evaluation tools and is never imported by the production voice
runtime.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import ahocorasick


@dataclass(frozen=True)
class BoHMatch:
    """Result of matching a research BoH artifact."""

    matched_phrases: tuple[str, ...] = field(default_factory=tuple)
    cleaned_text: str = ""
    n_chars_removed: int = 0


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).lower()


class VietnameseBoH:
    """Load and apply a research BoH artifact to an eval transcript."""

    DATA_ASR_DIR = Path(__file__).resolve().parents[3] / "data" / "asr"
    MODEL_ARTIFACT_DIR = DATA_ASR_DIR / "boh"
    DEFAULT_PATH = DATA_ASR_DIR / "vi_boh_v1.json"

    @classmethod
    def artifact_name_for_model(cls, model_key: str) -> str:
        return f"{model_key}_vi_boh_v1.json"

    @classmethod
    def path_for_model(cls, model_key: str) -> Path:
        return cls.MODEL_ARTIFACT_DIR / cls.artifact_name_for_model(model_key)

    def __init__(self, boh_path: str | Path | None = None):
        path = Path(boh_path) if boh_path else self.DEFAULT_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"BoH artifact not found: {path}\n"
                "Build it via: uv run python -m local.build_boh"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        self.metadata = data.get("metadata", {})
        self.boh_phrases = tuple(
            item["phrase"] for item in data.get("boh", []) if item.get("keep", True)
        )
        self.automaton = ahocorasick.Automaton()
        for idx, phrase in enumerate(self.boh_phrases):
            self.automaton.add_word(_normalize(phrase), (idx, phrase))
        self._ready = bool(self.boh_phrases)
        if self._ready:
            self.automaton.make_automaton()

    @property
    def model_key(self) -> str | None:
        value = self.metadata.get("model_key")
        return str(value) if value else None

    def is_compatible_with(self, model_key: str) -> bool:
        return self.model_key == model_key

    def match_and_clean(self, text: str) -> BoHMatch:
        if not self._ready or not text:
            return BoHMatch(matched_phrases=(), cleaned_text=text, n_chars_removed=0)
        normalized = _normalize(text)
        spans: list[tuple[int, int, str]] = []
        for end_idx, (_idx, original_phrase) in self.automaton.iter(normalized):
            start_idx = end_idx - len(_normalize(original_phrase)) + 1
            spans.append((start_idx, end_idx + 1, original_phrase))
        if not spans:
            return BoHMatch(matched_phrases=(), cleaned_text=text, n_chars_removed=0)

        spans.sort()
        merged: list[tuple[int, int, str]] = [spans[0]]
        for start, end, phrase in spans[1:]:
            last_start, last_end, last_phrase = merged[-1]
            if start <= last_end:
                last_len = last_end - last_start
                current_len = end - start
                keep_phrase = last_phrase if last_len >= current_len else phrase
                merged[-1] = (last_start, max(last_end, end), keep_phrase)
            else:
                merged.append((start, end, phrase))

        matched = tuple(phrase for _, _, phrase in merged)
        result_parts: list[str] = []
        previous = 0
        for start, end, _ in merged:
            result_parts.append(text[previous:start])
            previous = end
        result_parts.append(text[previous:])
        cleaned = " ".join("".join(result_parts).split())
        return BoHMatch(
            matched_phrases=matched,
            cleaned_text=cleaned,
            n_chars_removed=len(text) - len(cleaned),
        )

    def __len__(self) -> int:
        return len(self.boh_phrases)

    def __repr__(self) -> str:
        return f"VietnameseBoH(n_phrases={len(self.boh_phrases)})"
