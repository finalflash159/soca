"""Vietnamese Bag-of-Hallucinations matcher using Aho-Corasick.

Adapted from Barański et al. (ICASSP 2025). BoH = empirical list of phrases
PhoWhisper-tiny repeatedly emits when fed non-speech audio. Built by
running ASR over noise and counting repeats (see local/build_boh.py or
notebooks/02_build_vietnamese_boh.ipynb).

Aho-Corasick gives O(n + m + z) multi-pattern matching — much faster than
running `phrase in text` for hundreds of phrases on every transcript.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import ahocorasick


@dataclass(frozen=True)
class BoHMatch:
    """Result of matching a transcript against a BoH artifact."""

    matched_phrases: tuple[str, ...] = field(default_factory=tuple)
    cleaned_text: str = ""
    n_chars_removed: int = 0


def _normalize(text: str) -> str:
    """Same normalization used by build_boh.py before counting phrases.

    Without this, "Cảm ơn" in transcript and "cảm ơn" in BoH won't match.
    NFC + lowercase only — we keep punctuation since spans are character-aligned.
    """
    return unicodedata.normalize("NFC", text).lower()


class VietnameseBoH:
    """Loads BoH JSON, builds Aho-Corasick automaton, exposes match_and_clean.

    Usage:
        boh = VietnameseBoH()  # loads data/asr/vi_boh_v1.json
        result = boh.match_and_clean("cảm ơn các bạn đã xem video xin chào")
        # → BoHMatch(matched_phrases=("cảm ơn các bạn đã xem video",),
        #            cleaned_text="xin chào", n_chars_removed=27)
    """

    DATA_ASR_DIR = Path(__file__).resolve().parents[2] / "data" / "asr"
    MODEL_ARTIFACT_DIR = DATA_ASR_DIR / "boh"
    DEFAULT_PATH = DATA_ASR_DIR / "vi_boh_v1.json"

    @classmethod
    def artifact_name_for_model(cls, model_key: str) -> str:
        """Return the canonical BoH artifact filename for an ASR model key."""
        return f"{model_key}_vi_boh_v1.json"

    @classmethod
    def path_for_model(cls, model_key: str) -> Path:
        """Return the canonical model-specific BoH artifact path."""
        return cls.MODEL_ARTIFACT_DIR / cls.artifact_name_for_model(model_key)

    def __init__(self, boh_path: str | Path | None = None):
        path = Path(boh_path) if boh_path else self.DEFAULT_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"BoH artifact not found: {path}\nBuild it via: uv run python -m local.build_boh"
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
        """ASR model key used to build this artifact, if metadata is present."""
        value = self.metadata.get("model_key")
        return str(value) if value else None

    def is_compatible_with(self, model_key: str) -> bool:
        """Return true when artifact metadata matches the requested ASR model."""
        return self.model_key == model_key

    def match_and_clean(self, text: str) -> BoHMatch:
        """Find all BoH phrases in text. Return matched phrases + cleaned text.

        Overlapping matches: keep the longer one. Multiple disjoint matches
        all removed.
        """
        if not self._ready or not text:
            return BoHMatch(matched_phrases=(), cleaned_text=text, n_chars_removed=0)

        normalized = _normalize(text)
        spans: list[tuple[int, int, str]] = []
        for end_idx, (_idx, original_phrase) in self.automaton.iter(normalized):
            start_idx = end_idx - len(_normalize(original_phrase)) + 1
            spans.append((start_idx, end_idx + 1, original_phrase))

        if not spans:
            return BoHMatch(matched_phrases=(), cleaned_text=text, n_chars_removed=0)

        # Merge overlapping spans, keeping the longer one.
        spans.sort()
        merged: list[tuple[int, int, str]] = [spans[0]]
        for start, end, phrase in spans[1:]:
            last_start, last_end, last_phrase = merged[-1]
            if start <= last_end:
                last_len = last_end - last_start
                curr_len = end - start
                keep_phrase = last_phrase if last_len >= curr_len else phrase
                merged[-1] = (last_start, max(last_end, end), keep_phrase)
            else:
                merged.append((start, end, phrase))

        matched = tuple(phrase for _, _, phrase in merged)
        # Build cleaned text by excising matched spans from the original.
        result_parts: list[str] = []
        prev = 0
        for start, end, _ in merged:
            result_parts.append(text[prev:start])
            prev = end
        result_parts.append(text[prev:])
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
