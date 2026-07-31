from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from .english_inventory import is_trained_ipa

_NLTK_DATA = Path(__file__).resolve().parents[3] / "models" / "nltk_data"


@lru_cache(maxsize=1)
def _load_g2p() -> Any:
    """Load G2p once. Return None when the lib/data is missing so callers
    degrade to letter spelling instead of raising."""
    if _NLTK_DATA.is_dir():
        os.environ.setdefault("NLTK_DATA", str(_NLTK_DATA))
        try:
            # g2p_en imports nltk internally, so by the time this module runs
            # nltk.data.path may already be resolved from a bare NLTK_DATA env
            # var read at import time by an unrelated earlier import. Append
            # directly so our data dir is searched regardless of import order.
            import nltk

            if str(_NLTK_DATA) not in nltk.data.path:
                nltk.data.path.append(str(_NLTK_DATA))
        except Exception:
            pass
    try:
        from g2p_en import G2p
    except Exception:
        return None
    try:
        return G2p()
    except Exception:
        return None


def arpabet_to_ipa(phones: list[str]) -> str:
    """ARPABET from g2p_en -> IPA, via eng_to_ipa's own converter.

    Not hand-mapped: cmu_to_ipa both strips stress digits and applies the
    dialect table, so the output dialect matches by construction. g2p_en
    also emits spaces/punctuation tokens, so those are filtered first.
    """
    from eng_to_ipa.transcribe import cmu_to_ipa

    cmu = " ".join(phone.lower() for phone in phones if phone and phone[0].isalpha())
    if not cmu:
        return ""
    try:
        # eng_to_ipa is unannotated; pyright infers stress_marking: str from its
        # 'all' default, but the library's own runtime check treats None as "off".
        return cmu_to_ipa([[cmu]], mark=False, stress_marking=None)[0][0]  # type: ignore[arg-type]
    except Exception:
        return ""


class G2pEnBackend:
    """ForeignG2P backed by g2p_en. Safe when the lib/data is absent (to_ipa -> None)."""

    def __init__(self) -> None:
        self._cache: dict[str, str | None] = {}

    def to_ipa(self, token: str) -> str | None:
        key = token.lower()
        if key in self._cache:
            return self._cache[key]
        result = self._compute(key)
        self._cache[key] = result
        return result

    def _compute(self, token: str) -> str | None:
        g2p = _load_g2p()
        if g2p is None:
            return None
        try:
            phones = list(g2p(token))
        except Exception:
            return None
        ipa = arpabet_to_ipa(phones)
        # Trained-inventory gate: better to spell than to hit an untrained embedding.
        return ipa if is_trained_ipa(ipa) else None
