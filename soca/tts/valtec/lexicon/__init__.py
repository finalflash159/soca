"""Curated pronunciations for names statistical G2P cannot derive.

g2p_en predicts from English spelling rules learned over CMUdict. Technical
coinages, initialisms and brand names break those rules on purpose, so their
readings are looked up rather than predicted -- the same reason every
production TTS ships a lexicon.

Entries are grouped by domain so a term is added next to its peers. Values use
the eng_to_ipa dialect (see english_inventory) and tests verify every one
against the trained inventory, so a typo cannot reach the model.
"""

from __future__ import annotations

from .acronyms import ACRONYM_LEXICON
from .cmu_overrides import CMU_OVERRIDE_LEXICON
from .machine_learning import MACHINE_LEARNING_LEXICON
from .software import SOFTWARE_LEXICON
from .tools import TOOLS_LEXICON
from .trend_greetings import TREND_GREETING_LEXICON

# Lowercase-keyed domains, merged for lookup. Kept separate above so each
# domain stays reviewable; duplicate keys across domains are a test failure.
WORD_LEXICON_SOURCES: tuple[tuple[str, dict[str, str]], ...] = (
    ("machine_learning", MACHINE_LEARNING_LEXICON),
    ("software", SOFTWARE_LEXICON),
    ("tools", TOOLS_LEXICON),
)

WORD_LEXICON: dict[str, str] = {
    word: ipa for _domain, entries in WORD_LEXICON_SOURCES for word, ipa in entries.items()
}


class LexiconBackend:
    """ForeignG2P serving curated entries only; unknown tokens fall through.

    All-caps tokens resolve against ACRONYM_LEXICON and everything else against
    the merged lowercase lexicon, so "POST" the HTTP verb and "post" the
    ordinary English word stay independent.
    """

    def __init__(
        self,
        words: dict[str, str] | None = None,
        acronyms: dict[str, str] | None = None,
    ) -> None:
        self._words = dict(WORD_LEXICON if words is None else words)
        self._acronyms = dict(ACRONYM_LEXICON if acronyms is None else acronyms)

    def to_ipa(self, token: str) -> str | None:
        if len(token) > 1 and token.isupper():
            return self._acronyms.get(token)
        return self._words.get(token.lower())


__all__ = [
    "ACRONYM_LEXICON",
    "CMU_OVERRIDE_LEXICON",
    "MACHINE_LEARNING_LEXICON",
    "SOFTWARE_LEXICON",
    "TREND_GREETING_LEXICON",
    "TOOLS_LEXICON",
    "WORD_LEXICON",
    "WORD_LEXICON_SOURCES",
    "LexiconBackend",
]
