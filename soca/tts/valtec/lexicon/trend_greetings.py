"""Curated readings for short foreign greetings used by voice repair prompts.

The visible repair text stays in its familiar Latin spelling. These IPA values
are deliberately approximate Vietnamese-listener readings, constrained to the
English inventory trained into the Valtec checkpoint. They are not a general
foreign-language G2P and do not invoke a network service or an extra model.
"""

from __future__ import annotations

TREND_GREETING_LEXICON: dict[str, str] = {
    # Japanese もしもし (moshi moshi), the telephone greeting.
    "moshi": "moʊʃi",
    # Korean 안녕 / 안녕하세요 and 여보세요, common casual or phone greetings.
    "annyeong": "ɑnjɔŋ",
    "anyeong": "ɑnjɔŋ",
    "annyeonghaseyo": "ɑnjɔŋhasejo",
    "yeoboseyo": "jəbosejo",
}
