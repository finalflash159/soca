from __future__ import annotations

import pytest

from soca.tts.valtec.english_inventory import TRAINED_ENGLISH_IPA, is_trained_ipa
from soca.tts.valtec.foreign_g2p import ChainedForeignG2P
from soca.tts.valtec.lexicon import (
    ACRONYM_LEXICON,
    CMU_OVERRIDE_LEXICON,
    TREND_GREETING_LEXICON,
    WORD_LEXICON,
    WORD_LEXICON_SOURCES,
    LexiconBackend,
)

ALL_ENTRIES = {
    **WORD_LEXICON,
    **ACRONYM_LEXICON,
    **CMU_OVERRIDE_LEXICON,
    **TREND_GREETING_LEXICON,
}


def test_every_lexicon_entry_stays_inside_the_trained_inventory() -> None:
    """A typo in a hand-written entry must fail here, never reach the model."""
    for word, ipa in ALL_ENTRIES.items():
        outside = sorted({char for char in ipa if char not in TRAINED_ENGLISH_IPA})
        assert not outside, f"{word} -> {ipa!r} uses untrained characters: {outside}"
        assert is_trained_ipa(ipa)


def test_domain_sets_do_not_overlap() -> None:
    """Two domains claiming one word means an edit silently hits the wrong file."""
    seen: dict[str, str] = {}
    for domain, entries in WORD_LEXICON_SOURCES:
        for word in entries:
            assert word not in seen, f"{word!r} is in both {seen[word]} and {domain}"
            seen[word] = domain
    assert len(WORD_LEXICON) == len(seen)


def test_word_keys_are_lowercase_and_acronym_keys_are_uppercase() -> None:
    for word in WORD_LEXICON:
        assert word == word.lower(), f"{word!r} would never be matched"
    for acronym in ACRONYM_LEXICON:
        assert acronym == acronym.upper(), f"{acronym!r} would never be matched"


def test_lexicon_only_covers_tokens_the_cmu_path_misses() -> None:
    """Entries shadowing a CMU word are dead weight and hide upstream fixes.

    CMU_OVERRIDE_LEXICON is the one deliberate exception and is checked
    separately below.
    """
    eng_to_ipa = pytest.importorskip("eng_to_ipa")
    shadowed = [word for word in WORD_LEXICON if not eng_to_ipa.convert(word).endswith("*")]
    assert not shadowed, f"already handled by the CMU path: {shadowed}"


def test_every_cmu_override_shadows_a_real_and_different_cmu_entry() -> None:
    """An override must stay justified: CMU still has the word, still differs.

    If upstream corrects the dictionary this fails, prompting removal rather
    than freezing a stale hand-written reading forever.
    """
    eng_to_ipa = pytest.importorskip("eng_to_ipa")
    for word, ipa in CMU_OVERRIDE_LEXICON.items():
        cmu = eng_to_ipa.convert(word)
        assert not cmu.endswith("*"), f"{word!r} is not in CMU; it belongs in WORD_LEXICON"
        stripped = cmu.replace("ˈ", "").replace("ˌ", "")
        assert stripped != ipa, f"{word!r} override equals CMU ({cmu!r}); delete it"


def test_acronym_and_word_lookups_stay_independent() -> None:
    backend = LexiconBackend()
    # "POST" the HTTP verb is curated; "post" keeps its ordinary CMU reading.
    assert backend.to_ipa("POST") == "poʊst"
    assert backend.to_ipa("post") is None
    # Word entries are case-insensitive for ordinary capitalisation.
    assert backend.to_ipa("github") == "gɪthəb"
    assert backend.to_ipa("GitHub") == "gɪthəb"
    assert backend.to_ipa("xyzzy") is None


def test_trend_greetings_have_explicit_readings() -> None:
    assert TREND_GREETING_LEXICON["moshi"] == "moʊʃi"
    assert TREND_GREETING_LEXICON["annyeong"] == "ɑnjɔŋ"
    assert TREND_GREETING_LEXICON["yeoboseyo"] == "jəbosejo"


def test_lexicon_wins_over_a_statistical_guess() -> None:
    class _Guess:
        def to_ipa(self, token: str) -> str | None:
            return "gɪθəb"  # what g2p_en actually predicts for github

    chained = ChainedForeignG2P((LexiconBackend(), _Guess()))
    assert chained.to_ipa("github") == "gɪthəb"
    assert chained.to_ipa("something-else") == "gɪθəb"


def test_chain_falls_through_to_none_when_every_backend_declines() -> None:
    class _Silent:
        def to_ipa(self, token: str) -> str | None:
            return None

    assert ChainedForeignG2P((_Silent(), _Silent())).to_ipa("zzz") is None
    assert ChainedForeignG2P(()).to_ipa("zzz") is None


def test_statistical_backend_refuses_acronyms_so_they_stay_spelled() -> None:
    pytest.importorskip("g2p_en")
    from soca.tts.valtec.foreign_g2p_en import G2pEnBackend

    backend = G2pEnBackend()
    # Measured: g2p_en renders "TTS" as tiɛnis and "ID" as a word.
    assert backend.to_ipa("TTS") is None
    assert backend.to_ipa("ID") is None
    assert backend.to_ipa("RSS") is None
    # Lowercase still goes through.
    assert backend.to_ipa("tokenizer") == "toʊkənaɪzər"


def test_real_backend_chain_fixes_the_measured_mispronunciations() -> None:
    pytest.importorskip("g2p_en")
    from soca.tts.valtec.foreign_g2p_en import G2pEnBackend

    chained = ChainedForeignG2P((LexiconBackend(), G2pEnBackend()))
    # Measured wrong before the lexicon existed (plan §6.1b listening review).
    assert chained.to_ipa("github") == "gɪthəb"
    assert chained.to_ipa("nginx") == "ɛnʤɪnɛks"
    assert chained.to_ipa("pytorch") == "paɪtɔrʧ"
    assert chained.to_ipa("eval") == "ivæl"  # g2p_en said "evil"
    assert chained.to_ipa("cpu") == "sipiju"  # g2p_en said "ku"
    assert chained.to_ipa("JSON") == "ʤeɪsən"  # was letter-spelled
    # Regular morphology still comes from g2p_en, not the lexicon.
    assert chained.to_ipa("tokenizer") == "toʊkənaɪzər"
    assert "tokenizer" not in WORD_LEXICON
    # An acronym with no curated reading stays unclaimed, so it gets spelled.
    assert chained.to_ipa("TTFT") is None
