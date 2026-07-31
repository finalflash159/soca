from __future__ import annotations

import pytest

from soca.tts.valtec.english_inventory import TRAINED_ENGLISH_IPA, is_trained_ipa
from soca.tts.valtec.english_lexicon import TECH_LEXICON, LexiconBackend
from soca.tts.valtec.foreign_g2p import ChainedForeignG2P


def test_every_lexicon_entry_stays_inside_the_trained_inventory() -> None:
    """A typo in a hand-written entry must fail here, never reach the model."""
    for word, ipa in TECH_LEXICON.items():
        outside = sorted({char for char in ipa if char not in TRAINED_ENGLISH_IPA})
        assert not outside, f"{word} -> {ipa!r} uses untrained characters: {outside}"
        assert is_trained_ipa(ipa)


def test_lexicon_keys_are_lowercase_so_lookup_can_match() -> None:
    for word in TECH_LEXICON:
        assert word == word.lower(), f"{word!r} would never be matched"


def test_lexicon_only_covers_tokens_the_cmu_path_misses() -> None:
    """Entries shadowing a CMU word are dead weight and hide upstream fixes."""
    eng_to_ipa = pytest.importorskip("eng_to_ipa")
    shadowed = [
        word for word in TECH_LEXICON if not eng_to_ipa.convert(word).endswith("*")
    ]
    assert not shadowed, f"already handled by the CMU path: {shadowed}"


def test_lexicon_backend_returns_none_for_unknown_tokens() -> None:
    backend = LexiconBackend()
    assert backend.to_ipa("github") == "gɪthəb"
    assert backend.to_ipa("GitHub") == "gɪthəb"  # case-insensitive
    assert backend.to_ipa("xyzzy") is None


def test_lexicon_wins_over_a_statistical_guess() -> None:
    class _Guess:
        def to_ipa(self, token: str) -> str | None:
            return "gɪθəb"  # what g2p_en actually predicts for github

    chained = ChainedForeignG2P((LexiconBackend(), _Guess()))
    assert chained.to_ipa("github") == "gɪthəb"
    assert chained.to_ipa("anything-else") == "gɪθəb"


def test_chain_falls_through_to_none_when_every_backend_declines() -> None:
    class _Silent:
        def to_ipa(self, token: str) -> str | None:
            return None

    assert ChainedForeignG2P((_Silent(), _Silent())).to_ipa("zzz") is None
    assert ChainedForeignG2P(()).to_ipa("zzz") is None


def test_real_backend_chain_fixes_the_measured_mispronunciations() -> None:
    pytest.importorskip("g2p_en")
    from soca.tts.valtec.foreign_g2p_en import G2pEnBackend

    chained = ChainedForeignG2P((LexiconBackend(), G2pEnBackend()))
    # Measured wrong before the lexicon existed (plan §6.1 listening review).
    assert chained.to_ipa("github") == "gɪthəb"
    assert chained.to_ipa("nginx") == "ɛnʤɪnɛks"
    assert chained.to_ipa("pytorch") == "paɪtɔrʧ"
    # Regular morphology still comes from g2p_en, not the lexicon.
    assert chained.to_ipa("tokenizer") == "toʊkənaɪzər"
    assert "tokenizer" not in TECH_LEXICON
