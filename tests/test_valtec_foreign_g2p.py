from __future__ import annotations

from soca.tts.valtec.english_inventory import TRAINED_ENGLISH_IPA, is_trained_ipa
from soca.tts.valtec.foreign_g2p_en import arpabet_to_ipa
from soca.tts.valtec.g2p import PortableVietnameseG2P
from tests.test_valtec_g2p import _symbol_map


def _english_symbol_map() -> dict[str, int]:
    sid = dict(_symbol_map())
    for sym in sorted(TRAINED_ENGLISH_IPA):
        sid.setdefault(sym, len(sid))
    return sid


class _StubBackend:
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def to_ipa(self, token: str) -> str | None:
        return self.mapping.get(token.lower())


def _g2p(sid: dict[str, int], backend: object | None) -> PortableVietnameseG2P:
    return PortableVietnameseG2P(
        symbol_to_id=sid, language_id=7, tone_offset=16,
        add_blank=False, foreign_g2p=backend,  # type: ignore[arg-type]
    )


def test_arpabet_maps_to_eng_to_ipa_dialect() -> None:
    assert arpabet_to_ipa(["AH0"]) == "ə"
    assert arpabet_to_ipa(["AH1"]) == "ə"  # not ʌ
    assert arpabet_to_ipa(["ER0"]) == "ər"  # keeps r
    assert arpabet_to_ipa(["R"]) == "r"  # not ɹ
    assert arpabet_to_ipa(["G"]) == "g"  # ASCII, not U+0261
    assert arpabet_to_ipa(["CH"]) == "ʧ"  # ligature, not tʃ
    assert arpabet_to_ipa(["JH"]) == "ʤ"
    assert arpabet_to_ipa(["P", "AY1", "T", "AO0", "R", "CH"]) == "paɪtɔrʧ"
    assert arpabet_to_ipa([" ", ",", "EH1"]) == "ɛ"  # drops non-phones
    assert arpabet_to_ipa(["HH", "AH0", "B"]) == "həb"  # h must survive
    assert arpabet_to_ipa([]) == ""


def test_oov_word_uses_backend_instead_of_letter_spelling() -> None:
    sid = _english_symbol_map()
    result = _g2p(sid, _StubBackend({"embedding": "ɪmbɛdɪŋ"})).convert("embedding")
    assert sid["ɪ"] in result.phone_ids and sid["ŋ"] in result.phone_ids
    assert sid["ʐ"] not in result.phone_ids  # 'giy' (letter name for g) is gone
    assert result.foreign_phone_count > 0
    assert result.unknown_phoneme_count == 0


def test_backend_none_keeps_letter_spelling_contract() -> None:
    sid = _symbol_map()
    assert _g2p(sid, None).convert("C").unknown_phoneme_count == 0
    assert _g2p(sid, None).convert("TTS").foreign_phone_count > 0


def test_backend_returning_none_falls_back_to_spelling() -> None:
    sid = _english_symbol_map()
    assert _g2p(sid, _StubBackend({})).convert("xyzzy").foreign_phone_count > 0


def test_uppercase_acronym_never_reaches_backend() -> None:
    class _Boom:
        def to_ipa(self, token: str) -> str | None:
            raise AssertionError("acronym must spell, never call the backend")

    _g2p(_english_symbol_map(), _Boom()).convert("TTS")


def test_untrained_ipa_is_rejected_by_gate() -> None:
    assert is_trained_ipa("paɪtɔrʧ")
    assert not is_trained_ipa("paɪtɔɹʧ")  # ɹ untrained
    assert not is_trained_ipa("kʌp")  # ʌ untrained
