from __future__ import annotations

import pytest

from soca.tts.valtec.english_inventory import TRAINED_ENGLISH_IPA, is_trained_ipa
from soca.tts.valtec.foreign_g2p_en import arpabet_to_ipa
from soca.tts.valtec.g2p import PortableVietnameseG2P
from soca.tts.valtec.lexicon import CMU_OVERRIDE_LEXICON
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


def test_uppercase_acronym_falls_back_to_spelling_when_unclaimed() -> None:
    """A backend may claim an acronym ("JSON"), but declining must still spell.

    The CMU path stays lowercase-only regardless, so "ID" is never read as the
    English word "id".
    """
    sid = _english_symbol_map()
    seen: list[str] = []

    class _Declines:
        def to_ipa(self, token: str) -> str | None:
            seen.append(token)
            return None

    result = _g2p(sid, _Declines()).convert("TTS")

    assert seen == ["TTS"], "backend must receive the original casing"
    assert result.foreign_phone_count > 0  # spelled, not dropped
    assert result.unknown_phoneme_count == 0


def test_pronunciation_override_beats_the_cmu_dictionary() -> None:
    """CMU has "cache" but reads it "ca-shay"; the override must win."""
    sid = _english_symbol_map()
    plain = _g2p(sid, None).convert("cache")

    overridden = PortableVietnameseG2P(
        symbol_to_id=sid, language_id=7, tone_offset=16, add_blank=False,
        pronunciation_overrides={"cache": "kæʃ"},
    ).convert("cache")

    assert overridden.phone_ids != plain.phone_ids
    assert sid["ʃ"] in overridden.phone_ids
    assert sid["e"] not in overridden.phone_ids  # the spurious "-shay" diphthong
    assert overridden.unknown_phoneme_count == 0


def test_pronunciation_overrides_default_to_empty_and_change_nothing() -> None:
    sid = _english_symbol_map()
    assert _g2p(sid, None).pronunciation_overrides == {}
    assert (
        _g2p(sid, None).convert("cache").phone_ids
        == PortableVietnameseG2P(
            symbol_to_id=sid, language_id=7, tone_offset=16, add_blank=False,
            pronunciation_overrides={},
        ).convert("cache").phone_ids
    )


def test_demo_technical_terms_have_explicit_measured_readings() -> None:
    assert CMU_OVERRIDE_LEXICON["token"] == "tɔkɛn"
    assert CMU_OVERRIDE_LEXICON["remote"] == "rimoʊt"


def test_embedding_uses_native_speech_form_instead_of_foreign_lexicon() -> None:
    from soca.tts.valtec.lexicon import TECHNICAL_SPEECH_FORMS, TechnicalSpeechForm

    assert isinstance(TECHNICAL_SPEECH_FORMS["embedding"], TechnicalSpeechForm)
    assert TECHNICAL_SPEECH_FORMS["llm"].spoken == "large language model"
    assert TECHNICAL_SPEECH_FORMS["softmax"].spoken == "sóp mác"
    assert TECHNICAL_SPEECH_FORMS["cosine"].spoken == "cô sai"
    assert TECHNICAL_SPEECH_FORMS["paper"].spoken == "pây pờ"
    assert TECHNICAL_SPEECH_FORMS["embedding"].spoken == "em bê đinh"
    assert TECHNICAL_SPEECH_FORMS["api"].spoken == "ây pi ai"
    assert TECHNICAL_SPEECH_FORMS["pipeline"].spoken == "pai lain"
    assert TECHNICAL_SPEECH_FORMS["long-context"].spoken == "long con téc"
    assert TECHNICAL_SPEECH_FORMS["rope"].spoken == "rô pê"
    assert TECHNICAL_SPEECH_FORMS["scaling"].spoken == "sờ cê lình"
    assert TECHNICAL_SPEECH_FORMS["activation"].spoken == "ắc ti vây sần"
    assert TECHNICAL_SPEECH_FORMS["sparsity"].spoken == "sờ pác si ti"
    assert TECHNICAL_SPEECH_FORMS["interpretability"].spoken == (
        "in tơ pờ rơ tơ bi li ti"
    )
    assert TECHNICAL_SPEECH_FORMS["factuality"].spoken == "phác chu a li ti"
    assert TECHNICAL_SPEECH_FORMS["recompute"].spoken == "ri cầm piut"
    assert TECHNICAL_SPEECH_FORMS["err_connection_reset"].spoken == (
        "lỗi kết nối bị ngắt"
    )
    assert TECHNICAL_SPEECH_FORMS["remote"].spoken == "ri mốt"
    assert TECHNICAL_SPEECH_FORMS["remote"].pause_after
    assert TECHNICAL_SPEECH_FORMS["transformer"].spoken == "trăn phơ mơ"


def test_untrained_ipa_is_rejected_by_gate() -> None:
    assert is_trained_ipa("paɪtɔrʧ")
    assert not is_trained_ipa("paɪtɔɹʧ")  # ɹ untrained
    assert not is_trained_ipa("kʌp")  # ʌ untrained


def test_real_g2p_en_pronounces_oov_terms() -> None:
    pytest.importorskip("g2p_en")
    from soca.tts.valtec.foreign_g2p_en import G2pEnBackend

    backend = G2pEnBackend()
    for token in ("pytorch", "embedding", "github", "tokenizer"):
        ipa = backend.to_ipa(token)
        assert ipa, f"{token} could not be transcribed"
        assert all(c in TRAINED_ENGLISH_IPA for c in ipa)
