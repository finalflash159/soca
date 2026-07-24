# tests/test_valtec_g2p.py
from __future__ import annotations

import json
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

from soca.tts.valtec.g2p import TABLE_PATH, PortableVietnameseG2P

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_REVISION = "a5e77a138960c1101c022a61614d6ee72aeccadc"
UPSTREAM_G2P_URL = (
    "https://raw.githubusercontent.com/tronghieuit/valtec-tts/"
    f"{UPSTREAM_REVISION}/deployments/web/vietnamese_g2p.js"
)
UPSTREAM_G2P_SHA256 = "b568cc804e1f9d03a242cff869a71bcab2e662cbdd020190793c6eb3e861e63b"


def _symbol_map() -> dict[str, int]:
    payload = json.loads(TABLE_PATH.read_text(encoding="utf-8"))
    symbols = {"_", "UNK", *",.!?;:'\"()[]{}"}
    for table in payload["tables"].values():
        for value in table.values():
            if isinstance(value, str):
                symbols.add(value)
                symbols.update(char for char in value if not unicodedata.combining(char))
    symbols.add("wʷ")
    ordered = ["_", "UNK", *sorted(symbols - {"_", "UNK"})]
    return {symbol: index for index, symbol in enumerate(ordered)}


@pytest.fixture
def g2p() -> PortableVietnameseG2P:
    return PortableVietnameseG2P(
        symbol_to_id=_symbol_map(),
        language_id=7,
        tone_offset=16,
        add_blank=True,
    )


def test_generated_tables_pin_direct_github_source():
    payload = json.loads(TABLE_PATH.read_text(encoding="utf-8"))
    assert payload["generated_from"] == UPSTREAM_G2P_URL
    assert payload["source_repository"] == "https://github.com/tronghieuit/valtec-tts"
    assert payload["source_revision"] == UPSTREAM_REVISION
    assert payload["source_path"] == "deployments/web/vietnamese_g2p.js"
    assert payload["source_sha256"] == UPSTREAM_G2P_SHA256
    assert set(payload["tables"]) == {
        "onsets", "nuclei", "offglides", "onglides", "onoffglides",
        "codas", "tones", "gi", "qu",
    }


def test_ipa_tokenization_matches_upstream_character_semantics():
    symbol_to_id = {
        "_": 0,
        "UNK": 1,
        "x": 2,
        "w": 3,
        "wʷ": 4,
        "i": 5,
        "ə": 6,
        "n": 7,
        "iə": 8,
        "t": 9,
        "p": 10,
    }
    frontend = PortableVietnameseG2P(
        symbol_to_id=symbol_to_id,
        language_id=7,
        tone_offset=16,
        add_blank=False,
    )

    result = frontend.convert("khuyến tiếp")

    assert result.phone_ids == (0, 2, 4, 5, 6, 7, 9, 5, 6, 10, 0)
    assert symbol_to_id["iə"] not in result.phone_ids
    assert result.unknown_phoneme_count == 0


def test_six_vietnamese_tones_remain_distinct(g2p):
    result = g2p.convert("ma mà má mã mả mạ")
    spoken_tones = {tone for tone in result.tone_ids if tone >= 16}
    assert spoken_tones == {16, 17, 18, 19, 20, 21}


@pytest.mark.parametrize(
    "text",
    [
        "Tôi là Đặng Thùy Trâm.",
        "Nghỉ ngơi rồi khuyến nghị tiếp nhé.",
        "Định dạng lại bài sau, phân ra làm từng phân đoạn theo lời thoại nhân vật.",
    ],
)
def test_common_vietnamese_has_no_unknown_phoneme(g2p, text):
    result = g2p.convert(text)
    assert len(result.phone_ids) == len(result.tone_ids) == len(result.language_ids)
    assert result.unknown_phoneme_count == 0
    assert result.backend == "portable_web_port"


def test_oov_word_is_spelled_out_with_letter_names(g2p):
    # Upstream viphoneme reads OOV tokens letter by letter ("C" -> "si"),
    # it never feeds UNK embeddings into the acoustic model.
    result = g2p.convert("C")
    symbol_to_id = _symbol_map()
    id_to_symbol = {index: symbol for symbol, index in symbol_to_id.items()}
    spoken = [id_to_symbol[i] for i in result.phone_ids if i not in (0, symbol_to_id["UNK"])]
    assert spoken == ["ʂ", "i"]
    assert result.unknown_phoneme_count == 0


def test_oov_english_word_is_spelled_without_unknowns(g2p):
    assert g2p.convert("web").unknown_phoneme_count == 0
    assert g2p.convert("email").unknown_phoneme_count == 0


def test_english_word_uses_english_ipa_not_letter_spelling(g2p):
    # "test" -> t ɛ s t via eng_to_ipa; letter spelling would contain no ɛ.
    symbol_to_id = _symbol_map()
    result = g2p.convert("test")
    assert symbol_to_id["ɛ"] in result.phone_ids
    assert result.unknown_phoneme_count == 0


def test_all_caps_acronym_is_letter_spelled_not_read_as_english_word(g2p):
    # "TTS" is no Vietnamese syllable: spell "ti ti ét" like upstream viphoneme.
    # ("AI" lowercases to the valid syllable "ai" and is read as such upstream.)
    symbol_to_id = _symbol_map()
    result = g2p.convert("TTS")
    spoken = [i for i in result.phone_ids if i not in (0, symbol_to_id["UNK"])]
    assert len(spoken) >= 6
    assert result.unknown_phoneme_count == 0


def test_unmappable_character_still_counts_as_unknown(g2p):
    assert g2p.convert("ξ").unknown_phoneme_count == 1


def test_foreign_phone_count_marks_english_and_spelled_words(g2p):
    assert g2p.convert("email").foreign_phone_count > 0
    assert g2p.convert("TTS").foreign_phone_count > 0
    assert g2p.convert("xin chào").foreign_phone_count == 0


def test_punctuation_policy_maps_semicolon_and_drops_artifact_symbols(g2p):
    # '"', '(', '[', '{' produce audible artifacts on the Valtec checkpoint;
    # ';' is not in the model vocabulary. Map ';' to ',' and drop the rest.
    symbol_to_id = _symbol_map()
    result = g2p.convert('à; ("xem") [ừ] {ơ}')
    ids = set(result.phone_ids)
    assert symbol_to_id[","] in ids
    for dropped in '"()[]{};':
        assert symbol_to_id[dropped] not in ids
    assert result.unknown_phoneme_count == 0


def test_import_does_not_load_viphoneme_vinorm_or_torch():
    code = (
        "import sys; import soca.tts.valtec.g2p; "
        "assert 'viphoneme' not in sys.modules; "
        "assert 'vinorm' not in sys.modules; "
        "assert 'torch' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
