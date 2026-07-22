# tests/test_valtec_g2p.py
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

from soca.tts.valtec.g2p import PortableVietnameseG2P, TABLE_PATH

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_G2P = REPO_ROOT / "external/valtec-tts/deployments/web/vietnamese_g2p.js"


def _symbol_map() -> dict[str, int]:
    payload = json.loads(TABLE_PATH.read_text(encoding="utf-8"))
    symbols = {"_", "UNK", *",.!?;:'\"()[]{}"}
    for table in payload["tables"].values():
        for value in table.values():
            if isinstance(value, str):
                symbols.add(value)
                symbols.update(char for char in value if not unicodedata.combining(char))
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


def test_generated_tables_match_vendored_source_checksum():
    payload = json.loads(TABLE_PATH.read_text(encoding="utf-8"))
    assert payload["source_sha256"] == hashlib.sha256(UPSTREAM_G2P.read_bytes()).hexdigest()
    assert set(payload["tables"]) == {
        "onsets", "nuclei", "offglides", "onglides", "onoffglides",
        "codas", "tones", "gi", "qu",
    }


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


def test_unknown_word_is_counted_instead_of_silent_fallback(g2p):
    assert g2p.convert("xyz").unknown_phoneme_count == 1


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