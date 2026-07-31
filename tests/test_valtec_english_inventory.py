"""Guards against 'mapped to textbook IPA but the embedding is untrained'.

The norm floor is derived from MeloTTS's normal(0, 192**-0.5) init: untrained
rows sit around ~0.43, trained rows drift above 1.0. See
zplan/tts_foreign_g2p_fix_plan.vi.md §2.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from soca.tts.valtec.english_inventory import TRAINED_ENGLISH_IPA, WEAKLY_TRAINED
from soca.tts.valtec.foreign_g2p_en import arpabet_to_ipa

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = REPO_ROOT / "models/tts/valtec_multispeaker/reference/upstream"
TRAINED_NORM_FLOOR = 1.0
ARPABET = (
    "AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG "
    "OW OY P R S SH T TH UH UW V W Y Z ZH"
).split()


def test_converter_never_emits_a_character_outside_the_inventory() -> None:
    """Catches eng_to_ipa bumping its version and changing the symbols table."""
    for arpa in ARPABET:
        ipa = arpabet_to_ipa([arpa])
        missing = [c for c in ipa if c not in TRAINED_ENGLISH_IPA]
        assert not missing, f"{arpa} -> {ipa!r} uses characters outside the inventory: {missing}"


@pytest.mark.skipif(
    not (UPSTREAM / "text_encoder.onnx").is_file(),
    reason="Valtec artifact not provisioned",
)
def test_trained_inventory_matches_checkpoint_embeddings() -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import numpy_helper

    model = onnx.load(str(UPSTREAM / "text_encoder.onnx"))
    emb = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    weight = emb["enc_p.emb.weight"]
    sid = json.loads((UPSTREAM / "tts_config.json").read_text())["symbol_to_id"]
    norms = np.linalg.norm(weight, axis=1)

    for char in sorted(TRAINED_ENGLISH_IPA - WEAKLY_TRAINED):
        assert char in sid, f"{char!r} is missing from the vocab"
        assert norms[sid[char]] > TRAINED_NORM_FLOOR, (
            f"{char!r} norm={norms[sid[char]]:.2f} - not trained"
        )

    # Intentional exception: weak but above the dead baseline (~0.43). See §4.3.
    for char in sorted(WEAKLY_TRAINED):
        assert 0.5 < norms[sid[char]] <= TRAINED_NORM_FLOOR, (
            f"{char!r} norm={norms[sid[char]]:.2f} - no longer 'weak', update §4.3"
        )

    # Reverse trap: the IPA characters most often chosen by mistake must stay
    # outside the inventory.
    for char in ("ɹ", "ʌ", "ɡ", "ɐ"):
        assert char not in TRAINED_ENGLISH_IPA
        assert norms[sid[char]] < TRAINED_NORM_FLOOR
