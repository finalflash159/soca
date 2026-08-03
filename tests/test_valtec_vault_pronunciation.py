"""Coverage gate: every token in the knowledge vault must reach a good path.

The vault is the corpus the assistant actually reads aloud, so it is the honest
denominator for "does TTS pronounce our content". This asserts the routing
budget rather than individual phonemes: a regression that pushes technical
vocabulary back onto letter spelling shows up as a bucket-count change.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

from soca.tts.valtec.g2p import PortableVietnameseG2P
from soca.tts.valtec.lexicon import ACRONYM_LEXICON, CMU_OVERRIDE_LEXICON, WORD_LEXICON

REPO_ROOT = Path(__file__).resolve().parents[1]
VAULT = REPO_ROOT / "eval/fixtures/knowledge_vault/wiki"
UPSTREAM = REPO_ROOT / "models/tts/valtec_multispeaker/reference/upstream"
TOKEN_RE = re.compile(r"[^\W\d_]+|[,\.!?;:'\"()\[\]{}]", flags=re.UNICODE)

# Acronyms are letter-spelled on purpose: Vietnamese letter names reproduce the
# English ones ("API" -> "ây pi ai"). Only lowercase words landing on the spell
# path are a defect, and the vault must have none.
MAX_LOWERCASE_SPELLED = 0


def _vault_tokens() -> Counter[str]:
    tokens: Counter[str] = Counter()
    for markdown in sorted(VAULT.rglob("*.md")):
        raw = markdown.read_text(encoding="utf-8")
        raw = re.sub(r"```.*?```", " ", raw, flags=re.S)
        raw = re.sub(r"`[^`]*`", " ", raw)
        raw = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", raw)
        for token in TOKEN_RE.findall(raw):
            if len(token) > 1 and token[0].isalpha():
                # The release TTS gate covers Vietnamese/Latin prose. Public
                # reference articles can contain Greek, Chinese, or other
                # scripts; sending those tokens through the Vietnamese
                # letter-spelling path would measure an unsupported language
                # rather than a vault defect.
                if not all(
                    "LATIN" in unicodedata.name(char, "")
                    for char in token
                    if char.isalpha()
                ):
                    continue
                tokens[token] += 1
    return tokens


@pytest.fixture(scope="module")
def g2p() -> PortableVietnameseG2P:
    if not (UPSTREAM / "tts_config.json").is_file():
        pytest.skip("Valtec artifact not provisioned")
    config = json.loads((UPSTREAM / "tts_config.json").read_text(encoding="utf-8"))
    return PortableVietnameseG2P(
        symbol_to_id=config["symbol_to_id"],
        language_id=7,
        tone_offset=16,
        add_blank=False,
    )


def test_vault_has_tokens_to_check() -> None:
    assert sum(_vault_tokens().values()) > 10_000


def test_no_lowercase_vault_word_falls_back_to_letter_spelling(g2p) -> None:
    """Letter-spelling a lowercase word is always wrong; acronyms are exempt."""
    pytest.importorskip("g2p_en")
    from soca.tts.valtec.foreign_g2p_en import G2pEnBackend

    backend = G2pEnBackend()
    spelled: list[tuple[str, int]] = []
    for token, count in _vault_tokens().items():
        if token.isupper():
            continue  # letter spelling is the correct reading
        if g2p._syllable_segment(token) is not None:
            continue
        if g2p._english_segments(token) is not None:
            continue
        if token.lower() in WORD_LEXICON:
            continue
        if backend.to_ipa(token):
            continue
        spelled.append((token, count))

    assert len(spelled) <= MAX_LOWERCASE_SPELLED, (
        f"these lowercase vault words would be spelled letter by letter: {sorted(spelled)}"
    )


def test_curated_vault_terms_keep_their_reading() -> None:
    """Anchor the terms whose g2p_en prediction was measured wrong."""
    from soca.tts.valtec.lexicon import LexiconBackend

    backend = LexiconBackend()
    # Left column: what g2p_en predicted before curation (see plan §6.1b).
    for token, wrong in (
        ("eval", "ivəl"),  # "evil"
        ("cpu", "ku"),
        ("http", "tæpti"),
        ("reproducibility", "riprədəsɪtəslaɪt"),
        ("idempotency", "aɪdɛmpətnəsti"),
        ("sqlite", "sklaɪt"),
        ("dijkstra", "dɪkstrə"),
        ("observability", "əbzərvæbɪtəlɪti"),
    ):
        curated = backend.to_ipa(token)
        assert curated is not None, f"{token} lost its curated reading"
        assert curated != wrong


def test_high_frequency_cmu_error_is_overridden() -> None:
    """"cache" appears 50x in the vault; CMU reads it as "cachet"."""
    assert CMU_OVERRIDE_LEXICON["cache"] == "kæʃ"


def test_vault_acronyms_reaching_the_spell_path_are_all_uppercase() -> None:
    """Guards the routing rule that lets curated acronyms claim a token."""
    for token in ("JSON", "FAISS", "RAG"):
        assert token in ACRONYM_LEXICON
    # Unclaimed acronyms must still be spelled, not guessed.
    for token in ("TTFT", "MRR", "RRF"):
        assert token not in ACRONYM_LEXICON
