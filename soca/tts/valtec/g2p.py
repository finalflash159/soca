from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from .artifacts import ValtecOnnxArtifacts
from .foreign_g2p import ForeignG2P
from .frontend import ValtecModelInputs
from .normalizer import ValtecTextNormalizer

TABLE_PATH = Path(__file__).with_name("g2p_tables.json")
TONE_MAP = {1: 0, 2: 2, 3: 3, 4: 4, 5: 1, 6: 5}
PUNCTUATION = set(",.!?;:'\"()[]{}")
# Audit on the Valtec checkpoint: '"', '(', '[', '{' render audible artifacts,
# ';' is missing from the vocabulary and ':' renders with no audible pause,
# so only these symbols are spoken and ';'/':' become commas (a real pause).
SPOKEN_PUNCTUATION = {",", ".", "!", "?", "'"}
PUNCTUATION_REMAP = {";": ",", ":": ","}
MODIFIERS = {"ʷ", "ʰ", "ː"}
STRESS_MARKS = {"ˈ", "ˌ"}
# eng_to_ipa emits ligatures/rhotic vowels absent from the Valtec vocabulary.
ENGLISH_IPA_REWRITE = str.maketrans({"ʧ": "tʃ", "ʤ": "dʒ", "ɚ": "ə", "ɝ": "ə"})
# Upstream viphoneme spells OOV tokens letter by letter with these Vietnamese
# letter names instead of feeding UNK embeddings into the acoustic model.
LETTER_NAMES = {
    "a": "ây", "ă": "á", "â": "ớ", "b": "bi", "c": "si", "d": "đi", "đ": "đê",
    "e": "i", "ê": "ê", "f": "ép", "g": "giy", "h": "ếch", "i": "ai",
    "j": "giây", "k": "cây", "l": "eo", "m": "em", "n": "en", "o": "âu",
    "ô": "ô", "ơ": "ơ", "p": "pi", "q": "kiu", "r": "a", "s": "ét", "t": "ti",
    "u": "diu", "ư": "ư", "v": "vi", "w": "đắp liu", "x": "ít", "y": "quai",
    "z": "giét",
}


@lru_cache(maxsize=1)
def _english_ipa_converter() -> Any:
    """Same English G2P library upstream viphoneme uses; None when missing."""
    try:
        import eng_to_ipa
    except ImportError:
        return None
    return eng_to_ipa


@dataclass(frozen=True)
class _Syllable:
    onset: str
    nucleus: str
    coda: str
    tone: int
    oov: bool = False


def _load_tables(path: Path = TABLE_PATH) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported Valtec G2P table schema")
    tables = payload.get("tables")
    expected = {
        "onsets", "nuclei", "offglides", "onglides", "onoffglides",
        "codas", "tones", "gi", "qu",
    }
    if not isinstance(tables, dict) or set(tables) != expected:
        raise ValueError("Valtec G2P tables are incomplete")
    return tables


class PortableVietnameseG2P:
    def __init__(
        self,
        *,
        symbol_to_id: dict[str, int],
        language_id: int,
        tone_offset: int,
        add_blank: bool,
        table_path: Path = TABLE_PATH,
        foreign_g2p: ForeignG2P | None = None,
    ) -> None:
        if "_" not in symbol_to_id or "UNK" not in symbol_to_id:
            raise ValueError("Valtec symbol_to_id must contain '_' and 'UNK'")
        self.symbol_to_id = {str(symbol): int(index) for symbol, index in symbol_to_id.items()}
        if self.symbol_to_id["_"] != 0:
            raise ValueError("Valtec blank/boundary symbol '_' must have id 0")
        self.language_id = language_id
        self.tone_offset = tone_offset
        self.add_blank = add_blank
        self.tables = _load_tables(table_path)
        self.foreign_g2p = foreign_g2p

    def _transcribe(self, raw_word: str) -> _Syllable:
        word = raw_word.lower()
        onsets = self.tables["onsets"]
        codas = self.tables["codas"]
        onset = ""
        coda = ""
        onset_width = 0
        coda_width = 0
        for width in (3, 2, 1):
            candidate = word[:width]
            if candidate in onsets:
                onset = str(onsets[candidate])
                onset_width = width
                break
        for width in (2, 1):
            candidate = word[-width:]
            if candidate in codas:
                coda = str(codas[candidate])
                coda_width = width
                break

        stop = len(word) - coda_width if coda_width else len(word)
        nucleus_key = word[onset_width:stop]
        if len(word) == 3 and word[0] == "g" and word[1] in "iíìĩị" and coda:
            onset, nucleus_key = "z", "i"

        nucleus = ""
        if nucleus_key in self.tables["nuclei"]:
            nucleus = str(self.tables["nuclei"][nucleus_key])
        elif nucleus_key in self.tables["onglides"]:
            nucleus = str(self.tables["onglides"][nucleus_key])
            if onset != "kw":
                onset = f"{onset}w" if onset else "w"
        elif nucleus_key in self.tables["onoffglides"]:
            glide = str(self.tables["onoffglides"][nucleus_key])
            nucleus, coda = glide[:-1], glide[-1]
            if onset != "kw":
                onset = f"{onset}w" if onset else "w"
        elif nucleus_key in self.tables["offglides"]:
            glide = str(self.tables["offglides"][nucleus_key])
            nucleus, coda = glide[:-1], glide[-1]
        elif word in self.tables["gi"]:
            value = str(self.tables["gi"][word])
            onset, nucleus = value[0], value[1:]
        elif word in self.tables["qu"]:
            value = str(self.tables["qu"][word])
            onset, nucleus = value[:-1], value[-1]
        else:
            return _Syllable("", word, "", 1, oov=True)

        tone = next(
            (int(self.tables["tones"][char]) for char in word if char in self.tables["tones"]),
            1,
        )
        if nucleus == "a" and coda == "ɲ":
            nucleus = "ɛ"
        if nucleus == "a" and coda == "k" and coda_width == 2:
            nucleus = "ɛ"
        if nucleus in {"u", "o", "ɔ"}:
            if coda == "ŋ":
                coda = "ŋ͡m"
            elif coda == "k":
                coda = "k͡p"
        return _Syllable(onset, nucleus, coda, tone)

    def _tokenize_ipa(self, ipa: str, *, drop_unknown: bool = False) -> tuple[list[int], int]:
        phones: list[str] = []
        for char in ipa:
            if unicodedata.combining(char):
                continue
            if char in MODIFIERS:
                if phones:
                    phones[-1] += char
                continue
            phones.append(char)

        ids: list[int] = []
        unknown = 0
        for phone in phones:
            phone_id = self.symbol_to_id.get(phone)
            if phone_id is None:
                if drop_unknown:
                    continue
                phone_id = self.symbol_to_id["UNK"]
                unknown += 1
            ids.append(phone_id)
        return ids, unknown

    def _syllable_segment(self, word: str) -> tuple[list[int], int, int] | None:
        """Return (ids, internal_tone, unknown_count) or None when the word is OOV."""
        syllable = self._transcribe(word)
        if syllable.oov:
            return None
        ids, unknown = self._tokenize_ipa(
            f"{syllable.onset}{syllable.nucleus}{syllable.coda}"
        )
        return ids, TONE_MAP.get(syllable.tone, 0), unknown

    def _english_segments(self, token: str) -> list[tuple[list[int], int, int, bool]] | None:
        converter = _english_ipa_converter()
        if converter is None:
            return None
        ipa = converter.convert(token.lower())
        if not ipa or ipa.endswith("*") or " " in ipa:
            return None
        cleaned = "".join(
            char for char in ipa.translate(ENGLISH_IPA_REWRITE) if char not in STRESS_MARKS
        )
        ids, _ = self._tokenize_ipa(cleaned, drop_unknown=True)
        return [(ids, 0, 0, True)] if ids else None

    def _foreign_segments(self, token: str) -> list[tuple[list[int], int, int, bool]] | None:
        if self.foreign_g2p is None:
            return None
        ipa = self.foreign_g2p.to_ipa(token)
        if not ipa:
            return None
        ids, _ = self._tokenize_ipa(ipa, drop_unknown=True)
        return [(ids, 0, 0, True)] if ids else None

    def _word_segments(self, token: str) -> list[tuple[list[int], int, int, bool]]:
        """Return (ids, internal_tone, unknown_count, foreign) segments for a token."""
        direct = self._syllable_segment(token)
        if direct is not None:
            return [(*direct, False)]
        # Upstream viphoneme reads lowercase OOV words through English G2P and
        # spells all-caps acronyms/letters with Vietnamese letter names.
        if len(token) > 1 and not token.isupper():
            english = self._english_segments(token)
            if english is not None:
                return english
            foreign = self._foreign_segments(token)
            if foreign is not None:
                return foreign
        spelled = " ".join(
            LETTER_NAMES.get(char, char) for char in token.lower()
        ).split()
        return cast(list[tuple[list[int], int, int, bool]], [
            (*(self._syllable_segment(part) or ([self.symbol_to_id["UNK"]], 0, 1)), True)
            for part in spelled
        ])

    def convert(self, text: str) -> ValtecModelInputs:
        phone_ids: list[int] = [self.symbol_to_id["_"]]
        tone_ids: list[int] = [self.tone_offset]
        language_ids: list[int] = [self.language_id]
        foreign_flags: list[int] = [0]
        unknown_count = 0
        foreign_count = 0
        tokens = re.findall(r"[^\W\d_]+|[,\.!?;:'\"()\[\]{}]", text, flags=re.UNICODE)
        for token in tokens:
            if token in PUNCTUATION:
                spoken = PUNCTUATION_REMAP.get(token, token)
                phone_id = (
                    self.symbol_to_id.get(spoken)
                    if spoken in SPOKEN_PUNCTUATION
                    else None
                )
                if phone_id is None:
                    continue
                phone_ids.append(phone_id)
                tone_ids.append(self.tone_offset)
                language_ids.append(self.language_id)
                foreign_flags.append(0)
                continue
            for ids, internal_tone, unknown, foreign in self._word_segments(token):
                phone_ids.extend(ids)
                tone_ids.extend([internal_tone + self.tone_offset] * len(ids))
                language_ids.extend([self.language_id] * len(ids))
                foreign_flags.extend([int(foreign)] * len(ids))
                unknown_count += unknown
                if foreign:
                    foreign_count += len(ids)

        phone_ids.append(self.symbol_to_id["_"])
        tone_ids.append(self.tone_offset)
        language_ids.append(self.language_id)
        foreign_flags.append(0)
        if self.add_blank:
            phones_with_blank: list[int] = []
            tones_with_blank: list[int] = []
            languages_with_blank: list[int] = []
            flags_with_blank: list[int] = []
            for phone, tone, language, flag in zip(
                phone_ids, tone_ids, language_ids, foreign_flags, strict=True
            ):
                phones_with_blank.extend((0, phone))
                tones_with_blank.extend((0, tone))
                languages_with_blank.extend((self.language_id, language))
                flags_with_blank.extend((0, flag))
            phone_ids = [*phones_with_blank, 0]
            tone_ids = [*tones_with_blank, 0]
            language_ids = [*languages_with_blank, self.language_id]
            foreign_flags = [*flags_with_blank, 0]
        return ValtecModelInputs(
            phone_ids=tuple(phone_ids),
            tone_ids=tuple(tone_ids),
            language_ids=tuple(language_ids),
            backend="portable_web_port",
            unknown_phoneme_count=unknown_count,
            foreign_phone_count=foreign_count,
            foreign_flags=tuple(foreign_flags) if foreign_count else (),
        )


class ValtecVietnameseFrontend:
    def __init__(self, normalizer: ValtecTextNormalizer, g2p: PortableVietnameseG2P) -> None:
        self.normalizer = normalizer
        self.g2p = g2p

    @classmethod
    def from_artifacts(cls, artifacts: ValtecOnnxArtifacts) -> ValtecVietnameseFrontend:
        config = json.loads(artifacts.config.read_text(encoding="utf-8"))
        symbol_to_id = config.get("symbol_to_id")
        if not isinstance(symbol_to_id, dict):
            raise ValueError("Valtec tts_config.json is missing symbol_to_id")
        configured_language = config.get("language_id_map", {}).get("VI")
        if configured_language is not None and int(configured_language) != artifacts.language_id_vi:
            raise ValueError("Valtec manifest/config Vietnamese language id mismatch")
        foreign: ForeignG2P | None = None
        if config.get("foreign_g2p") == "g2p_en":
            from .foreign_g2p_en import G2pEnBackend

            foreign = G2pEnBackend()
        return cls(
            ValtecTextNormalizer(),
            PortableVietnameseG2P(
                symbol_to_id=symbol_to_id,
                language_id=artifacts.language_id_vi,
                tone_offset=artifacts.tone_offset_vi,
                add_blank=artifacts.add_blank,
                foreign_g2p=foreign,
            ),
        )

    def prepare(self, text: str) -> ValtecModelInputs:
        normalized = self.normalizer.normalize(text)
        if not normalized:
            raise ValueError("Valtec text became empty after normalization")
        return self.g2p.convert(normalized)
