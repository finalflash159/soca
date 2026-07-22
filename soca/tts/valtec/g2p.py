from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ValtecOnnxArtifacts
from .frontend import ValtecModelInputs
from .normalizer import ValtecTextNormalizer

TABLE_PATH = Path(__file__).with_name("g2p_tables.json")
TONE_MAP = {1: 0, 2: 2, 3: 3, 4: 4, 5: 1, 6: 5}
PUNCTUATION = set(",.!?;:'\"()[]{}")
MODIFIERS = {"ʷ", "ʰ", "ː"}


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

    def _tokenize_ipa(self, ipa: str) -> tuple[list[int], int]:
        ids: list[int] = []
        unknown = 0
        index = 0
        while index < len(ipa):
            if unicodedata.combining(ipa[index]):
                index += 1
                continue
            for width in range(min(4, len(ipa) - index), 0, -1):
                symbol = ipa[index : index + width]
                if symbol in self.symbol_to_id:
                    ids.append(self.symbol_to_id[symbol])
                    index += width
                    break
            else:
                if ipa[index] not in MODIFIERS:
                    ids.append(self.symbol_to_id["UNK"])
                    unknown += 1
                index += 1
        return ids, unknown

    def convert(self, text: str) -> ValtecModelInputs:
        phone_ids: list[int] = [self.symbol_to_id["_"]]
        tone_ids: list[int] = [self.tone_offset]
        language_ids: list[int] = [self.language_id]
        unknown_count = 0
        tokens = re.findall(r"[^\W\d_]+|[,\.!?;:'\"()\[\]{}]", text, flags=re.UNICODE)
        for token in tokens:
            if token in PUNCTUATION:
                known = token in self.symbol_to_id
                phone_ids.append(self.symbol_to_id.get(token, self.symbol_to_id["UNK"]))
                tone_ids.append(self.tone_offset)
                language_ids.append(self.language_id)
                unknown_count += int(not known)
                continue
            syllable = self._transcribe(token)
            internal_tone = TONE_MAP.get(syllable.tone, 0)
            if syllable.oov:
                ids, unknown = [self.symbol_to_id["UNK"]], 1
            else:
                ids, unknown = self._tokenize_ipa(
                    f"{syllable.onset}{syllable.nucleus}{syllable.coda}"
                )
            phone_ids.extend(ids)
            tone_ids.extend([internal_tone + self.tone_offset] * len(ids))
            language_ids.extend([self.language_id] * len(ids))
            unknown_count += unknown

        phone_ids.append(self.symbol_to_id["_"])
        tone_ids.append(self.tone_offset)
        language_ids.append(self.language_id)
        if self.add_blank:
            phones_with_blank: list[int] = []
            tones_with_blank: list[int] = []
            languages_with_blank: list[int] = []
            for phone, tone, language in zip(phone_ids, tone_ids, language_ids, strict=True):
                phones_with_blank.extend((0, phone))
                tones_with_blank.extend((0, tone))
                languages_with_blank.extend((self.language_id, language))
            phone_ids = [*phones_with_blank, 0]
            tone_ids = [*tones_with_blank, 0]
            language_ids = [*languages_with_blank, self.language_id]
        return ValtecModelInputs(
            phone_ids=tuple(phone_ids),
            tone_ids=tuple(tone_ids),
            language_ids=tuple(language_ids),
            backend="portable_web_port",
            unknown_phoneme_count=unknown_count,
        )


class ValtecVietnameseFrontend:
    def __init__(self, normalizer: ValtecTextNormalizer, g2p: PortableVietnameseG2P) -> None:
        self.normalizer = normalizer
        self.g2p = g2p

    @classmethod
    def from_artifacts(cls, artifacts: ValtecOnnxArtifacts) -> "ValtecVietnameseFrontend":
        config = json.loads(artifacts.config.read_text(encoding="utf-8"))
        symbol_to_id = config.get("symbol_to_id")
        if not isinstance(symbol_to_id, dict):
            raise ValueError("Valtec tts_config.json is missing symbol_to_id")
        configured_language = config.get("language_id_map", {}).get("VI")
        if configured_language is not None and int(configured_language) != artifacts.language_id_vi:
            raise ValueError("Valtec manifest/config Vietnamese language id mismatch")
        return cls(
            ValtecTextNormalizer(),
            PortableVietnameseG2P(
                symbol_to_id=symbol_to_id,
                language_id=artifacts.language_id_vi,
                tone_offset=artifacts.tone_offset_vi,
                add_blank=artifacts.add_blank,
            ),
        )

    def prepare(self, text: str) -> ValtecModelInputs:
        normalized = self.normalizer.normalize(text)
        if not normalized:
            raise ValueError("Valtec text became empty after normalization")
        return self.g2p.convert(normalized)