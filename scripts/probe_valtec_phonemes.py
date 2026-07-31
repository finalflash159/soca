"""Render OOV English words through the Valtec ONNX pipeline under three
phonemization variants, for the manual audio review required before
`foreign_g2p: "g2p_en"` can become the default (see
zplan/tts_foreign_g2p_fix_plan.vi.md §6).

  A - current production behaviour (no foreign backend): OOV words are
      spelled out letter by letter.
  B - the approach this plan ships: curated lexicon first, then g2p_en
      ARPABET converted through eng_to_ipa's own dialect table, gated to the
      trained inventory. This is the exact chain from_artifacts() builds.
  C - contrast only: the same ARPABET mapped to "textbook" IPA symbols
      (r->r turned voiced uvular etc.) that measured as untrained on this
      checkpoint. Exists to make the untrained-embedding failure audible;
      never shipped in soca/.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from soca.tts.valtec import (
    PortableVietnameseG2P,
    ValtecOnnxTTS,
    ValtecVietnameseFrontend,
    resolve_valtec_onnx_artifacts,
)
from soca.tts.valtec.foreign_g2p import ChainedForeignG2P
from soca.tts.valtec.foreign_g2p_en import G2pEnBackend, _load_g2p
from soca.tts.valtec.lexicon import LexiconBackend
from soca.tts.valtec.normalizer import ValtecTextNormalizer

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = REPO_ROOT / "models" / "tts" / "valtec_multispeaker" / "reference" / "upstream"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "notebooks" / "outputs" / "g2p_probe"

DEFAULT_WORDS = (
    "embedding", "github", "pytorch", "zalo", "openai", "naruto", "frontend",
    "tokenizer", "anthropic", "huggingface", "kubernetes", "chatbot",
    "softmax", "encoder", "onnxruntime", "numpy", "pypi", "readme",
    "pytest", "typescript", "webpack", "postgresql",
)

# Contrast backend only (variant C): the ARPABET->IPA choices v1 originally
# made (R->r turned uvular, G->U+0261, AH->ʌ, CH->tʃ, JH->dʒ, ER->ɚ), every
# one of which measured as an untrained embedding row (plan §2.4/§2.6).
_NAIVE_ARPABET_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɚ",
    "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
}


class _NaiveTextbookIpaBackend:
    """Contrast-only ForeignG2P for column C. Deliberately ungated."""

    def to_ipa(self, token: str) -> str | None:
        g2p = _load_g2p()
        if g2p is None:
            return None
        try:
            phones = list(g2p(token.lower()))
        except Exception:
            return None
        arpabet = [re.sub(r"\d", "", phone.upper()) for phone in phones if phone and phone[0].isalpha()]
        ipa = "".join(_NAIVE_ARPABET_TO_IPA.get(phone, "") for phone in arpabet)
        return ipa or None


def _build_frontend(
    symbol_to_id: dict[str, int], artifacts: Any, foreign_g2p: Any, overrides: dict[str, str]
) -> ValtecVietnameseFrontend:
    g2p = PortableVietnameseG2P(
        symbol_to_id=symbol_to_id,
        language_id=artifacts.language_id_vi,
        tone_offset=artifacts.tone_offset_vi,
        add_blank=artifacts.add_blank,
        foreign_g2p=foreign_g2p,
        pronunciation_overrides=overrides,
    )
    return ValtecVietnameseFrontend(ValtecTextNormalizer(), g2p)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--word", action="append", default=[], dest="words")
    parser.add_argument("--voice", default="NF")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--artifact-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    import soundfile as sf

    args = build_parser().parse_args()
    words = tuple(args.words) or DEFAULT_WORDS
    args.output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = resolve_valtec_onnx_artifacts(args.artifact_root, allow_reference=True)
    config = json.loads(artifacts.config.read_text(encoding="utf-8"))
    symbol_to_id = config["symbol_to_id"]

    # Same composition from_artifacts() builds for production.
    from soca.tts.valtec.lexicon import CMU_OVERRIDE_LEXICON

    shipped_backend = ChainedForeignG2P((LexiconBackend(), G2pEnBackend()))
    naive_backend = _NaiveTextbookIpaBackend()
    variants = {
        "A_current_spelling": (None, {}, "letter-spelled (current production behaviour)"),
        "B_trained_dialect": (shipped_backend, CMU_OVERRIDE_LEXICON, shipped_backend.to_ipa),
        "C_naive_ipa": (naive_backend, {}, naive_backend.to_ipa),
    }

    report: list[dict[str, Any]] = []
    for suffix, (backend, overrides, ipa_source) in variants.items():
        frontend = _build_frontend(symbol_to_id, artifacts, backend, overrides)
        engine = ValtecOnnxTTS(
            artifact_root=args.artifact_root,
            allow_reference=artifacts.role == "reference",
            allow_candidate=artifacts.role == "candidate",
            frontend=frontend,
            seed=args.seed,
            sentence_chunking=False,
        )
        for word in words:
            result = engine.synthesize(word, voice=args.voice)
            wav_path = args.output_dir / f"{word}_{suffix}.wav"
            sf.write(wav_path, result.audio, result.sample_rate)
            ipa = ipa_source(word) if callable(ipa_source) else ipa_source
            report.append(
                {
                    "word": word,
                    "variant": suffix,
                    "ipa": ipa,
                    "wav": wav_path.name,
                    "audio_duration_ms": round(result.audio_duration_ms, 1),
                    "backend": engine.frontend_metadata["backend"],
                    "unknown_phoneme_count": engine.frontend_metadata["unknown_phoneme_count"],
                }
            )
            print(f"{word:16s} {suffix:20s} ipa={ipa!r:20s} {result.audio_duration_ms:7.1f} ms")

    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
