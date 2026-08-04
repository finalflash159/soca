from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TTS_MODELS_ROOT = REPO_ROOT / "models" / "tts"


@dataclass(frozen=True)
class ValtecTTSConfig:
    key: str = "valtec_multispeaker"
    display_name: str = "Valtec multi-speaker Vietnamese TTS"
    runner: str = "valtec_onnx"
    default_voice: str = "NF"
    voices: tuple[str, ...] = ("NF", "SF", "NM1", "SM", "NM2")
    # The release artifact is calibrated at 1.0.  The application preset is
    # faster while keeping Valtec's adaptive anti-slur pacing active. Caption
    # reveal timing uses each synthesized chunk's actual audio duration.
    length_scale: float = 0.85
    license: str = "CC BY-NC 2.0 (HF valtecAI-team/valtec-tts-pretrained, verified 2026-07-23)"
    source_url: str = "https://github.com/tronghieuit/valtec-tts"

    @property
    def local_dir(self) -> Path:
        return TTS_MODELS_ROOT / self.key


VALTEC_TTS_CONFIG = ValtecTTSConfig()
