from .base import TTSEngine
from .config import VALTEC_TTS_CONFIG
from .valtec import (
    ValtecOnnxTTS,
    ValtecVietnameseFrontend,
    resolve_current_valtec_release,
    resolve_valtec_onnx_artifacts,
)


def create_tts_engine(*, voice: str | None = None) -> TTSEngine:
    release_root = resolve_current_valtec_release()
    artifacts = resolve_valtec_onnx_artifacts(release_root)
    return ValtecOnnxTTS(
        artifact_root=release_root,
        frontend=ValtecVietnameseFrontend.from_artifacts(artifacts),
        voice=voice or VALTEC_TTS_CONFIG.default_voice,
    )
