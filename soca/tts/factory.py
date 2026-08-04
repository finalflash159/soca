from .base import TTSEngine
from .config import VALTEC_TTS_CONFIG
from .errors import TTSRuntimeUnavailableError
from .valtec import (
    ValtecOnnxTTS,
    ValtecVietnameseFrontend,
    resolve_current_valtec_release,
    resolve_valtec_onnx_artifacts,
)


def create_tts_engine(*, voice: str | None = None) -> TTSEngine:
    # Artifact resolution and manifest validation raise plain builtins
    # (FileNotFoundError/KeyError/ValueError). Wrap them in the runtime-domain
    # error so eval/smoke tooling that catches TTSRuntimeUnavailableError can
    # skip gracefully instead of crashing when a release is missing/malformed.
    try:
        release_root = resolve_current_valtec_release()
        artifacts = resolve_valtec_onnx_artifacts(release_root)
        frontend = ValtecVietnameseFrontend.from_artifacts(artifacts)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise TTSRuntimeUnavailableError(f"Valtec artifact is not ready: {exc}") from exc
    return ValtecOnnxTTS(
        artifact_root=release_root,
        frontend=frontend,
        voice=voice or VALTEC_TTS_CONFIG.default_voice,
        length_scale=VALTEC_TTS_CONFIG.length_scale,
    )
