from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from .audio_utils import build_result, empty_result
from .base import TTSResult
from .errors import TTSRuntimeUnavailableError
from .registry import TTSModelConfig


class F5VietnameseTTS:
    ENGINE_NAME = "f5-tts"

    def __init__(
        self,
        config: TTSModelConfig,
        *,
        voice: str | None = None,
    ) -> None:
        self.config = config
        self.voice = voice or config.default_voice
        self._model: Any | None = None
        self._ensure_loaded()

    def _reference(self) -> tuple[str, str]:
        audio_env = self.config.reference_audio_env_var
        text_env = self.config.reference_text_env_var
        ref_audio = os.environ.get(audio_env or "")
        ref_text = os.environ.get(text_env or "")
        if not ref_audio or not ref_text:
            raise TTSRuntimeUnavailableError(
                f"{self.config.key} requires reference audio/text env vars: "
                f"{audio_env}=<wav path> and {text_env}=<transcript>"
            )
        ref_path = Path(ref_audio).expanduser()
        if not ref_path.exists():
            raise TTSRuntimeUnavailableError(f"F5 reference audio not found: {ref_path}")
        return str(ref_path), ref_text

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        self._reference()
        try:
            from f5_tts.api import F5TTS
        except ImportError as exc:
            raise TTSRuntimeUnavailableError(
                "F5 Vietnamese TTS requires f5-tts. Install in a compatible env with: "
                "uv sync --extra tts-f5"
            ) from exc

        if not self.config.hf_repo or not self.config.hf_model_file:
            raise RuntimeError(f"Missing F5 artifact metadata for {self.config.key}")

        ckpt_file = hf_hub_download(
            repo_id=self.config.hf_repo,
            filename=self.config.hf_model_file,
            local_dir=self.config.local_dir,
        )
        vocab_file = ""
        if self.config.hf_config_file:
            vocab_file = hf_hub_download(
                repo_id=self.config.hf_repo,
                filename=self.config.hf_config_file,
                local_dir=self.config.local_dir,
            )

        try:
            self._model = F5TTS(
                model=self.config.sdk_model or "F5TTS_Base",
                ckpt_file=ckpt_file,
                vocab_file=vocab_file,
            )
        except Exception as exc:
            raise TTSRuntimeUnavailableError(
                f"F5 runtime could not load {self.config.key}: {exc}"
            ) from exc

    def list_voices(self) -> list[str]:
        return [self.config.default_voice]

    def synthesize(self, text: str, voice: str | None = None) -> TTSResult:
        text_clean = text.strip()
        selected_voice = voice or self.voice
        if selected_voice != self.config.default_voice:
            raise ValueError(
                f"{self.config.key} exposes one configured reference voice: "
                f"{self.config.default_voice!r}"
            )
        if not text_clean:
            return empty_result(text_clean, selected_voice, self.ENGINE_NAME, 24000)

        ref_audio, ref_text = self._reference()
        self._ensure_loaded()

        t0 = time.perf_counter()
        wav, sample_rate, _ = self._model.infer(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=text_clean,
            show_info=lambda *_args, **_kwargs: None,
            progress=None,
        )
        return build_result(
            text=text_clean,
            audio=wav,
            sample_rate=int(sample_rate),
            started_at=t0,
            voice=selected_voice,
            engine=self.ENGINE_NAME,
        )
