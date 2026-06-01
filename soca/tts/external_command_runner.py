from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

import soundfile as sf

from .audio_utils import build_result, empty_result
from .base import TTSResult
from .errors import TTSRuntimeUnavailableError
from .registry import TTSModelConfig


class ExternalCommandTTS:
    ENGINE_NAME = "external-command-tts"

    def __init__(
        self,
        config: TTSModelConfig,
        *,
        voice: str | None = None,
        lazy: bool = True,
    ) -> None:
        self.config = config
        self.voice = voice or config.default_voice
        if not lazy:
            self._ensure_loaded()

    def _command_template(self) -> str:
        command_env_var = self.config.command_env_var
        command = os.environ.get(command_env_var or "")
        if not command:
            raise TTSRuntimeUnavailableError(
                f"{self.config.key} uses an external command runner. Set {command_env_var} "
                "to a command template that writes WAV to {output}. Available placeholders: "
                "{text}, {voice}, {output}."
            )
        return command

    def _ensure_loaded(self) -> None:
        self._command_template()

    def list_voices(self) -> list[str]:
        configured = os.environ.get(self.config.voices_env_var or "")
        if configured:
            return [item.strip() for item in configured.split(",") if item.strip()]
        return [self.config.default_voice]

    def synthesize(self, text: str, voice: str | None = None) -> TTSResult:
        text_clean = text.strip()
        selected_voice = voice or self.voice
        voices = self.list_voices()
        if selected_voice not in voices:
            raise ValueError(
                f"Unknown external TTS voice {selected_voice!r}. Available voices: {voices}. "
                f"Set {self.config.voices_env_var} to configure voice names."
            )
        if not text_clean:
            return empty_result(text_clean, selected_voice, self.ENGINE_NAME, 24000)

        command_template = self._command_template()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            output_path = handle.name

        command = command_template.format(
            text=shlex.quote(text_clean),
            voice=shlex.quote(selected_voice),
            output=shlex.quote(output_path),
        )
        command_args = shlex.split(command)

        t0 = time.perf_counter()
        try:
            completed = subprocess.run(
                command_args,
                shell=False,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"External TTS command failed with exit code {completed.returncode}: "
                    f"{completed.stderr.strip() or completed.stdout.strip()}"
                )

            audio, sample_rate = sf.read(output_path, dtype="float32", always_2d=False)
            return build_result(
                text=text_clean,
                audio=audio,
                sample_rate=int(sample_rate),
                started_at=t0,
                voice=selected_voice,
                engine=f"{self.ENGINE_NAME}:{self.config.key}",
            )
        finally:
            Path(output_path).unlink(missing_ok=True)
