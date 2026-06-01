from __future__ import annotations

import json
import os
import re
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .audio_utils import build_result, empty_result
from .base import TTSResult
from .errors import TTSRuntimeUnavailableError
from .registry import TTSModelConfig

VOICE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,63}$")
AUTO_VOICE = "auto"


class OmniVoiceTTS:
    ENGINE_NAME = "omnivoice"

    def __init__(
        self,
        config: TTSModelConfig,
        *,
        voice: str | None = None,
        lazy: bool = True,
    ) -> None:
        self.config = config
        self.voice = voice or config.default_voice
        self._model: Any | None = None
        self._voice_prompt_cache: dict[str, Any] = {}
        if not lazy:
            self._ensure_loaded()

    def _device_and_dtype(self) -> tuple[str, Any]:
        try:
            import torch
        except ImportError as exc:
            raise TTSRuntimeUnavailableError(
                "OmniVoice requires torch and omnivoice. Install with: "
                "uv pip install omnivoice"
            ) from exc

        if torch.cuda.is_available():
            return "cuda", torch.float16
        if torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            from omnivoice import OmniVoice
        except ImportError as exc:
            raise TTSRuntimeUnavailableError(
                "OmniVoice requires the omnivoice package. Install with: "
                "uv pip install omnivoice"
            ) from exc

        if not self.config.hf_repo:
            raise RuntimeError(f"Missing OmniVoice repo for {self.config.key}")

        device, dtype = self._device_and_dtype()
        try:
            self._model = OmniVoice.from_pretrained(
                self.config.hf_repo,
                device_map=device,
                dtype=dtype,
            )
        except Exception as exc:
            raise TTSRuntimeUnavailableError(
                f"OmniVoice runtime could not load {self.config.key}: {exc}"
            ) from exc

    def list_voices(self) -> list[str]:
        return list(
            dict.fromkeys(
                [
                    self.config.default_voice,
                    AUTO_VOICE,
                    *self._voice_designs(),
                    *self._saved_voice_ids(),
                ]
            )
        )

    @property
    def voice_bank_dir(self) -> Path:
        return self.config.local_dir / "voices"

    def _voice_dir(self, voice_id: str) -> Path:
        return self.voice_bank_dir / voice_id

    def _saved_voice_ids(self) -> list[str]:
        if not self.voice_bank_dir.exists():
            return []
        return sorted(
            path.name
            for path in self.voice_bank_dir.iterdir()
            if path.is_dir() and (path / "prompt.pt").exists() and (path / "meta.json").exists()
        )

    def _voice_designs(self) -> dict[str, str]:
        designs = dict(self.config.voice_designs or {})
        configured = os.environ.get(self.config.voices_env_var or "")
        if not configured:
            return designs

        configured = configured.strip()
        if configured.startswith("{"):
            try:
                payload = json.loads(configured)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid {self.config.voices_env_var} JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{self.config.voices_env_var} JSON must be an object.")
            designs.update({str(key): str(value) for key, value in payload.items() if str(key).strip()})
            return designs

        for item in configured.split(";"):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                alias, prompt = item.split("=", 1)
                designs[alias.strip()] = prompt.strip()
            else:
                designs[item] = item
        return {alias: prompt for alias, prompt in designs.items() if alias and prompt}

    def _validate_voice_id(self, voice_id: str) -> str:
        voice_id = voice_id.strip()
        if not VOICE_ID_RE.fullmatch(voice_id):
            raise ValueError(
                "Voice id must be 2-64 characters and contain only letters, numbers, "
                "underscore, or hyphen."
            )
        reserved = {self.config.default_voice, AUTO_VOICE, *self._voice_designs()}
        if voice_id in reserved:
            raise ValueError(f"Voice id {voice_id!r} is reserved by a built-in OmniVoice voice.")
        return voice_id

    def _instruction_for_voice(self, selected_voice: str) -> str | None:
        if selected_voice in {self.config.default_voice, AUTO_VOICE}:
            return None

        designs = self._voice_designs()
        if selected_voice not in designs:
            voices = [self.config.default_voice, *designs]
            raise ValueError(
                f"Unknown OmniVoice voice design {selected_voice!r}. Available voices: {voices}. "
                f"Set {self.config.voices_env_var} as JSON, for example: "
                """'{"female_young": "female, young adult, moderate pitch"}'"""
            )
        return designs[selected_voice]

    def _load_voice_clone_prompt(self, voice_id: str) -> Any:
        if voice_id in self._voice_prompt_cache:
            return self._voice_prompt_cache[voice_id]

        prompt_path = self._voice_dir(voice_id) / "prompt.pt"
        if not prompt_path.exists():
            return None

        try:
            import torch
            from omnivoice.models.omnivoice import VoiceClonePrompt
        except ImportError as exc:
            raise TTSRuntimeUnavailableError(
                "Saved OmniVoice voices require torch and omnivoice. Install with: "
                "uv pip install omnivoice"
            ) from exc

        payload = torch.load(prompt_path, map_location="cpu", weights_only=True)
        prompt = VoiceClonePrompt(
            ref_audio_tokens=payload["ref_audio_tokens"],
            ref_text=str(payload["ref_text"]),
            ref_rms=float(payload["ref_rms"]),
        )
        self._voice_prompt_cache[voice_id] = prompt
        return prompt

    def freeze_voice(
        self,
        *,
        voice_id: str,
        ref_audio: Path,
        ref_text: str,
        source_voice: str | None = None,
        source_sample: Path | None = None,
        overwrite: bool = False,
    ) -> Path:
        voice_id = self._validate_voice_id(voice_id)
        ref_audio = ref_audio.expanduser().resolve()
        if not ref_audio.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_audio}")
        ref_text = ref_text.strip()
        if not ref_text:
            raise ValueError("Reference text must not be empty.")

        voice_dir = self._voice_dir(voice_id)
        if voice_dir.exists() and not overwrite:
            raise FileExistsError(f"Saved OmniVoice voice already exists: {voice_dir}")

        voice_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_loaded()

        prompt = self._model.create_voice_clone_prompt(
            ref_audio=str(ref_audio),
            ref_text=ref_text,
        )

        try:
            import torch
        except ImportError as exc:
            raise TTSRuntimeUnavailableError(
                "Saving OmniVoice voice prompts requires torch. Install with: uv pip install omnivoice"
            ) from exc

        torch.save(
            {
                "ref_audio_tokens": prompt.ref_audio_tokens.detach().cpu(),
                "ref_text": prompt.ref_text,
                "ref_rms": float(prompt.ref_rms),
            },
            voice_dir / "prompt.pt",
        )
        shutil.copy2(ref_audio, voice_dir / "ref.wav")
        metadata = {
            "voice_id": voice_id,
            "engine": self.ENGINE_NAME,
            "model_key": self.config.key,
            "hf_repo": self.config.hf_repo,
            "source_voice": source_voice,
            "source_sample": str(source_sample.resolve()) if source_sample else str(ref_audio),
            "ref_audio": "ref.wav",
            "ref_text": ref_text,
            "created_at": datetime.now(UTC).isoformat(),
        }
        (voice_dir / "meta.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return voice_dir

    def synthesize(self, text: str, voice: str | None = None) -> TTSResult:
        text_clean = text.strip()
        selected_voice = voice or self.voice
        if not text_clean:
            return empty_result(text_clean, selected_voice, self.ENGINE_NAME, 24000)

        self._ensure_loaded()
        voice_clone_prompt = self._load_voice_clone_prompt(selected_voice)
        instruct = None if voice_clone_prompt is not None else self._instruction_for_voice(selected_voice)
        t0 = time.perf_counter()
        audio_items = self._model.generate(
            text=text_clean,
            language="vi",
            instruct=instruct,
            voice_clone_prompt=voice_clone_prompt,
        )
        sample_rate = int(getattr(self._model, "sampling_rate", 24000))
        return build_result(
            text=text_clean,
            audio=audio_items[0],
            sample_rate=sample_rate,
            started_at=t0,
            voice=selected_voice,
            engine=self.ENGINE_NAME,
        )
