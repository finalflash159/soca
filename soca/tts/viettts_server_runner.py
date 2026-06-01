from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .audio_utils import build_result, empty_result, wav_bytes_to_float32_mono
from .base import TTSResult
from .errors import TTSRuntimeUnavailableError
from .registry import TTSModelConfig


class VietTTSServerTTS:
    ENGINE_NAME = "viet-tts-server"
    DEFAULT_BASE_URL = "http://127.0.0.1:8298"

    def __init__(
        self,
        config: TTSModelConfig,
        *,
        voice: str | None = None,
        lazy: bool = True,
    ) -> None:
        self.config = config
        self.voice = voice or config.default_voice
        self.base_url = os.environ.get(config.server_url_env_var or "", self.DEFAULT_BASE_URL).rstrip("/")
        if not lazy:
            self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        return

    def _request(self, path: str, *, data: dict[str, Any] | None = None) -> bytes:
        url = f"{self.base_url}{path}"
        headers = {"Authorization": "Bearer viet-tts"}
        payload = None
        if data is not None:
            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(url, data=payload, headers=headers, method="POST" if data else "GET")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except urllib.error.URLError as exc:
            raise TTSRuntimeUnavailableError(
                f"VietTTS server is not reachable at {self.base_url}. Start it with: "
                "viettts server --host 0.0.0.0 --port 8298"
            ) from exc

    @staticmethod
    def _parse_voice_payload(payload: bytes) -> list[str]:
        data = json.loads(payload.decode("utf-8"))
        if isinstance(data, dict):
            for key in ("data", "voices", "items"):
                if key in data:
                    data = data[key]
                    break

        voices: list[str] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    voices.append(item)
                elif isinstance(item, dict):
                    value = item.get("id") or item.get("voice") or item.get("name")
                    if value is not None:
                        voices.append(str(value))
        return voices

    def list_voices(self) -> list[str]:
        payload = self._request("/v1/voices")
        return self._parse_voice_payload(payload) or [self.config.default_voice]

    def synthesize(self, text: str, voice: str | None = None) -> TTSResult:
        text_clean = text.strip()
        selected_voice = voice or self.voice
        if not text_clean:
            return empty_result(text_clean, selected_voice, self.ENGINE_NAME, 24000)

        t0 = time.perf_counter()
        payload = self._request(
            "/v1/audio/speech",
            data={
                "model": "tts-1",
                "voice": selected_voice,
                "input": text_clean,
                "speed": 1.0,
                "response_format": "wav",
            },
        )
        audio, sample_rate = wav_bytes_to_float32_mono(payload)
        return build_result(
            text=text_clean,
            audio=audio,
            sample_rate=sample_rate,
            started_at=t0,
            voice=selected_voice,
            engine=self.ENGINE_NAME,
        )

