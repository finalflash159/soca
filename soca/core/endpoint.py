from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sounddevice as sd


@dataclass(frozen=True)
class EndpointConfig:
    sample_rate: int = 16000
    block_ms: int = 100
    endpoint_silence_ms: int = 700  # user silent for 0.7s -> end of turn
    max_record_ms: int = 10000
    min_audio_ms: int = 300


def should_stop_recording(
    speech_timestamps: list[dict[str, int]],
    total_samples: int,
    sample_rate: int,
    endpoint_silence_ms: int,
) -> bool:
    if not speech_timestamps:
        return False

    last_speech_end = speech_timestamps[-1]["end"]
    silence_samples = total_samples - last_speech_end
    silence_ms = silence_samples / sample_rate * 1000
    return silence_ms >= endpoint_silence_ms


def block_samples(config: EndpointConfig) -> int:
    return int(config.sample_rate * config.block_ms / 1000)


def record_until_silence(
    detector,
    config: EndpointConfig | None = None,
) -> np.ndarray:
    config = config or EndpointConfig()
    n_block_samples = block_samples(config)
    max_samples = int(config.sample_rate * config.max_record_ms / 1000)

    chunks: list[np.ndarray] = []
    has_seen_speech = False
    total_samples = 0

    with sd.InputStream(samplerate=config.sample_rate, channels=1, dtype="float32") as stream:
        while total_samples < max_samples:
            block, overflowed = stream.read(n_block_samples)
            if overflowed:
                # Keep recording; occasional should not kill the demo
                pass

            block_mono = np.asarray(block, dtype=np.float32).reshape(-1)
            chunks.append(block_mono)
            total_samples += len(block_mono)

            audio = np.concatenate(chunks).astype(np.float32, copy=False)

            min_samples = int(config.sample_rate * config.min_audio_ms / 1000)
            if len(audio) < min_samples:
                continue

            timestamps = detector.speech_timestamps(audio)
            if timestamps:
                has_seen_speech = True

            if has_seen_speech and should_stop_recording(
                timestamps,
                total_samples=len(audio),
                sample_rate=config.sample_rate,
                endpoint_silence_ms=config.endpoint_silence_ms,
            ):
                break

    if not chunks:
        return np.array([], dtype=np.float32)

    return np.concatenate(chunks).astype(np.float32, copy=False)
