from __future__ import annotations

import os
import sys
from collections import deque

import numpy as np
import sounddevice as sd
import torch
from pywebrtc_audio import AudioProcessor
from silero_vad import load_silero_vad

from soca.core.aec_reference import AECReference

# Silero VAD v5 requires exactly this many samples per call at 16 kHz.
_SILERO_FRAME_SAMPLES = 512
_SILERO_SAMPLE_RATE = 16000

# How much trailing audio to keep so the interrupting words are not lost. The
# interrupt fires after ~sustained_ms of speech, so this only needs to hold that
# head of the utterance (the rest is recorded afterwards by record_until_silence).
_CAPTURE_WINDOW_MS = 1200

# Opt-in debug: `SOCA_BARGE_DEBUG=1 soca voice` prints the speech probability the
# mic feeds the VAD (to stderr). Lets you see whether prob spikes coincide with
# SoCa speaking (echo via a headset mic) or appear at random (ambient noise).
_DEBUG = os.environ.get("SOCA_BARGE_DEBUG", "") not in ("", "0")


def update_speech_run(run_ms: float, is_speech: bool, block_ms: float) -> float:
    """Cumulative ms of speech detected in a row; reset to 0 on silence."""
    return run_ms + block_ms if is_speech else 0.0


def should_interrupt(run_ms: float, sustained_ms: float) -> bool:
    return run_ms >= sustained_ms


class BargeInListener:
    """Listen to the mic during playback; set ``interrupt_event`` on sustained speech.

    Owns its OWN Silero VAD model instead of borrowing the pipeline's
    ``SpeechDetector``: the model is stateful and not thread-safe, so sharing it
    between the recorder/ASR and this listener thread would corrupt its internal
    state. Create one listener and reuse it across turns (the model loads once);
    ``run`` resets the model state each turn.

    Phase 1 (headphones, no echo) = layers L2 (high VAD threshold) + L3 (sustained
    window). AEC / SNR / backchannel are later phases.

    Thresholds can be tuned live without editing code:
        SOCA_BARGE_THRESHOLD=0.8 SOCA_BARGE_SUSTAINED_MS=700 soca voice
    """

    def __init__(
        self,
        *,
        sample_rate: int = _SILERO_SAMPLE_RATE,
        block_ms: int = 32,
        sustained_ms: float = 600,
        vad_threshold: float = 0.7,
        enable_aec: bool = False,
    ) -> None:
        self.model = load_silero_vad()
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        # Env overrides win so you can tune during a live 7-C session.
        self.sustained_ms = float(os.environ.get("SOCA_BARGE_SUSTAINED_MS", sustained_ms))
        self.vad_threshold = float(os.environ.get("SOCA_BARGE_THRESHOLD", vad_threshold))
        # Audio that triggered the most recent interrupt; the caller prepends it to
        # the next recording so the user's first words survive. None until an
        # interrupt fires, and reset at the start of every ``run``.
        self.captured: np.ndarray | None = None
        self.reference = None
        self._aec = None

        if enable_aec:
            self.reference = AECReference()
            self._aec = AudioProcessor(
                sample_rate= self.sample_rate,
                num_channels=1,
                echo_cancellation=True,
                noise_suppression=True,
                auto_gain_control=False,
                # Delay loa→mic. Tune live: SOCA_AEC_DELAY_MS=150 soca voice
                stream_delay_ms=int(os.environ.get("SOCA_AEC_DELAY_MS", 80)),
            )

    def run(self, interrupt_event, stop_event) -> None:
        n = int(self.sample_rate * self.block_ms / 1000)
        assert n == _SILERO_FRAME_SAMPLES, (
            f"Silero VAD needs exactly {_SILERO_FRAME_SAMPLES} samples "
            f"@{_SILERO_SAMPLE_RATE} Hz; block_ms={self.block_ms} gives {n}."
        )

        self.captured = None  # clear any carry-over from a previous turn

        if self.reference is not None:
            self.reference.clear()

        self.model.reset_states()  # fresh streaming context for this turn
        run_ms = 0.0
        # Rolling buffer of recent mic audio. The listener consumes the mic that
        # the recorder would otherwise read, so we keep the last ~window of audio
        # to hand back as the head of the interrupting utterance.
        window: deque[np.ndarray] = deque(maxlen=max(1, int(_CAPTURE_WINDOW_MS / self.block_ms)))
        with sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32", blocksize=n
        ) as stream:
            while not stop_event.is_set() and not interrupt_event.is_set():
                try:
                    block, _ = stream.read(n)
                    # Copy: the stream reuses its internal buffer on the next read,
                    # so a view would be overwritten before we use it.
                    mono = block.reshape(-1)[:n].astype("float32", copy=True)
                    if self._aec is not None and self.reference is not None:
                        far = self.reference.pull(n)
                        mono = self._aec.process(mono, far)
                    prob = float(self.model(torch.from_numpy(mono), self.sample_rate).item())
                except Exception:
                    # Transient read/model hiccup: skip this block and keep
                    # listening so the thread never dies silently.
                    continue

                window.append(mono)
                is_speech = prob >= self.vad_threshold
                run_ms = update_speech_run(run_ms, is_speech, self.block_ms)
                if _DEBUG and is_speech:
                    print(
                        f"[barge-in] prob={prob:4.2f} run={run_ms:5.0f}/{self.sustained_ms:.0f}ms",
                        file=sys.stderr,
                        flush=True,
                    )
                if should_interrupt(run_ms, self.sustained_ms):
                    self.captured = np.concatenate(list(window)).astype("float32", copy=False)
                    if _DEBUG:
                        print(f"[barge-in] -> INTERRUPT (prob={prob:.2f})", file=sys.stderr, flush=True)
                    interrupt_event.set()
                    return
