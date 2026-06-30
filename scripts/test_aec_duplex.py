"""Path B prototype — persistent duplex stream + AEC + VAD + barge-in.

Proves the Path B mechanics end-to-end BEFORE touching pipeline.py: one persistent
`sd.Stream` plays a tone (stand-in for TTS = far) while capturing the mic (near),
runs AEC inline, then Silero VAD on the cleaned audio and the same barge-in rule
as the real listener.

Run with the headset that misfires:
    .venv/bin/python scripts/test_aec_duplex.py
    SOCA_AEC_DELAY_MS=40 .venv/bin/python scripts/test_aec_duplex.py

Tone plays for ~6s. Expected if Path B works:
    - You stay SILENT  -> prob stays LOW (AEC kills the tone), NO interrupt.
    - You SPEAK        -> prob jumps, run climbs to 600ms -> "INTERRUPT".
If it interrupts while you are silent, the tone echo is leaking through -> tune
SOCA_AEC_DELAY_MS. If it never fires when you speak, threshold/AEC is too aggressive.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import sounddevice as sd
import torch
from pywebrtc_audio import AudioProcessor
from silero_vad import load_silero_vad

RATE = 16000
FRAME = 512  # 32 ms == Silero VAD frame
BLOCK_MS = 32
THRESHOLD = float(os.environ.get("SOCA_BARGE_THRESHOLD", 0.7))
SUSTAINED_MS = float(os.environ.get("SOCA_BARGE_SUSTAINED_MS", 200))
SECONDS = 6


def main() -> None:
    model = load_silero_vad()
    model.reset_states()
    aec = AudioProcessor(
        sample_rate=RATE,
        num_channels=1,
        echo_cancellation=True,
        noise_suppression=True,
        auto_gain_control=False,
        stream_delay_ms=int(os.environ.get("SOCA_AEC_DELAY_MS", 40)),
    )

    # Far = a 330 Hz tone (stand-in for SoCa's voice) for the whole run.
    n_total = RATE * SECONDS
    t = np.arange(n_total) / RATE
    far_all = (0.25 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)

    print(f"delay={aec.stream_delay_ms}ms threshold={THRESHOLD} sustained={SUSTAINED_MS}ms")
    print("Tone đang phát. NGỒI IM vài giây (không được fire), rồi NÓI (phải fire).\n")

    run_ms = 0.0
    pos = 0
    with sd.Stream(samplerate=RATE, blocksize=FRAME, channels=1, dtype="float32") as stream:
        while pos < n_total:
            far = far_all[pos : pos + FRAME]
            pos += FRAME
            if len(far) < FRAME:
                far = np.concatenate([far, np.zeros(FRAME - len(far), np.float32)])

            stream.write(far.reshape(-1, 1))        # phát far (TTS giả)
            near, _ = stream.read(FRAME)            # thu mic (giọng + echo)
            near = near.flatten().astype(np.float32)
            clean = aec.process(near, far)          # khử echo (far/near đồng bộ duplex)

            prob = float(model(torch.from_numpy(clean), RATE).item())
            is_speech = prob >= THRESHOLD
            run_ms = run_ms + BLOCK_MS if is_speech else 0.0
            if is_speech:
                print(f"prob={prob:4.2f} run={run_ms:5.0f}/{SUSTAINED_MS:.0f}ms",
                      file=sys.stderr, flush=True)
            if run_ms >= SUSTAINED_MS:
                print(f"-> INTERRUPT (prob={prob:.2f}) at t={pos/RATE:.1f}s", file=sys.stderr, flush=True)
                return

    print("Hết tone, KHÔNG interrupt — nếu bạn có nói mà không fire thì hạ threshold/sustained.")


if __name__ == "__main__":
    main()
