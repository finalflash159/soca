"""Step 0 — Standalone AEC smoke test (validates AEC3 on THIS machine/headset).

Plays a tone through your CURRENT output device while recording the mic, runs
WebRTC AEC3 with PERFECT alignment (single duplex stream → far/near sample-synced,
which the SoCa integration cannot guarantee). This isolates one question:

    "Can AEC3 cancel the echo on my hardware AT ALL?"

How to run (use the SAME headset/device that barge-in misfires on):
    .venv/bin/python scripts/test_aec.py
    SOCA_AEC_DELAY_MS=150 .venv/bin/python scripts/test_aec.py   # sweep delay

Phase 1 (0-3s): STAY SILENT  -> measures echo reduction (raw vs cleaned).
Phase 2 (3-6s): SPEAK        -> listen to aec_clean.wav: voice kept, tone gone?

Verdict:
    >10 dB reduction  -> AEC3 works here; SoCa's problem is integration alignment.
    < 6 dB reduction  -> AEC3 can't tame this echo path; pick a different approach.
"""

from __future__ import annotations

import os

import numpy as np
import sounddevice as sd
import soundfile as sf
from pywebrtc_audio import AudioProcessor

RATE = 16000
FRAME = 160  # 10 ms
SECONDS = 6
SILENT_PHASE_S = 3  # measure echo reduction on this first window (don't speak)

# NS OFF so the only thing that can remove the tone is the echo canceller itself
# (noise suppression could also kill a stationary tone and fake a good result).
ap = AudioProcessor(
    sample_rate=RATE,
    num_channels=1,
    echo_cancellation=True,
    noise_suppression=False,
    auto_gain_control=False,
    stream_delay_ms=int(os.environ.get("SOCA_AEC_DELAY_MS", 40)),
)


def rms_db(x: np.ndarray) -> float:
    return 20.0 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-9)


def main() -> None:
    n_total = RATE * SECONDS
    t = np.arange(n_total) / RATE
    # 330 Hz tone at moderate level = the "SoCa voice" stand-in (far / reference).
    far_all = (0.25 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)

    print(f"stream_delay_ms = {ap.stream_delay_ms}")
    print("Phát tone qua OUTPUT hiện tại (đeo headset đang lỗi vào).")
    print(f"  PHA 1 (0-{SILENT_PHASE_S}s): NGỒI IM.")
    print(f"  PHA 2 ({SILENT_PHASE_S}-{SECONDS}s): NÓI vào mic.\n")

    raw_frames: list[np.ndarray] = []
    clean_frames: list[np.ndarray] = []
    with sd.Stream(samplerate=RATE, blocksize=FRAME, channels=1, dtype="float32") as stream:
        for i in range(0, n_total - FRAME, FRAME):
            far = far_all[i : i + FRAME]
            stream.write(far.reshape(-1, 1))        # phát far ra loa
            near, _ = stream.read(FRAME)            # thu mic (near = giọng + echo)
            near = near.flatten().astype(np.float32)
            clean = ap.process(near, far)           # khử echo (far/near đồng bộ sẵn)
            raw_frames.append(near)
            clean_frames.append(clean)

    raw = np.concatenate(raw_frames)
    clean = np.concatenate(clean_frames)
    sf.write("aec_raw.wav", raw, RATE)
    sf.write("aec_clean.wav", clean, RATE)

    half = RATE * SILENT_PHASE_S
    raw_db = rms_db(raw[:half])
    clean_db = rms_db(clean[:half])
    reduction = raw_db - clean_db

    print(f"Echo-only (PHA 1): raw={raw_db:6.1f} dB | clean={clean_db:6.1f} dB "
          f"| KHỬ = {reduction:4.1f} dB")
    if reduction > 10:
        print("✅ AEC3 KHỬ ĐƯỢC echo trên máy bạn. Vấn đề của SoCa là ALIGN tích hợp → tune được.")
    elif reduction > 6:
        print("🟡 Khử yếu. Thử SOCA_AEC_DELAY_MS khác (80/120/160/200) rồi chạy lại.")
    else:
        print("❌ AEC3 gần như KHÔNG khử. Headset/phần cứng khó cho AEC → đổi hướng (mic rời / barge-in off).")
    print("\nNghe aec_raw.wav vs aec_clean.wav: clean phải mất tiếng tít, PHA 2 giữ giọng bạn.")


if __name__ == "__main__":
    main()
