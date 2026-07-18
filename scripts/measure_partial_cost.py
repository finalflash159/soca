# scripts/measure_partial_cost.py — dev only
import time

import numpy as np

from soca.asr import VietnameseASR

asr = VietnameseASR(num_threads=4)              # default model_key (matches realtime)
for secs in (1, 3, 5, 8):
    audio = (np.random.randn(16000 * secs) * 0.01).astype(np.float32)
    asr.transcribe(audio)                       # warm
    t0 = time.perf_counter()
    asr.transcribe(audio)
    print(f"{secs}s -> {(time.perf_counter()-t0)*1000:.0f}ms")   # numbers should be close
