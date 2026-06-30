from __future__ import annotations

import threading

import numpy as np

_AEC_SAMPLE_RATE = 16000 # far set at 16k to match mic/VAD/AEC

class AECReference:
    """Far-end reference shared between playback (push) and barge-in listener (pull)."""

    def __init__(self, capacity_samples: int = _AEC_SAMPLE_RATE) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._cap = capacity_samples
        self._lock = threading.Lock()

    def push(self, samples_16k: np.ndarray) -> None:
        """Playback call: push the far-end audio to the reference buffer."""
        with self._lock:
            self._buf = np.concatenate([self._buf, samples_16k])[-self._cap:]

    def pull(self, n: int) -> np.ndarray:
        """Listener call: pull the OLDEST n samples (FIFO).

        AEC3 expects the render (far) signal in time order, so drain the FRONT of
        the buffer, not the newest tail. Underrun is padded at the END (future) to
        keep the render stream continuous.
        """
        with self._lock:
            head = self._buf[:n].copy()
            self._buf = self._buf[n:]
        if len(head) < n:
            head = np.concatenate([head, np.zeros(n - len(head), dtype=np.float32)])
        return head

    def clear(self) -> None:
        """Clear the reference buffer."""
        with self._lock:
            self._buf = self._buf[:0]

