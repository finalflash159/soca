from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np

from soca.core.duplex_aec_sink import DuplexAecSink, _DuplexPlaybackSession


def test_duplex_session_pads_only_once_at_finish() -> None:
    calls = []
    released = []
    sink = SimpleNamespace(rate=16_000, frame=4)

    def process(far_audio, *, interrupt_event):
        del interrupt_event
        calls.append(np.asarray(far_audio).copy())
        return False, len(far_audio)

    sink._process_far_frames = process
    sink._release_playback_session = released.append
    session = _DuplexPlaybackSession(sink)

    first = session.write(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    second = session.write(np.array([4.0, 5.0, 6.0], dtype=np.float32))

    assert first.played is False
    assert second.played is True
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], [1.0, 2.0, 3.0, 4.0])

    session.finish()

    assert len(calls) == 2
    np.testing.assert_array_equal(calls[1], [5.0, 6.0, 0.0, 0.0])
    assert released == [session]


def test_duplex_uses_identical_far_audio_for_speaker_and_aec(monkeypatch) -> None:
    speaker_blocks = []
    aec_far_blocks = []

    class FakeStream:
        def write(self, block):
            speaker_blocks.append(np.asarray(block).reshape(-1).copy())

        def read(self, frame):
            return np.zeros((frame, 1), dtype=np.float32), False

    class FakeAec:
        def process(self, near, far):
            del near
            aec_far_blocks.append(np.asarray(far).copy())
            return np.zeros_like(far)

    sink = object.__new__(DuplexAecSink)
    sink.rate = 16_000
    sink.frame = 4
    sink.block_ms = 0.25
    sink.sustained_ms = 400.0
    sink.vad_threshold = 0.7
    sink._stream = FakeStream()
    sink._aec = FakeAec()
    sink._model = lambda _audio, _rate: SimpleNamespace(item=lambda: 0.0)
    sink._window = deque(maxlen=4)
    sink._run_ms = 0.0
    sink.captured = None
    monkeypatch.setattr(
        "soca.core.duplex_aec_sink.torch.from_numpy",
        lambda value: value,
    )

    far = np.arange(8, dtype=np.float32)
    interrupted, consumed = sink._process_far_frames(
        far,
        interrupt_event=None,
    )

    assert interrupted is False
    assert consumed == 8
    assert len(speaker_blocks) == len(aec_far_blocks) == 2
    for speaker, aec_far in zip(speaker_blocks, aec_far_blocks, strict=True):
        np.testing.assert_array_equal(speaker, aec_far)
