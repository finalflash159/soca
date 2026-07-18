import time
from dataclasses import replace

import numpy as np

from soca.core import endpoint as endpoint_module
from soca.core.endpoint import (
    EndpointConfig,
    _start_partial_worker,
    block_samples,
    record_until_silence,
    should_stop_recording,
)


class ScriptedStream:
    """Fake stream: yields scripted blocks + context-manager protocol."""
    def __init__(self, blocks): self.blocks = list(blocks)
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self, n):
        if not self.blocks:
            raise AssertionError("read past the scripted blocks")
        return self.blocks.pop(0), False


class ScriptedModel:
    def __init__(self, probs): self.probs = list(probs)
    def __call__(self, tensor, sr):
        class _O:
            def __init__(self, v): self._v = v
            def item(self): return self._v
        return _O(self.probs.pop(0) if self.probs else 0.0)
    def reset_states(self): pass

class StreamingDetector:
    def __init__(self, probs):
        self.model = ScriptedModel(probs)
        self.threshold = 0.5
    def speech_timestamps(self, audio): raise AssertionError("batch path must not be called")

class FakeInputStream:
    def __init__(self, blocks: list[np.ndarray]):
        self.blocks = list(blocks)
        self.kwargs: dict[str, object] = {}
        self.read_sizes: list[int] = []
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.exited = True
        return False

    def read(self, n_samples: int):
        self.read_sizes.append(n_samples)
        if not self.blocks:
            raise AssertionError("record_until_silence read more blocks than expected")
        return self.blocks.pop(0), False


class FakeDetector:
    def __init__(self, first_detection_samples: int | None, speech_end_samples: int):
        self.first_detection_samples = first_detection_samples
        self.speech_end_samples = speech_end_samples
        self.calls: list[int] = []

    def speech_timestamps(self, audio: np.ndarray) -> list[dict[str, int]]:
        self.calls.append(len(audio))
        if self.first_detection_samples is None:
            return []
        if len(audio) < self.first_detection_samples:
            return []
        return [{"start": 0, "end": self.speech_end_samples}]


def install_fake_input_stream(monkeypatch, stream: FakeInputStream):
    def fake_input_stream(**kwargs):
        stream.kwargs = kwargs
        return stream

    monkeypatch.setattr(endpoint_module.sd, "InputStream", fake_input_stream)


def test_block_samples_uses_config_sample_rate_and_block_ms():
    config = EndpointConfig(sample_rate=16000, block_ms=100)

    assert block_samples(config) == 1600


def test_should_not_stop_before_any_speech():
    assert should_stop_recording(
        speech_timestamps=[],
        total_samples=16000,
        sample_rate=16000,
        endpoint_silence_ms=700,
    ) is False


def test_should_not_stop_when_silence_is_short():
    timestamps = [{"start": 1000, "end": 16000}]

    result = should_stop_recording(
        speech_timestamps=timestamps,
        total_samples=16000 + 4000,  # 250ms silence
        sample_rate=16000,
        endpoint_silence_ms=700,
    )

    assert result is False


def test_should_stop_after_endpoint_silence():
    timestamps = [{"start": 1000, "end": 16000}]

    result = should_stop_recording(
        speech_timestamps=timestamps,
        total_samples=16000 + 12000,  # 750ms silence
        sample_rate=16000,
        endpoint_silence_ms=700,
    )

    assert result is True


def test_should_treat_negative_silence_as_not_ready():
    timestamps = [{"start": 1000, "end": 20000}]

    result = should_stop_recording(
        speech_timestamps=timestamps,
        total_samples=16000,
        sample_rate=16000,
        endpoint_silence_ms=700,
    )

    assert result is False


def test_record_until_silence_stops_after_speech_then_endpoint_silence(monkeypatch):
    config = EndpointConfig(
        sample_rate=1000,
        block_ms=100,
        endpoint_silence_ms=200,
        max_record_ms=1000,
        min_audio_ms=100,
        adaptive=False,  # regression guard: legacy fixed-threshold behavior
    )
    blocks = [
        np.full((100, 1), fill_value=value, dtype=np.float32)
        for value in (0.1, 0.2, 0.3, 0.4, 0.5)
    ]
    stream = FakeInputStream(blocks)
    install_fake_input_stream(monkeypatch, stream)
    detector = FakeDetector(first_detection_samples=200, speech_end_samples=150)

    audio = record_until_silence(detector, config=config)

    assert stream.entered is True
    assert stream.exited is True
    assert stream.kwargs == {"samplerate": 1000, "channels": 1, "dtype": "float32"}
    assert stream.read_sizes == [100, 100, 100, 100]
    assert detector.calls == [100, 200, 300, 400]
    assert audio.dtype == np.float32
    assert audio.shape == (400,)
    assert audio[-1] == np.float32(0.4)


def test_record_until_silence_prepends_barge_in_prefix(monkeypatch):
    # The barge-in prefix is treated as speech already seen, so the recording
    # closes after endpoint silence even though the new blocks are all silent.
    config = EndpointConfig(
        sample_rate=1000,
        block_ms=100,
        endpoint_silence_ms=200,
        max_record_ms=1000,
        min_audio_ms=100,
        adaptive=False,  # regression guard: legacy fixed-threshold behavior
    )
    prefix = np.full(150, fill_value=0.9, dtype=np.float32)  # 150ms of captured words
    blocks = [np.zeros((100, 1), dtype=np.float32) for _ in range(3)]
    stream = FakeInputStream(blocks)
    install_fake_input_stream(monkeypatch, stream)
    # Speech ends at sample 150 (end of the prefix); nothing new is spoken.
    detector = FakeDetector(first_detection_samples=100, speech_end_samples=150)

    audio = record_until_silence(detector, config=config, prefix=prefix)

    # The very first detector call already sees the prefix + the first new block.
    assert detector.calls[0] == 250
    # Two silent blocks are enough: 150 -> 250 (100ms silence) -> 350 (200ms => stop).
    assert stream.read_sizes == [100, 100]
    # Prefix survives at the front of the returned audio.
    assert np.array_equal(audio[:150], prefix)
    assert audio.shape == (350,)


def test_record_until_silence_keeps_recording_until_max_when_no_speech(monkeypatch):
    config = EndpointConfig(
        sample_rate=1000,
        block_ms=100,
        endpoint_silence_ms=100,
        max_record_ms=300,
        min_audio_ms=100,
    )
    blocks = [
        np.zeros((100, 1), dtype=np.float32),
        np.zeros((100, 1), dtype=np.float32),
        np.zeros((100, 1), dtype=np.float32),
    ]
    stream = FakeInputStream(blocks)
    install_fake_input_stream(monkeypatch, stream)
    detector = FakeDetector(first_detection_samples=None, speech_end_samples=0)

    audio = record_until_silence(detector, config=config)

    assert stream.read_sizes == [100, 100, 100]
    assert detector.calls == [100, 200, 300]
    assert audio.dtype == np.float32
    assert audio.shape == (300,)

def test_adaptive_patient_does_not_cut_short_utterance():
    # ~0.8s speech then ~0.9s silence: fixed-700 would cut, adaptive patient(1100) does NOT.
    n_speech, n_sil = 25, 28                        # 800ms speech + 896ms im
    # max_record = EXACT scripted length: recorder stops at max right after the last block,
    # so it never over-reads (ScriptedStream raises when empty), as long as adaptive holds.
    config = replace(
        EndpointConfig(adaptive=True, max_record_ms=(n_speech + n_sil) * 32),
        block_ms=32,                                # 1 block=512 samples=32ms <-> 1 frame
    )
    probs = [0.9] * n_speech + [0.1] * n_sil
    blocks = [np.zeros((512, 1), dtype=np.float32) for _ in range(n_speech + n_sil)]
    detector = StreamingDetector(probs)
    audio = record_until_silence(
        detector, config=config, stream_factory=lambda **kw: ScriptedStream(blocks)
    )
    assert len(audio) == (n_speech + n_sil) * 512   # ran the whole script, did NOT cut early


def test_partial_worker_transcribes_and_reports():
    calls = []
    chunks = [np.zeros(16000, dtype=np.float32)]
    config = EndpointConfig(partial_interval_ms=30)
    worker = _start_partial_worker(
        chunks, config,
        on_partial=lambda c, t: calls.append((c, t)),
        transcriber=lambda audio: "xin chào",
    )
    worker.notify_speech()
    time.sleep(0.15)
    worker.stop()
    assert ("xin chào", "") in calls or ("", "xin chào") in calls


def test_partial_worker_never_runs_without_speech():
    calls = []
    worker = _start_partial_worker(
        [np.zeros(16000, dtype=np.float32)], EndpointConfig(partial_interval_ms=30),
        on_partial=lambda c, t: calls.append((c, t)),
        transcriber=lambda audio: "x",
    )
    time.sleep(0.15)
    worker.stop()      # no notify_speech -> never transcribes
    assert calls == []


def test_partial_worker_survives_transcriber_crash():
    def boom(audio): raise RuntimeError("ASR boom")
    worker = _start_partial_worker(
        [np.zeros(16000, dtype=np.float32)], EndpointConfig(partial_interval_ms=30),
        on_partial=lambda c, t: None, transcriber=boom,
    )
    worker.notify_speech()
    time.sleep(0.15)
    worker.stop()                        # must not raise -> passing = quiet
