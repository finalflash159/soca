import numpy as np

from soca.core import endpoint as endpoint_module
from soca.core.endpoint import (
    EndpointConfig,
    block_samples,
    record_until_silence,
    should_stop_recording,
)


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
