from __future__ import annotations

import numpy as np
import soundfile as sf

from soca.core.audio_out import (
    NullAudioPlayer,
    SoundDevicePlayer,
    WavFileSink,
    _to_float32_mono,
)


def test_to_float32_mono_converts_stereo_int16_to_mono_float32() -> None:
    audio = np.array([[0, 32767], [-32768, 0]], dtype=np.int16)

    result = _to_float32_mono(audio)

    assert result.dtype == np.float32
    assert result.shape == (2,)
    assert result.flags.c_contiguous
    assert np.all(result <= 1.0)
    assert np.all(result >= -1.0)


def test_null_audio_player_returns_noop_result_with_duration() -> None:
    player = NullAudioPlayer()
    audio = np.zeros(2400, dtype=np.float32)

    result = player.play(audio, sample_rate=24000)

    assert result.played is False
    assert result.sample_rate == 24000
    assert result.audio_duration_ms == 100.0
    assert result.latency_ms == 0.0
    assert result.reason == "null_audio_player"


def test_sound_device_player_empty_audio_is_noop() -> None:
    player = SoundDevicePlayer()

    result = player.play(np.array([], dtype=np.float32), sample_rate=24000)

    assert result.played is False
    assert result.sample_rate == 24000
    assert result.audio_duration_ms == 0.0
    assert result.reason == "empty_audio"


def test_sound_device_player_uses_resampled_output_rate(monkeypatch) -> None:
    calls: list[tuple[np.ndarray, int, bool]] = []

    def fake_play(audio, samplerate, blocking=True):
        calls.append((audio, samplerate, blocking))

    monkeypatch.setattr("soca.core.audio_out.sd.play", fake_play)
    player = SoundDevicePlayer(output_sample_rate=8000)
    audio = np.ones(16000, dtype=np.float32)

    result = player.play(audio, sample_rate=16000, blocking=False)

    assert result.played is True
    assert result.sample_rate == 8000
    assert result.audio_duration_ms == 1000.0
    assert len(calls) == 1
    played_audio, played_rate, blocking = calls[0]
    assert played_rate == 8000
    assert blocking is False
    assert len(played_audio) == 8000


def test_wav_file_sink_writes_audio_file(tmp_path) -> None:
    output_path = tmp_path / "tts.wav"
    sink = WavFileSink(output_path)
    audio = np.zeros(1200, dtype=np.float32)

    result = sink.play(audio, sample_rate=24000)

    assert result.played is True
    assert output_path.exists()
    assert result.reason.startswith("saved:")

    info = sf.info(output_path)
    assert info.samplerate == 24000
    assert info.frames == 1200


def test_wav_file_sink_empty_audio_is_noop(tmp_path) -> None:
    output_path = tmp_path / "empty.wav"
    sink = WavFileSink(output_path)

    result = sink.play(np.array([], dtype=np.float32), sample_rate=24000)

    assert result.played is False
    assert result.reason == "empty_audio"
    assert not output_path.exists()


def test_sounddevice_session_reuses_one_output_stream(monkeypatch) -> None:
    instances = []

    class FakeOutputStream:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.writes = []
            self.started = False
            self.stopped = False
            self.closed = False
            instances.append(self)

        def start(self):
            self.started = True

        def write(self, block):
            self.writes.append(np.asarray(block).copy())
            return False

        def stop(self):
            self.stopped = True

        def abort(self):
            self.stopped = True

        def close(self):
            self.closed = True

    monkeypatch.setattr("soca.core.audio_out.sd.OutputStream", FakeOutputStream)
    player = SoundDevicePlayer(output_sample_rate=24_000)

    session = player.begin_playback(16_000)
    first = session.write(np.ones(480, dtype=np.float32))
    second = session.write(np.ones(240, dtype=np.float32))
    session.finish()

    assert len(instances) == 1
    assert instances[0].started is True
    assert instances[0].stopped is True
    assert instances[0].closed is True
    assert first.played is True
    assert second.played is True
    assert session.sample_rate == 24_000
    assert session.output_underflow_count == 0
