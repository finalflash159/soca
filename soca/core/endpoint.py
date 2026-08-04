from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from threading import Event

import numpy as np
import sounddevice as sd

from soca.core.turn_taking import (
    PARTIAL_EWMA_DOWN,
    PARTIAL_EWMA_UP,
    PARTIAL_MARGIN,
    IncrementalVadTracker,
    LocalAgreement,
    clamp_interval_ms,
    required_silence_from_p,
)

_DEBUG = os.environ.get("SOCA_ENDPOINT_DEBUG", "") not in ("", "0")

class _PartialWorker:
    """Background re-transcriber: buffer -> text -> LocalAgreement -> callback.

    SELF-TUNING cadence (closed-loop): seed from warmup (Tier 1), then each transcribe
    measures real wall-time and EWMAs it to adjust the next interval (Tier 2). Runs only after speech.
    """

    def __init__(self, chunks, config, *, on_partial, transcriber):
        self._chunks = chunks                    # list SHARED with the main loop
        self._interval = config.partial_interval_ms / 1000        # SEED (seconds) - Tier 1
        self._ewma_ms = config.partial_interval_ms / PARTIAL_MARGIN  # initial per-call estimate
        self._on_partial = on_partial
        self._transcriber = transcriber
        self._speech = threading.Event()
        self._done = threading.Event()
        self._agreement = LocalAgreement()
        self._last_len = 0
        self._thread: threading.Thread | None = None
        if on_partial is not None and transcriber is not None:
            self._thread = threading.Thread(
                target=self._run, daemon=True, name="soca-partial-asr"
            )
            self._thread.start()

    def notify_speech(self) -> None:
        self._speech.set()

    def stop(self) -> None:
        self._done.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _adapt_interval(self, sample_ms: float) -> None:
        """Tier 2 online: ASYMMETRIC EWMA of REAL wall-time -> next interval.

        NO rho (the sample already includes contention/thermal). Slower -> widen FAST
        (large alpha) to avoid starving CPU; faster -> tighten SLOWLY (small alpha).
        """
        alpha = PARTIAL_EWMA_UP if sample_ms > self._ewma_ms else PARTIAL_EWMA_DOWN
        self._ewma_ms = alpha * sample_ms + (1 - alpha) * self._ewma_ms
        self._interval = clamp_interval_ms(PARTIAL_MARGIN * self._ewma_ms) / 1000

    def _run(self) -> None:
        while not self._done.wait(self._interval):        # self._interval changes dynamically
            if not self._speech.is_set():
                continue
            snapshot_chunks = list(self._chunks)          # copy REF list (GIL-safe)
            if not snapshot_chunks:
                continue
            audio = np.concatenate(snapshot_chunks).astype(np.float32, copy=False)
            if len(audio) - self._last_len < 6400:        # <400ms new audio -> skip this tick
                continue
            self._last_len = len(audio)
            t0 = time.perf_counter()
            try:
                text = self._transcriber(audio) or ""
            except Exception:                             # ASR error must not kill the mic
                continue                                  # (not measured -> EWMA stays clean)
            self._adapt_interval((time.perf_counter() - t0) * 1000)   # measure REAL wall-time
            committed, tentative = self._agreement.update(text)
            try:
                self._on_partial(committed, tentative)
            except Exception:
                continue


def _start_partial_worker(chunks, config, *, on_partial, transcriber) -> _PartialWorker:
    return _PartialWorker(chunks, config, on_partial=on_partial, transcriber=transcriber)


@dataclass(frozen=True)
class EndpointConfig:
    sample_rate: int = 16000
    block_ms: int = 100
    endpoint_silence_ms: int = 700  # user silent for 0.7s -> end of turn
    max_record_ms: int = 10000
    min_audio_ms: int = 300
    adaptive: bool = True
    # Do not let an ordinary within-turn pause close the turn immediately
    # after the VAD floor.  The 1.8s floor is the release setting measured on
    # the Vietnamese turn-taking replay; Smart Turn still controls patience up
    # to the 3s ceiling after this guard.
    floor_silence_ms: int = 1800
    ceil_silence_ms: int = 3000
    use_incremental_vad: bool = True
    partial_interval_ms: int = 900

def _apply_env_overrides(config: EndpointConfig) -> EndpointConfig:
    """Live-tuning knobs, same culture as SOCA_BARGE_* (frozen -> replace)."""
    from dataclasses import replace

    changes: dict = {}
    if os.environ.get("SOCA_ENDPOINT_ADAPTIVE", "") in ("0", "false"):
        changes["adaptive"] = False
    for env, fld in (
        ("SOCA_ENDPOINT_FLOOR_MS", "floor_silence_ms"),
        ("SOCA_ENDPOINT_CEIL_MS", "ceil_silence_ms"),
        ("SOCA_PARTIAL_INTERVAL_MS", "partial_interval_ms"),
    ):
        value = os.environ.get(env)
        if value:
            changes[fld] = int(value)
    return replace(config, **changes) if changes else config

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


def _turn_window(chunks, config, *, max_s: float = 8.0):
    """Return the current turn, including the pause being evaluated.

    Smart Turn is trained/integrated as an end-of-turn classifier over the
    current turn after VAD observes silence.  Removing the trailing silence
    before inference discards the endpoint signal and makes the model judge
    only the preceding speech.  Keep the last eight seconds, matching the
    upstream model contract.
    """
    if not chunks:
        return None
    cap_blocks = max(2, int((config.ceil_silence_ms + max_s * 1000) / max(config.block_ms, 1)) + 2)
    audio = np.concatenate(chunks[-cap_blocks:]).astype(np.float32, copy=False)
    start = max(0, len(audio) - int(max_s * config.sample_rate))
    return audio[start:]


def _decide_required_silence(config, silence_ms, chunks, turn_detector) -> float:
    if not config.adaptive:
        return float(config.endpoint_silence_ms)          # fixed, deterministic
    if silence_ms < config.floor_silence_ms:
        return float(config.ceil_silence_ms)
    window = _turn_window(chunks, config)
    if window is None:
        return float(config.ceil_silence_ms)
    try:
        p = turn_detector.p_still_speaking(window)
    except Exception as exc:
        print(
            f"[endpoint] turn_detector error: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return float(config.ceil_silence_ms)
    return required_silence_from_p(p, config)


def record_until_silence(
    detector,
    config: EndpointConfig | None = None,
    stop_event: Event | None = None,
    prefix: np.ndarray | None = None,
    *,
    turn_detector=None,
    on_partial=None,
    partial_transcriber=None,
    stream_factory=None,
) -> np.ndarray:
    config = _apply_env_overrides(config or EndpointConfig())
    stream_factory = stream_factory or sd.InputStream
    n_block_samples = block_samples(config)
    max_samples = int(config.sample_rate * config.max_record_ms / 1000)
    vad_model = getattr(detector, "model", None)
    tracker = None
    if (
        config.adaptive
        and config.use_incremental_vad
        and vad_model is not None
        and config.sample_rate == 16000
    ):
        tracker = IncrementalVadTracker(
            vad_model, threshold=getattr(detector, "threshold", 0.5)
        )
        tracker.reset()
    if config.adaptive and turn_detector is None:
        raise ValueError("adaptive endpoint requires a turn detector")

    chunks: list[np.ndarray] = []
    has_seen_speech = False
    total_samples = 0

    # Barge-in carry-over: seed the buffer with the audio the listener already
    # captured (the start of the user's interrupting words). Mark speech as seen
    # so endpointing can close promptly even if the user stops talking right away.
    if prefix is not None and len(prefix) > 0:
        prefix = np.asarray(prefix, dtype=np.float32).reshape(-1)
        chunks.append(prefix)
        has_seen_speech = True
        total_samples = len(prefix)
        if tracker is not None:
            tracker.seed_speech(len(prefix) / config.sample_rate * 1000)

    worker = _start_partial_worker(
        chunks, config, on_partial=on_partial, transcriber=partial_transcriber
    )
    try:
        with stream_factory(
            samplerate=config.sample_rate, channels=1, dtype="float32"
        ) as stream:
            while total_samples < max_samples:
                if stop_event is not None and stop_event.is_set():
                    break
                block, _overflowed = stream.read(n_block_samples)
                block_mono = np.asarray(block, dtype=np.float32).reshape(-1)
                chunks.append(block_mono)
                total_samples += len(block_mono)

                if total_samples < int(config.sample_rate * config.min_audio_ms / 1000):
                    continue

                if tracker is not None:
                    tracker.feed(block_mono)
                    if tracker.has_speech:
                        worker.notify_speech()
                    speech_ms = tracker.speech_ms
                    silence_ms = tracker.silence_run_ms
                    seen = tracker.has_speech
                else:
                    audio = np.concatenate(chunks).astype(np.float32, copy=False)
                    timestamps = detector.speech_timestamps(audio)
                    seen = has_seen_speech or bool(timestamps)
                    speech_ms = (
                        sum(t["end"] - t["start"] for t in timestamps)
                        / config.sample_rate * 1000
                    )
                    last_end = timestamps[-1]["end"] if timestamps else 0
                    silence_ms = (len(audio) - last_end) / config.sample_rate * 1000
                    if seen:
                        worker.notify_speech()

                has_seen_speech = seen
                if not has_seen_speech:
                    continue

                required = _decide_required_silence(
                    config, silence_ms, chunks, turn_detector
                )
                if _DEBUG:
                    print(
                        f"[endpoint] adaptive={config.adaptive} "
                        f"speech={speech_ms:5.0f}ms "
                        f"silence={silence_ms:4.0f}/{required:4.0f}ms",
                        file=sys.stderr, flush=True,
                    )
                if silence_ms >= required:
                    break
    finally:
        worker.stop()

    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)
