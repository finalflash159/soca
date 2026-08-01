from __future__ import annotations

import inspect
import random
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from queue import Queue
from threading import Event
from typing import Any

import numpy as np

from soca.asr import looks_like_context_echo
from soca.core import (
    AudioSink,
    EndpointConfig,
    LLMUsage,
    ResolvedVoiceRuntimeConfig,
    SoundDevicePlayer,
    StreamingEvent,
    TurnUsage,
    VoiceRuntimeBundle,
    audio_duration_ms,
    build_voice_runtime,
    record_until_silence,
    warm_up_voice_runtime,
)
from soca.core.repair import (
    RepairAction,
    RepairChoice,
    RepairKind,
    RepairTimings,
    default_repair_catalog,
)
from soca.core.text_chunking import normalize_text_for_tts
from soca.memory import SessionMemory
from soca.tts import VALTEC_TTS_CONFIG


@dataclass(frozen=True)
class VoiceMonitorEvent:
    type: str
    text: str = ""
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    usage: TurnUsage | None = None


VoiceRuntimeBuilder = Callable[..., VoiceRuntimeBundle]
VoiceRecorder = Callable[..., np.ndarray]
VoiceEventQueue = Queue[VoiceMonitorEvent | None]

# How often SoCa playfully calls out while nobody is speaking (ms of silence
# between greetings). Spaced so it feels like a gentle "alo?", not a nag.
_SILENCE_CALLOUT_INTERVAL_MS = 20_000


class VoiceMonitorController:
    """Thread-side adapter from VoicePipeline streaming events to UI events."""

    def __init__(
        self,
        config: ResolvedVoiceRuntimeConfig,
        *,
        runtime_builder: VoiceRuntimeBuilder = build_voice_runtime,
        recorder: VoiceRecorder = record_until_silence,
        player: AudioSink | None = None,
        warmup: bool = True,
        session_memory: SessionMemory | None = None,
        repair_timings: RepairTimings | None = None,
    ) -> None:
        self.config = config
        self.runtime_builder = runtime_builder
        self.recorder = recorder
        self.player = player or SoundDevicePlayer()
        self.warmup = warmup
        self.session_memory = session_memory
        self.repair_timings = repair_timings or RepairTimings()
        self.repair_catalog = default_repair_catalog()
        self.bundle: VoiceRuntimeBundle | None = None
        self._warmed_up = False
        self._idle_started_at: float | None = None
        self._no_reply_rng = random.Random()
        self._recent_no_reply_prompt_ids: deque[str] = deque(maxlen=8)
        # A DuplexAecSink player does barge-in (AEC + VAD) inline and exposes a
        # ``captured`` carry-over buffer; a plain/fake player (tests) does not.
        self._supports_barge_in = hasattr(self.player, "captured")
        # Audio kept from a barge-in interrupt, prepended to the next recording.
        self._pending_prefix: np.ndarray | None = None
        # Passive silence: SoCa periodically calls out the playful "alo, có ai
        # không? / moshi moshi?" greetings (no_input.attempt_1), cycling without
        # repeats. It only winds down (sleep + handover) after a long quiet.
        self._silence_callouts_done = 0

    def run_turn(self, queue: VoiceEventQueue) -> None:
        """Run one microphone turn and push normalized events to ``queue``."""
        stop_event = Event()
        self.run_loop(queue, stop_event=stop_event, max_turns=1)

    def run_loop(
        self,
        queue: VoiceEventQueue,
        *,
        stop_event: Event,
        max_turns: int | None = None,
    ) -> None:
        """Run the continuous voice loop and stream events to ``queue``.

        This method is synchronous by design: it runs in one worker thread.
        Textual's main thread consumes the queue and owns all widget writes.
        """
        turns = 0
        try:
            bundle = self._ensure_bundle(queue)
            queue.put(
                VoiceMonitorEvent(
                    "loop_started",
                    "Voice loop started",
                    metadata={"profile": self.config.profile_key},
                )
            )
            self._ensure_idle_clock()

            while not stop_event.is_set():
                turns += 1
                queue.put(
                    VoiceMonitorEvent(
                        "turn_start",
                        "Voice turn started",
                        metadata={"turn_index": turns},
                    )
                )
                if self._supports_barge_in:
                    queue.put(
                        VoiceMonitorEvent(
                            "barge_in",
                            "Barge-in armed",
                            metadata={"phase": "armed"},
                        )
                    )
                self._run_one_turn(bundle, queue, stop_event=stop_event)
                if stop_event.is_set():
                    break
                queue.put(
                    VoiceMonitorEvent(
                        "turn_end",
                        "Voice turn ended",
                        metadata={"turn_index": turns},
                    )
                )
                if max_turns is not None and turns >= max_turns:
                    break

            queue.put(
                VoiceMonitorEvent(
                    "loop_stopped",
                    "Voice loop stopped",
                    metadata={"turns": turns, "requested": stop_event.is_set()},
                )
            )
        except Exception as exc:  # pragma: no cover - terminal/runtime boundary
            queue.put(
                VoiceMonitorEvent(
                    "error",
                    str(exc),
                    metadata={"traceback": traceback.format_exc()},
                )
            )
        finally:
            queue.put(None)

    def stop(self) -> None:
        self.player.stop()

    def _ensure_bundle(self, queue: VoiceEventQueue) -> VoiceRuntimeBundle:
        if self.bundle is not None:
            return self.bundle

        queue.put(
            VoiceMonitorEvent(
                "loading",
                "Loading voice runtime",
                metadata={
                    "profile": self.config.profile_key,
                    "asr_model": self.config.asr_model,
                    "llm_model": self.config.llm_model,
                    "tts_model": VALTEC_TTS_CONFIG.key,
                    "voice": self.config.tts_voice,
                },
            )
        )
        t0 = time.perf_counter()
        self.bundle = self._build_runtime_bundle()
        queue.put(
            VoiceMonitorEvent(
                "ready",
                "Voice runtime ready",
                latency_ms=(time.perf_counter() - t0) * 1000,
                metadata={
                    "memory_status": self.bundle.memory_status,
                    "knowledge_status": self.bundle.knowledge_status,
                    "asr_guard_status": self.bundle.asr_guard_status,
                    "llm_backend": (
                        self.bundle.llm_settings.backend
                        if self.bundle.llm_settings is not None
                        else "unknown"
                    ),
                    "llm_provider": (
                        self.bundle.llm_settings.provider_key
                        if self.bundle.llm_settings is not None
                        else ""
                    ),
                    "llm_model": (
                        self.bundle.llm_settings.model_id
                        if self.bundle.llm_settings is not None
                        else self.bundle.config.llm_model
                    ),
                },
            )
        )

        if self.warmup and not self._warmed_up:
            for result in warm_up_voice_runtime(self.bundle):
                queue.put(
                    VoiceMonitorEvent(
                        "warmup",
                        result.component,
                        latency_ms=result.latency_ms,
                        metadata={"ok": result.ok, "detail": result.detail},
                    )
                )
            self._warmed_up = True

        return self.bundle

    def _build_runtime_bundle(self) -> VoiceRuntimeBundle:
        if self.session_memory is None:
            return self.runtime_builder(self.config)

        try:
            signature = inspect.signature(self.runtime_builder)
        except (TypeError, ValueError):
            signature = None

        supports_session_memory = signature is not None and (
            "session_memory" in signature.parameters
            or any(
                param.kind is inspect.Parameter.VAR_KEYWORD
                for param in signature.parameters.values()
            )
        )
        if supports_session_memory:
            return self.runtime_builder(self.config, session_memory=self.session_memory)
        return self.runtime_builder(self.config)

    def _run_one_turn(
        self,
        bundle: VoiceRuntimeBundle,
        queue: VoiceEventQueue,
        *,
        stop_event: Event | None = None,
    ) -> None:
        endpoint_config = EndpointConfig(
            endpoint_silence_ms=self.config.endpoint_silence_ms,
            max_record_ms=self.config.max_record_ms,
            partial_interval_ms=bundle.partial_interval_ms,  # seed from warmup
            adaptive=self.config.adaptive_endpoint,
        )
        queue.put(VoiceMonitorEvent("recording", "Listening"))
        t0 = time.perf_counter()
        if self._supports_barge_in:
            # Close any duplex stream left open by the previous turn (incl. a
            # passive-silence callout) so the recorder can reclaim the mic.
            self.player.stop()
        record_kwargs: dict[str, Any] = {
            "config": endpoint_config,
            "stop_event": stop_event,
            "turn_detector": bundle.turn_detector,
        }
        if self._pending_prefix is not None:
            record_kwargs["prefix"] = self._pending_prefix
            self._pending_prefix = None
        if bundle.partial_enabled and self._recorder_accepts("on_partial"):
            record_kwargs["on_partial"] = lambda committed, tentative: queue.put(
                VoiceMonitorEvent(
                    "asr_partial",
                    f"{committed} {tentative}".strip(),
                    metadata={"committed": committed, "tentative": tentative},
                )
            )
            record_kwargs["partial_transcriber"] = self._build_partial_transcriber(bundle)
        audio = self.recorder(bundle.detector, **record_kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000
        if stop_event is not None and stop_event.is_set():
            return
        queue.put(
            VoiceMonitorEvent(
                "recorded",
                "Recorded",
                latency_ms=latency_ms,
                metadata={
                    "samples": int(len(audio)),
                    "duration_s": len(audio) / 16000 if len(audio) else 0.0,
                },
            )
        )
        if len(audio):
            rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float32, copy=False)))))
            queue.put(
                VoiceMonitorEvent(
                    "voice_level",
                    "Voice level",
                    metadata={"rms": min(1.0, rms), "source": "microphone"},
                )
            )
        if not self._audio_has_speech(bundle, audio):
            self._handle_passive_silence(bundle, queue, stop_event=stop_event)
            return

        self._mark_user_spoke()
        self._stream_pipeline_events(bundle, audio, queue, stop_event=stop_event)

    def _recorder_accepts(self, name: str) -> bool:
        """Signature-guard like _turn_streaming: old fake recorders do not break."""
        import inspect

        try:
            params = inspect.signature(self.recorder).parameters
        except (TypeError, ValueError):
            return False
        return name in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

    @staticmethod
    def _build_partial_transcriber(bundle):
        """RobustASR wraps the raw ASR backend at .asr — partial uses the RAW one (cheap, no guards)."""
        inner = getattr(bundle.asr, "asr", None) or bundle.asr
        if not hasattr(inner, "transcribe"):
            return None

        try:
            params = inspect.signature(inner.transcribe).parameters
        except (TypeError, ValueError):
            params = {}
        accepts_context = "context" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

        def transcribe(audio):
            # context="" is DELIBERATE, not a missing value: partial exists
            # only to render the caption, so it stays cheap and never risks
            # the context-echo failure mode (§Q1b.3) leaking onto it. The
            # final transcript that reaches the LLM uses the real context.
            active_context = ""
            kwargs = {"context": active_context} if accepts_context else {}
            result = inner.transcribe(audio, **kwargs)
            text = getattr(result, "text", "") or ""
            # Belt-and-suspenders: partial has no RobustASR in front of it,
            # so if a future change ever starts passing a real context here
            # and someone forgets this comment, the echo would otherwise go
            # straight to the caption with zero protection.
            if looks_like_context_echo(text, active_context):
                return ""
            return text

        return transcribe

    def _stream_pipeline_events(
        self,
        bundle: VoiceRuntimeBundle,
        audio: np.ndarray,
        queue: VoiceEventQueue,
        *,
        stop_event: Event | None = None,
    ) -> None:
        runtime_meta: dict[str, Any] = {}
        first_tts_meta: dict[str, Any] | None = None
        tts_chunks = 0

        # The DuplexAecSink player sets ``interrupt_event`` from inside ``play`` when
        # it hears sustained speech (no separate listener thread needed).
        interrupt_event = threading.Event() if self._supports_barge_in else None
        stream_kwargs: dict[str, Any] = {"audio_sink": self.player}
        if interrupt_event is not None:
            stream_kwargs["interrupt_event"] = interrupt_event

        progress_setter = getattr(bundle.assistant_runtime, "set_progress_callback", None)
        if callable(progress_setter):
            progress_setter(
                lambda stage: queue.put(
                    VoiceMonitorEvent(
                        "progress",
                        metadata={"stage": str(stage)},
                    )
                )
            )
        try:
            for event in bundle.pipeline.turn_streaming(audio, **stream_kwargs):
                metadata = dict(event.metadata or {})
                usage: TurnUsage | None = None

                if event.type == "runtime":
                    runtime_meta = metadata
                elif event.type == "tts":
                    tts_chunks += 1
                    first_tts_meta = first_tts_meta or metadata
                elif event.type == "done":
                    if metadata.get("interrupted") and self._supports_barge_in:
                        queue.put(
                            VoiceMonitorEvent(
                                "barge_in",
                                "Barge-in detected",
                                metadata={"phase": "fired", "source": "duplex_aec"},
                            )
                        )
                        # Keep the (echo-cancelled) interrupting words for next turn.
                        self._pending_prefix = getattr(self.player, "captured", None)
                    usage = _build_voice_usage(
                        event=event,
                        runtime_meta=runtime_meta,
                        first_tts_meta=first_tts_meta,
                        tts_chunks=tts_chunks,
                    )
                    self._mark_idle_from_done_event(event)

                queue.put(_to_monitor_event(event, usage=usage))
        finally:
            if callable(progress_setter):
                progress_setter(None)
            if self._supports_barge_in:
                self.player.stop()  # close duplex stream so the recorder reclaims the mic

    def _audio_has_speech(self, bundle: VoiceRuntimeBundle, audio: np.ndarray) -> bool:
        if len(audio) == 0:
            return False

        speech_timestamps = getattr(bundle.detector, "speech_timestamps", None)
        if speech_timestamps is None:
            # Unit tests often use a plain object detector. Keep legacy behavior
            # there: non-empty fake audio is treated as speech.
            return True

        try:
            return bool(speech_timestamps(audio))
        except Exception:
            # A VAD failure should not silently drop user input. Let the normal
            # ASR/repair path decide instead.
            return True

    def _ensure_idle_clock(self) -> None:
        if self._idle_started_at is None:
            self._idle_started_at = time.perf_counter()

    def _mark_user_spoke(self) -> None:
        # A real turn clears the silence clock and the call-out counter.
        self._idle_started_at = None
        self._silence_callouts_done = 0

    def _mark_idle_from_done_event(self, event: StreamingEvent) -> None:
        # SoCa finished talking — start counting silence from now.
        self._idle_started_at = time.perf_counter()

    def _handle_passive_silence(
        self,
        bundle: VoiceRuntimeBundle,
        queue: VoiceEventQueue,
        *,
        stop_event: Event | None,
    ) -> None:
        if self._idle_started_at is None:
            self._idle_started_at = time.perf_counter()
        silence_ms = (time.perf_counter() - self._idle_started_at) * 1000

        # After a long quiet stretch, gently wind down: sleep + hand over to chat.
        if silence_ms >= self.repair_timings.sleep_voice_at_ms:
            choice = self.repair_catalog.select(
                RepairKind.SESSION_INACTIVE,
                "sleep",
                rng=self._no_reply_rng,
                recent_ids=tuple(self._recent_no_reply_prompt_ids),
            )
            self._recent_no_reply_prompt_ids.append(choice.prompt_id)
            self._speak_no_reply_choice(
                bundle, queue, choice, silence_ms=silence_ms, stop_event=stop_event
            )
            return

        # Not time for the next call-out yet → keep listening quietly.
        if silence_ms < self._silence_callouts_done * _SILENCE_CALLOUT_INTERVAL_MS:
            return

        # Playful presence check: cycle the no_input.attempt_1 greetings
        # ("alo, có ai không? / moshi moshi? / annyeong?") without repeats.
        choice = self.repair_catalog.select(
            RepairKind.NO_INPUT,
            "attempt_1",
            rng=self._no_reply_rng,
            recent_ids=tuple(self._recent_no_reply_prompt_ids),
        )
        self._recent_no_reply_prompt_ids.append(choice.prompt_id)
        self._silence_callouts_done += 1
        self._speak_no_reply_choice(
            bundle, queue, choice, silence_ms=silence_ms, stop_event=stop_event
        )

    def _speak_no_reply_choice(
        self,
        bundle: VoiceRuntimeBundle,
        queue: VoiceEventQueue,
        choice: RepairChoice,
        *,
        silence_ms: float,
        stop_event: Event | None,
    ) -> None:
        turn_start = time.perf_counter()
        leaves_voice = choice.action in (RepairAction.SLEEP_VOICE, RepairAction.HANDOVER_TO_CHAT)
        handover_target = "chat" if leaves_voice else None
        metadata = {
            "repair_kind": choice.kind.value,
            "repair_action": choice.action.value,
            "repair_attempt": self._silence_callouts_done,
            "handover_target": handover_target,
            "technical_reason": "passive_silence",
            "silence_ms": silence_ms,
        }
        queue.put(VoiceMonitorEvent("repair", choice.text, metadata=metadata))

        speech_text = normalize_text_for_tts(choice.text) or choice.text.strip()
        t0 = time.perf_counter()
        tts_result = bundle.tts.synthesize(speech_text)
        audio_ready = time.perf_counter()
        tts_metadata = {
            "chunk_index": 0,
            "ttfa_ms": (audio_ready - turn_start) * 1000,
            "tts_latency_ms": tts_result.latency_ms,
            "delivery": "repair",
            "repair_kind": choice.kind.value,
            "repair_action": choice.action.value,
        }
        queue.put(
            VoiceMonitorEvent(
                "tts",
                speech_text,
                latency_ms=(time.perf_counter() - t0) * 1000,
                metadata=tts_metadata,
            )
        )

        playback_metadata: dict[str, Any] = {
            **tts_metadata,
            "sync_granularity": "audio_chunk",
        }
        duration_ms = audio_duration_ms(len(tts_result.audio), tts_result.sample_rate)
        if duration_ms is not None:
            playback_metadata["audio_duration_ms"] = duration_ms
        queue.put(VoiceMonitorEvent("playback_started", speech_text, metadata=playback_metadata))
        playback = self.player.play(tts_result.audio, tts_result.sample_rate, blocking=True)
        queue.put(
            VoiceMonitorEvent(
                "audio",
                speech_text,
                latency_ms=_maybe_float(getattr(playback, "latency_ms", None)),
                metadata={
                    **tts_metadata,
                    "playback_latency_ms": _maybe_float(getattr(playback, "latency_ms", None)),
                },
            )
        )
        queue.put(
            VoiceMonitorEvent(
                "done",
                choice.text,
                latency_ms=(time.perf_counter() - turn_start) * 1000,
                metadata={
                    "rejected": True,
                    "terminal_status": "cancelled" if leaves_voice else "needs_clarification",
                    "rejection_reason": "passive_silence",
                    **metadata,
                },
            )
        )
        if leaves_voice and stop_event is not None:
            stop_event.set()


def _to_monitor_event(event: StreamingEvent, *, usage: TurnUsage | None) -> VoiceMonitorEvent:
    return VoiceMonitorEvent(
        type=event.type,
        text=event.text,
        latency_ms=event.latency_ms,
        metadata=dict(event.metadata or {}),
        usage=usage,
    )


def _build_voice_usage(
    *,
    event: StreamingEvent,
    runtime_meta: dict[str, Any],
    first_tts_meta: dict[str, Any] | None,
    tts_chunks: int,
) -> TurnUsage:
    done_meta = dict(event.metadata or {})
    first_tts = dict(first_tts_meta or {})
    route = str(done_meta.get("runtime_route") or runtime_meta.get("route") or "unknown")
    blocked = bool(done_meta.get("runtime_blocked") or runtime_meta.get("blocked") or False)
    llm = _coerce_llm_usage(runtime_meta.get("llm_usage"))

    return TurnUsage.from_voice(
        route=route,
        blocked=blocked,
        llm=llm,
        stage_latencies_ms=done_meta.get("stage_latencies_ms"),
        total_turn_latency_ms=event.latency_ms,
        first_tts_latency_ms=_maybe_float(first_tts.get("tts_latency_ms")),
        ttfa_ms=_maybe_float(first_tts.get("ttfa_ms")),
        tts_chunks=tts_chunks,
    )


def _coerce_llm_usage(value: Any) -> LLMUsage | None:
    if value is None:
        return None
    if isinstance(value, LLMUsage):
        return value
    if isinstance(value, dict):
        return LLMUsage(
            prompt_tokens=int(value.get("prompt_tokens") or 0),
            completion_tokens=int(value.get("completion_tokens") or 0),
            ttft_ms=_maybe_float(value.get("ttft_ms")) or 0.0,
            total_latency_ms=_maybe_float(value.get("total_latency_ms")) or 0.0,
            tokens_per_second=_maybe_float(value.get("tokens_per_second")) or 0.0,
        )
    return None


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "VoiceEventQueue",
    "VoiceMonitorController",
    "VoiceMonitorEvent",
    "VoiceRecorder",
    "VoiceRuntimeBuilder",
]
