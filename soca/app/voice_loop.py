from __future__ import annotations

import inspect
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
from rich.console import Console
from rich.text import Text

from soca.app.console import (
    print_recorded_audio,
    print_runtime_header,
    print_streaming_event,
    print_waiting_for_speech,
    print_warmup_result,
    print_warmup_start,
)
from soca.app.style.palette import ACCENT, ICON, st
from soca.app.usage_view import print_turn_usage
from soca.core import AudioSink, EndpointConfig, SoundDevicePlayer, record_until_silence
from soca.core.text_chunking import normalize_text_for_tts
from soca.core.usage import TurnUsage
from soca.core.voice_runtime import (
    ResolvedVoiceRuntimeConfig,
    VoiceRuntimeBundle,
    VoiceRuntimeWarmupError,
    build_voice_runtime,
    warm_up_voice_runtime,
)
from soca.tts import VALTEC_TTS_CONFIG

InputFn = Callable[[str], str]
RuntimeBuilder = Callable[[ResolvedVoiceRuntimeConfig], VoiceRuntimeBundle]
Recorder = Callable[..., np.ndarray]
def run_voice_loop(
    config: ResolvedVoiceRuntimeConfig,
    *,
    no_speak_rejections: bool = False,
    no_speak_repairs: bool = False,
    press_enter_to_record: bool = False,
    warmup: bool = True,
    show_usage: bool = False,
    console: Console | None = None,
    input_fn: InputFn = input,
    runtime_builder: RuntimeBuilder = build_voice_runtime,
    recorder: Recorder = record_until_silence,
    player: AudioSink | None = None,
    max_turns: int | None = None,
) -> int:
    """Run the interactive microphone voice loop.

    The optional callables are for tests and future CLI reuse. Production callers use
    the defaults, which build real models, record from the microphone, and play audio.
    """
    console = console or Console()
    bundle = runtime_builder(config)
    active_player: AudioSink | None = None
    try:
        active_player = player or SoundDevicePlayer()
        player = active_player
        # A DuplexAecSink player does barge-in (AEC + VAD) inline and exposes a
        # ``captured`` carry-over buffer; a plain player does not.
        supports_barge_in = hasattr(player, "captured")
        # `--no-speak-repairs` is the new name; `--no-speak-rejections` is a kept alias.
        suppress_repairs = no_speak_repairs or no_speak_rejections

        if warmup:
            print_warmup_start(console)
            warmup_results = warm_up_voice_runtime(bundle)
            for result in warmup_results:
                print_warmup_result(console, result)
            failures = [result for result in warmup_results if not result.ok]
            if failures:
                raise VoiceRuntimeWarmupError(tuple(failures))

        endpoint_config = EndpointConfig(
            endpoint_silence_ms=config.endpoint_silence_ms,
            max_record_ms=config.max_record_ms,
            adaptive=config.adaptive_endpoint,
        )

        print_runtime_header(
            console,
            profile_key=config.profile_key,
            asr_model=config.asr_model,
            llm_model=config.llm_model,
            tts_engine=VALTEC_TTS_CONFIG.key,
            voice=config.tts_voice,
            memory_status=bundle.memory_status,
            knowledge_status=bundle.knowledge_status,
            asr_guard_status=bundle.asr_guard_status,
        )

        completed_turns = 0
        # Audio captured when the previous turn was barge-in interrupted; prepended to
        # the next recording so the user's first words are not lost.
        pending_prefix: np.ndarray | None = None
        while max_turns is None or completed_turns < max_turns:
            if press_enter_to_record:
                input_fn("\nPress ENTER and speak. Ctrl+C to quit.")
            print_waiting_for_speech(console, manual_start=press_enter_to_record)

            record_kwargs: dict[str, Any] = {
                "config": endpoint_config,
                "turn_detector": bundle.turn_detector,
            }
            if pending_prefix is not None:
                record_kwargs["prefix"] = pending_prefix
                pending_prefix = None
            audio = recorder(bundle.detector, **record_kwargs)
            duration_s = (
                len(audio) / endpoint_config.sample_rate if endpoint_config.sample_rate > 0 else 0.0
            )
            print_recorded_audio(console, duration_s=duration_s)

            response_open = False
            runtime_meta: dict = {}
            first_tts_meta: dict | None = None
            tts_chunks = 0
            turn_interrupted = False
            interrupt_event = threading.Event() if supports_barge_in else None
            for event in _turn_streaming(
                bundle.pipeline,
                audio,
                player,
                speak_rejections=not suppress_repairs,
                interrupt_event=interrupt_event,
            ):
                if event.type == "llm_token":
                    if not response_open:
                        console.print(Text(f"{ICON.BIRD} ", style=st(f"bold {ACCENT}")), end="")
                        response_open = True
                    console.print(event.text, end="", markup=False, highlight=False, soft_wrap=True)
                    continue

                # Collect per-turn telemetry for `--usage` before any early `continue`.
                if event.type == "runtime":
                    runtime_meta = event.metadata or {}
                elif event.type == "tts":
                    tts_chunks += 1
                    if first_tts_meta is None:
                        first_tts_meta = event.metadata or {}

                streamed_this_turn = response_open
                if response_open:
                    console.print()  # close the live response line
                    response_open = False

                # When tokens already streamed live, the runtime panel would just
                # repeat the same text, so skip it.
                if event.type == "runtime" and streamed_this_turn:
                    continue

                rejected = bool(event.metadata and event.metadata.get("rejected"))
                if event.type == "done" and event.metadata:
                    turn_interrupted = bool(event.metadata.get("interrupted"))
                # The follow-up text is printed via the `repair` event; here we only
                # speak it when the pipeline emitted a rejected `done` without audio.
                if (
                    event.type == "done"
                    and rejected
                    and event.text
                    and not suppress_repairs
                    and tts_chunks == 0
                ):
                    speech_text = normalize_text_for_tts(event.text) or event.text.strip()
                    tts_result = bundle.tts.synthesize(speech_text)
                    player.play(tts_result.audio, tts_result.sample_rate, blocking=True)

                print_streaming_event(console, event)

                if event.type == "done" and show_usage and not rejected:
                    done_meta = event.metadata or {}
                    first_tts = first_tts_meta or {}
                    usage = TurnUsage.from_voice(
                        route=done_meta.get("runtime_route") or runtime_meta.get("route", ""),
                        blocked=bool(done_meta.get("runtime_blocked", False)),
                        llm=runtime_meta.get("llm_usage"),
                        stage_latencies_ms=done_meta.get("stage_latencies_ms"),
                        total_turn_latency_ms=event.latency_ms,
                        first_tts_latency_ms=first_tts.get("tts_latency_ms"),
                        ttfa_ms=first_tts.get("ttfa_ms"),
                        tts_chunks=tts_chunks,
                    )
                    print_turn_usage(console, usage)

            if supports_barge_in:
                player.stop()  # close the duplex stream so the recorder reclaims the mic
                if turn_interrupted:
                    pending_prefix = getattr(player, "captured", None)
            completed_turns += 1

        return 0
    finally:
        cleanup_failures: list[tuple[str, Exception]] = []
        if active_player is not None:
            stop_player = getattr(active_player, "stop", None)
            if callable(stop_player):
                try:
                    stop_player()
                except Exception as exc:  # noqa: BLE001 - continue deterministic teardown
                    cleanup_failures.append(("audio", exc))
        try:
            bundle.close()
        except Exception as exc:  # noqa: BLE001 - expose cleanup failure
            cleanup_failures.append(("runtime", exc))
        if cleanup_failures:
            details = "; ".join(f"{name}: {error}" for name, error in cleanup_failures)
            raise RuntimeError(
                f"Voice loop cleanup failed: {details}"
            ) from cleanup_failures[0][1]


def _turn_streaming(
    pipeline: Any,
    audio: np.ndarray,
    player: AudioSink,
    *,
    speak_rejections: bool,
    interrupt_event: threading.Event | None = None,
):
    kwargs: dict[str, Any] = {"audio_sink": player}
    try:
        signature = inspect.signature(pipeline.turn_streaming)
    except (TypeError, ValueError):
        signature = None
    supports_extra_kwargs = signature is not None and any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    )
    if signature is not None and (
        "speak_rejections" in signature.parameters or supports_extra_kwargs
    ):
        kwargs["speak_rejections"] = speak_rejections

    accepts_interrupt = signature is not None and (
        "interrupt_event" in signature.parameters or supports_extra_kwargs
    )
    # The DuplexAecSink player sets ``interrupt_event`` from inside ``play`` when it
    # hears sustained speech (no separate listener thread needed).
    if interrupt_event is not None and accepts_interrupt:
        kwargs["interrupt_event"] = interrupt_event
    return pipeline.turn_streaming(audio, **kwargs)
