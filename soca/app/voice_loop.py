from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
from rich.console import Console

from soca.app.console import (
    print_recorded_audio,
    print_runtime_header,
    print_streaming_event,
    print_waiting_for_speech,
    print_warmup_result,
    print_warmup_start,
)
from soca.app.usage_view import print_turn_usage
from soca.core import AudioSink, EndpointConfig, SoundDevicePlayer, record_until_silence
from soca.core.text_chunking import normalize_text_for_tts
from soca.core.usage import TurnUsage
from soca.core.voice_runtime import (
    ResolvedVoiceRuntimeConfig,
    VoiceRuntimeBundle,
    build_voice_runtime,
    warm_up_voice_runtime,
)

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
    player = player or SoundDevicePlayer()
    # `--no-speak-repairs` is the new name; `--no-speak-rejections` is a kept alias.
    suppress_repairs = no_speak_repairs or no_speak_rejections

    if warmup:
        print_warmup_start(console)
        for result in warm_up_voice_runtime(bundle):
            print_warmup_result(console, result)

    endpoint_config = EndpointConfig(
        endpoint_silence_ms=config.endpoint_silence_ms,
        max_record_ms=config.max_record_ms,
    )

    print_runtime_header(
        console,
        profile_key=config.profile_key,
        asr_model=config.asr_model,
        llm_model=config.llm_model,
        tts_model=config.tts_model,
        voice=config.tts_voice,
        memory_status=bundle.memory_status,
        knowledge_status=bundle.knowledge_status,
        asr_guard_status=bundle.asr_guard_status,
    )

    completed_turns = 0
    while max_turns is None or completed_turns < max_turns:
        if press_enter_to_record:
            input_fn("\nPress ENTER and speak. Ctrl+C to quit.")
        print_waiting_for_speech(console, manual_start=press_enter_to_record)

        audio = recorder(bundle.detector, config=endpoint_config)
        duration_s = (
            len(audio) / endpoint_config.sample_rate if endpoint_config.sample_rate > 0 else 0.0
        )
        print_recorded_audio(console, duration_s=duration_s)

        response_open = False
        runtime_meta: dict = {}
        first_tts_meta: dict | None = None
        tts_chunks = 0
        for event in _turn_streaming(
            bundle.pipeline,
            audio,
            player,
            speak_rejections=not suppress_repairs,
        ):
            if event.type == "llm_token":
                if not response_open:
                    console.print("[blue]SoCa:[/blue] ", end="")
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

        completed_turns += 1

    return 0


def _turn_streaming(
    pipeline: Any,
    audio: np.ndarray,
    player: AudioSink,
    *,
    speak_rejections: bool,
):
    kwargs: dict[str, Any] = {"audio_sink": player}
    try:
        signature = inspect.signature(pipeline.turn_streaming)
    except (TypeError, ValueError):
        signature = None
    supports_extra_kwargs = (
        signature is not None
        and any(param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    )
    if signature is not None and (
        "speak_rejections" in signature.parameters or supports_extra_kwargs
    ):
        kwargs["speak_rejections"] = speak_rejections
    return pipeline.turn_streaming(audio, **kwargs)
