from __future__ import annotations

from collections.abc import Callable

import numpy as np
from rich.console import Console

from soca.app.console import (
    print_recorded_audio,
    print_rejection_fallback,
    print_runtime_header,
    print_streaming_event,
    print_waiting_for_speech,
    print_warmup_result,
    print_warmup_start,
)
from soca.core import AudioSink, EndpointConfig, SoundDevicePlayer, record_until_silence
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
    press_enter_to_record: bool = False,
    warmup: bool = True,
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
        for event in bundle.pipeline.turn_streaming(audio, audio_sink=player):
            if event.type == "llm_token":
                if not response_open:
                    console.print("[blue]SoCa:[/blue] ", end="")
                    response_open = True
                console.print(event.text, end="", markup=False, highlight=False, soft_wrap=True)
                continue

            streamed_this_turn = response_open
            if response_open:
                console.print()  # close the live response line
                response_open = False

            # When tokens already streamed live, the runtime panel would just
            # repeat the same text, so skip it.
            if event.type == "runtime" and streamed_this_turn:
                continue

            rejected = bool(event.metadata and event.metadata.get("rejected"))
            if event.type == "done" and rejected and event.text and not no_speak_rejections:
                print_rejection_fallback(console, event.text)
                tts_result = bundle.tts.synthesize(event.text)
                player.play(tts_result.audio, tts_result.sample_rate, blocking=True)

            print_streaming_event(console, event)

        completed_turns += 1

    return 0
