from __future__ import annotations

from typing import TYPE_CHECKING

from .audio_out import (
    AudioPlaybackSession,
    AudioSink,
    NullAudioPlayer,
    PlaybackResult,
    SoundDevicePlayer,
    StreamingAudioSink,
    WavFileSink,
)
from .endpoint import EndpointConfig, block_samples, record_until_silence, should_stop_recording
from .guardrails import (
    GuardrailAction,
    GuardrailEvent,
    GuardrailPolicy,
    GuardrailStage,
    check_final_output,
    check_input_text,
    check_knowledge_read_path,
    check_tool_call,
    check_tool_result,
    check_untrusted_text,
)
from .metrics import MetricsLogger
from .pipeline import PipelineResult, VoicePipeline
from .profiles import (
    DEFAULT_VOICE_RUNTIME_PROFILE_KEY,
    VOICE_RUNTIME_PROFILES,
    VoiceRuntimeProfile,
    get_voice_runtime_profile,
    validate_voice_runtime_profiles,
)
from .streaming import StreamingEvent, pop_ready_sentence
from .text_chunking import chunk_text_for_tts, split_sentences
from .tool_routing import (
    ParsedToolDecision,
    RouterOutputError,
    SemanticRouterConfig,
    ToolRouterConfig,
    ToolRouterDecision,
    build_tool_decision_schema,
    parse_tool_decision,
)
from .turn import (
    RuntimeResult,
    RuntimeRoute,
    RuntimeStreamEvent,
    RuntimeTrace,
    TurnFrame,
)
from .usage import LLMUsage, SessionUsage, TurnUsage

if TYPE_CHECKING:
    from .runtime import (
        AssistantRuntime,
        DefaultRuntimeToolRouter,
        RuntimeOptions,
        RuntimeToolRouter,
    )
    from .voice_runtime import (
        ResolvedVoiceRuntimeConfig,
        VoiceRuntimeBundle,
        VoiceRuntimeWarmupResult,
        build_voice_runtime,
        resolve_voice_runtime_config,
        warm_up_voice_runtime,
    )


def __getattr__(name: str):
    if name in {"AssistantRuntime", "DefaultRuntimeToolRouter", "RuntimeOptions", "RuntimeToolRouter"}:
        from .runtime import (
            AssistantRuntime,
            DefaultRuntimeToolRouter,
            RuntimeOptions,
            RuntimeToolRouter,
        )

        return {
            "AssistantRuntime": AssistantRuntime,
            "DefaultRuntimeToolRouter": DefaultRuntimeToolRouter,
            "RuntimeOptions": RuntimeOptions,
            "RuntimeToolRouter": RuntimeToolRouter,
        }[name]
    if name in {
        "ResolvedVoiceRuntimeConfig",
        "VoiceRuntimeBundle",
        "VoiceRuntimeWarmupResult",
        "build_voice_runtime",
        "resolve_voice_runtime_config",
        "warm_up_voice_runtime",
    }:
        from .voice_runtime import (
            ResolvedVoiceRuntimeConfig,
            VoiceRuntimeBundle,
            VoiceRuntimeWarmupResult,
            build_voice_runtime,
            resolve_voice_runtime_config,
            warm_up_voice_runtime,
        )

        return {
            "ResolvedVoiceRuntimeConfig": ResolvedVoiceRuntimeConfig,
            "VoiceRuntimeBundle": VoiceRuntimeBundle,
            "VoiceRuntimeWarmupResult": VoiceRuntimeWarmupResult,
            "build_voice_runtime": build_voice_runtime,
            "resolve_voice_runtime_config": resolve_voice_runtime_config,
            "warm_up_voice_runtime": warm_up_voice_runtime,
        }[name]
    raise AttributeError(name)

__all__ = [
    "AudioPlaybackSession",
    "AudioSink",
    "StreamingAudioSink",
    "AssistantRuntime",
    "DefaultRuntimeToolRouter",
    "ToolRouterConfig",
    "SemanticRouterConfig",
    "ToolRouterDecision",
    "ParsedToolDecision",
    "RouterOutputError",
    "build_tool_decision_schema",
    "parse_tool_decision",
    "EndpointConfig",
    "GuardrailAction",
    "GuardrailEvent",
    "GuardrailPolicy",
    "GuardrailStage",
    "MetricsLogger",
    "NullAudioPlayer",
    "PlaybackResult",
    "RuntimeOptions",
    "LLMUsage",
    "SessionUsage",
    "TurnUsage",
    "RuntimeResult",
    "RuntimeRoute",
    "RuntimeStreamEvent",
    "RuntimeTrace",
    "RuntimeToolRouter",
    "ResolvedVoiceRuntimeConfig",
    "SoundDevicePlayer",
    "WavFileSink",
    "PipelineResult",
    "TurnFrame",
    "VoicePipeline",
    "VoiceRuntimeBundle",
    "VoiceRuntimeWarmupResult",
    "block_samples",
    "check_final_output",
    "check_input_text",
    "check_knowledge_read_path",
    "check_tool_call",
    "check_tool_result",
    "check_untrusted_text",
    "record_until_silence",
    "should_stop_recording",
    "split_sentences",
    "chunk_text_for_tts",
    "StreamingEvent",
    "pop_ready_sentence",
    "DEFAULT_VOICE_RUNTIME_PROFILE_KEY",
    "VOICE_RUNTIME_PROFILES",
    "VoiceRuntimeProfile",
    "get_voice_runtime_profile",
    "validate_voice_runtime_profiles",
    "build_voice_runtime",
    "resolve_voice_runtime_config",
    "warm_up_voice_runtime",
]
