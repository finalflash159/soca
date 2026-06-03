from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from soca.asr import ASR_MODEL_REGISTRY, SpeechDetector
from soca.asr.robust_asr import RobustASR
from soca.asr.whisper_onnx import VietnameseASR
from soca.core.pipeline import VoicePipeline
from soca.core.profiles import get_voice_runtime_profile
from soca.core.repair import default_repair_catalog
from soca.core.runtime import AssistantRuntime, DefaultRuntimeToolRouter, RuntimeOptions
from soca.knowledge import KnowledgeContextBuilder, MarkdownVaultKnowledgeSource
from soca.llm import LocalLlamaCppLLM
from soca.llm.registry import LLM_MODEL_REGISTRY
from soca.memory import MarkdownLongTermMemory, MemoryContextBuilder, SessionMemory
from soca.tools import (
    KnowledgeReadTool,
    KnowledgeSearchTool,
    LocalTimeTool,
    ToolRuntime,
)
from soca.tts import TTS_MODEL_REGISTRY, create_tts_engine


@dataclass(frozen=True)
class ResolvedVoiceRuntimeConfig:
    profile_key: str
    asr_model: str
    llm_model: str
    tts_model: str
    tts_voice: str | None
    endpoint_silence_ms: int
    max_record_ms: int
    max_tokens: int
    temperature: float
    top_p: float
    vault: Path
    no_memory: bool = False
    memory_chars: int = 2200
    profile_chars: int = 900
    session_chars: int = 1300
    session_turns: int = 6
    turn_chars: int = 500
    llm_threads: int = 8
    llm_gpu_layers: int = -1


@dataclass
class VoiceRuntimeBundle:
    config: ResolvedVoiceRuntimeConfig
    detector: SpeechDetector
    asr: RobustASR
    llm: LocalLlamaCppLLM
    tts: object
    assistant_runtime: AssistantRuntime
    pipeline: VoicePipeline
    memory_status: str
    knowledge_status: str
    session_memory: SessionMemory | None = None

    @property
    def asr_guard_status(self) -> str:
        return f"BoH={self.asr.boh_status}; confidence={self.asr.confidence_guard_status}"


@dataclass(frozen=True)
class VoiceRuntimeWarmupResult:
    component: str
    ok: bool
    latency_ms: float
    detail: str = ""


def resolve_voice_runtime_config(
    *,
    profile_key: str,
    asr_model: str | None = None,
    llm_model: str | None = None,
    tts_model: str | None = None,
    tts_voice: str | None = None,
    endpoint_silence_ms: int | None = None,
    max_record_ms: int | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    vault: str | Path | None = None,
    no_memory: bool = False,
    memory_chars: int = 2200,
    profile_chars: int = 900,
    session_chars: int = 1300,
    session_turns: int = 6,
    turn_chars: int = 500,
    llm_threads: int = 8,
    llm_gpu_layers: int = -1,
) -> ResolvedVoiceRuntimeConfig:
    profile = get_voice_runtime_profile(profile_key)
    resolved_tts_model = tts_model or profile.tts_model

    if resolved_tts_model not in TTS_MODEL_REGISTRY:
        valid = ", ".join(sorted(TTS_MODEL_REGISTRY))
        raise ValueError(f"Unknown TTS model key: {resolved_tts_model}. Valid keys: {valid}")

    resolved_asr_model = asr_model or profile.asr_model
    if resolved_asr_model not in ASR_MODEL_REGISTRY:
        valid = ", ".join(sorted(ASR_MODEL_REGISTRY))
        raise ValueError(f"Unknown ASR model key: {resolved_asr_model}. Valid keys: {valid}")

    resolved_llm_model = llm_model or profile.llm_model
    if resolved_llm_model not in LLM_MODEL_REGISTRY:
        valid = ", ".join(sorted(LLM_MODEL_REGISTRY))
        raise ValueError(f"Unknown LLM model key: {resolved_llm_model}. Valid keys: {valid}")

    if tts_voice is not None:
        resolved_tts_voice = tts_voice
    elif resolved_tts_model == profile.tts_model:
        resolved_tts_voice = profile.tts_voice or TTS_MODEL_REGISTRY[resolved_tts_model].default_voice
    else:
        resolved_tts_voice = TTS_MODEL_REGISTRY[resolved_tts_model].default_voice

    return ResolvedVoiceRuntimeConfig(
        profile_key=profile_key,
        asr_model=resolved_asr_model,
        llm_model=resolved_llm_model,
        tts_model=resolved_tts_model,
        tts_voice=resolved_tts_voice,
        endpoint_silence_ms=(
            endpoint_silence_ms if endpoint_silence_ms is not None else profile.endpoint_silence_ms
        ),
        max_record_ms=max_record_ms if max_record_ms is not None else profile.max_record_ms,
        max_tokens=max_tokens if max_tokens is not None else profile.max_tokens,
        temperature=temperature if temperature is not None else profile.temperature,
        top_p=top_p if top_p is not None else profile.top_p,
        vault=Path(vault or Path.home() / "KnowledgeVault").expanduser().resolve(),
        no_memory=no_memory,
        memory_chars=memory_chars,
        profile_chars=profile_chars,
        session_chars=session_chars,
        session_turns=session_turns,
        turn_chars=turn_chars,
        llm_threads=llm_threads,
        llm_gpu_layers=llm_gpu_layers,
    )


def build_voice_runtime(
    config: ResolvedVoiceRuntimeConfig,
    *,
    session_memory: SessionMemory | None = None,
) -> VoiceRuntimeBundle:
    detector = SpeechDetector()
    asr = RobustASR(asr=VietnameseASR(model_key=config.asr_model), vad=detector)
    llm = LocalLlamaCppLLM(
        model_key=config.llm_model,
        n_threads=config.llm_threads,
        n_gpu_layers=config.llm_gpu_layers,
    )

    memory_status = "disabled"
    knowledge_status = "disabled"
    memory_builder = None
    knowledge_builder = None
    tools = [LocalTimeTool()]

    if config.vault.is_dir():
        source = MarkdownVaultKnowledgeSource(config.vault, include_globs=("wiki/**/*.md",))
        knowledge_builder = KnowledgeContextBuilder(source)
        tools.extend([KnowledgeSearchTool(source), KnowledgeReadTool(source)])
        knowledge_status = f"enabled:{config.vault / 'wiki'}"
    else:
        knowledge_status = f"disabled:not_found:{config.vault}"
        if not config.no_memory:
            memory_status = f"disabled:not_found:{config.vault}"

    if not config.no_memory and config.vault.is_dir():
        long_term_memory = MarkdownLongTermMemory(
            config.vault,
            max_chars=config.profile_chars,
        )
        session_memory = (
            session_memory
            if session_memory is not None
            else SessionMemory(
                max_turns=config.session_turns,
                max_chars=config.session_chars,
                max_turn_chars=config.turn_chars,
            )
        )
        memory_builder = MemoryContextBuilder(
            long_term=long_term_memory,
            session=session_memory,
            max_chars=config.memory_chars,
            profile_chars=config.profile_chars,
        )
        memory_status = f"enabled:{config.vault / 'memory' / 'profile.md'}"

    tool_runtime = ToolRuntime(tools)
    assistant_runtime = AssistantRuntime(
        llm=llm,
        tool_runtime=tool_runtime,
        tool_router=DefaultRuntimeToolRouter(
            knowledge_search_prefixes=("wiki:", "knowledge:", "wiki ", "knowledge ")
        ),
        knowledge_builder=knowledge_builder,
        memory_builder=memory_builder,
        options=RuntimeOptions(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        ),
    )

    tts = create_tts_engine(config.tts_model, voice=config.tts_voice)
    pipeline = VoicePipeline(
        asr=asr,
        llm=llm,
        tts=tts,
        assistant_runtime=assistant_runtime,
        repair_catalog=default_repair_catalog(),
    )
    return VoiceRuntimeBundle(
        config=config,
        detector=detector,
        asr=asr,
        llm=llm,
        tts=tts,
        assistant_runtime=assistant_runtime,
        pipeline=pipeline,
        memory_status=memory_status,
        knowledge_status=knowledge_status,
        session_memory=session_memory if not config.no_memory else None,
    )


def warm_up_voice_runtime(
    bundle: VoiceRuntimeBundle,
    *,
    asr_seconds: float = 1.0,
    llm_prompt: str = "xin chào",
    tts_text: str = "Xin chào, tôi là SoCa.",
) -> tuple[VoiceRuntimeWarmupResult, ...]:
    """Eagerly trigger first-call paths before entering the live loop.

    Constructors load the major model artifacts, but several runtimes still do
    work on first inference: ONNX kernel setup, llama.cpp first token path,
    TTS speaker/voice prompt loading, or backend kernel caches. Warmup pays
    that cost once at startup instead of on the user's first spoken turn.
    """
    results = [
        _warm_up_asr(bundle, seconds=asr_seconds),
        _warm_up_llm(bundle, prompt=llm_prompt),
        _warm_up_tts(bundle, text=tts_text),
    ]
    return tuple(results)


def _warm_up_asr(bundle: VoiceRuntimeBundle, *, seconds: float) -> VoiceRuntimeWarmupResult:
    t0 = time.perf_counter()
    try:
        sample_rate = getattr(bundle.asr.asr, "SAMPLING_RATE", 16000)
        audio = np.zeros(max(int(sample_rate * seconds), 1), dtype=np.float32)
        bundle.asr.asr.transcribe(audio, max_new_tokens=1)
        return VoiceRuntimeWarmupResult(
            component="asr",
            ok=True,
            latency_ms=(time.perf_counter() - t0) * 1000,
            detail=bundle.config.asr_model,
        )
    except Exception as exc:
        return VoiceRuntimeWarmupResult(
            component="asr",
            ok=False,
            latency_ms=(time.perf_counter() - t0) * 1000,
            detail=str(exc),
        )


def _warm_up_llm(bundle: VoiceRuntimeBundle, *, prompt: str) -> VoiceRuntimeWarmupResult:
    t0 = time.perf_counter()
    try:
        bundle.llm.generate(
            prompt,
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            inject_persona=True,
        )
        return VoiceRuntimeWarmupResult(
            component="llm",
            ok=True,
            latency_ms=(time.perf_counter() - t0) * 1000,
            detail=bundle.config.llm_model,
        )
    except Exception as exc:
        return VoiceRuntimeWarmupResult(
            component="llm",
            ok=False,
            latency_ms=(time.perf_counter() - t0) * 1000,
            detail=str(exc),
        )


def _warm_up_tts(bundle: VoiceRuntimeBundle, *, text: str) -> VoiceRuntimeWarmupResult:
    t0 = time.perf_counter()
    try:
        bundle.tts.synthesize(text)
        return VoiceRuntimeWarmupResult(
            component="tts",
            ok=True,
            latency_ms=(time.perf_counter() - t0) * 1000,
            detail=f"{bundle.config.tts_model}:{bundle.config.tts_voice}",
        )
    except Exception as exc:
        return VoiceRuntimeWarmupResult(
            component="tts",
            ok=False,
            latency_ms=(time.perf_counter() - t0) * 1000,
            detail=str(exc),
        )
