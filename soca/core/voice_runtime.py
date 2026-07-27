from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from soca.asr import ASR_MODEL_REGISTRY, SpeechDetector
from soca.asr.robust_asr import RobustASR, load_confidence_guard_calibration
from soca.asr.whisper_onnx import VietnameseASR
from soca.core.knowledge_setup import build_knowledge_runtime_setup
from soca.core.pipeline import VoicePipeline
from soca.core.profiles import get_voice_runtime_profile
from soca.core.repair import default_repair_catalog
from soca.core.runtime import AssistantRuntime, DefaultRuntimeToolRouter, RuntimeOptions
from soca.core.smart_turn import SmartTurnDetector
from soca.core.turn_taking import partial_interval_from_cost
from soca.knowledge.factory import RetrievalConfig
from soca.knowledge.hybrid_source import HybridKnowledgeSource
from soca.knowledge.intent_gate import RetrievalIntentGate
from soca.llm import LocalLlamaCppLLM
from soca.llm.registry import LLM_MODEL_REGISTRY
from soca.memory import MarkdownLongTermMemory, MemoryContextBuilder, SessionMemory
from soca.tools import LocalTimeTool, ToolRuntime
from soca.tts import VALTEC_TTS_CONFIG, create_tts_engine


@dataclass(frozen=True)
class ResolvedVoiceRuntimeConfig:
    profile_key: str
    asr_model: str
    llm_model: str
    tts_voice: str | None
    endpoint_silence_ms: int
    adaptive_endpoint: bool
    max_record_ms: int
    max_tokens: int
    temperature: float
    top_p: float
    first_clause_enabled: bool
    first_clause_min_chars: int
    first_clause_min_words: int
    first_clause_max_scan_chars: int
    pcm_crossfade_enabled: bool
    pcm_crossfade_ms: float
    vault: Path
    no_memory: bool = False
    memory_chars: int = 2200
    profile_chars: int = 900
    session_chars: int = 1300
    session_turns: int = 6
    turn_chars: int = 500
    llm_threads: int = 8
    llm_gpu_layers: int = -1
    knowledge_limit: int = 3
    knowledge_retrieval_mode: str = "cached_sparse"
    knowledge_dense_backend: str = "fastembed"
    voice_knowledge_mode: str = "off"
    knowledge_intent_threshold: float | None = None


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
    turn_detector: SmartTurnDetector | None = None
    session_memory: SessionMemory | None = None
    partial_interval_ms: int = 800  # partial cadence seed (handles device variance)
    partial_enabled: bool = True  # False when the device is too slow for partials

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
    tts_voice: str | None = None,
    endpoint_silence_ms: int | None = None,
    adaptive_endpoint: bool | None = None,
    max_record_ms: int | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    first_clause_enabled: bool | None = None,
    vault: str | Path | None = None,
    no_memory: bool = False,
    memory_chars: int = 2200,
    profile_chars: int = 900,
    session_chars: int = 1300,
    session_turns: int = 6,
    turn_chars: int = 500,
    llm_threads: int = 8,
    llm_gpu_layers: int = -1,
    knowledge_limit: int | None = None,
    knowledge_retrieval_mode: str | None = None,
    knowledge_dense_backend: str | None = None,
    voice_knowledge_mode: str | None = None,
    knowledge_intent_threshold: float | None = None,
) -> ResolvedVoiceRuntimeConfig:
    profile = get_voice_runtime_profile(profile_key)

    resolved_asr_model = asr_model or profile.asr_model
    if resolved_asr_model not in ASR_MODEL_REGISTRY:
        valid = ", ".join(sorted(ASR_MODEL_REGISTRY))
        raise ValueError(f"Unknown ASR model key: {resolved_asr_model}. Valid keys: {valid}")

    resolved_llm_model = llm_model or profile.llm_model
    if resolved_llm_model not in LLM_MODEL_REGISTRY:
        valid = ", ".join(sorted(LLM_MODEL_REGISTRY))
        raise ValueError(f"Unknown LLM model key: {resolved_llm_model}. Valid keys: {valid}")

    resolved_tts_voice = tts_voice or profile.tts_voice or VALTEC_TTS_CONFIG.default_voice
    if resolved_tts_voice not in VALTEC_TTS_CONFIG.voices:
        valid = ", ".join(VALTEC_TTS_CONFIG.voices)
        raise ValueError(f"Unknown Valtec voice: {resolved_tts_voice!r}. Valid voices: {valid}")

    resolved_limit = knowledge_limit if knowledge_limit is not None else profile.knowledge_limit
    resolved_retrieval = knowledge_retrieval_mode or profile.knowledge_retrieval_mode
    resolved_backend = knowledge_dense_backend or profile.knowledge_dense_backend
    resolved_voice_mode = voice_knowledge_mode or profile.voice_knowledge_mode
    resolved_threshold = (
        knowledge_intent_threshold
        if knowledge_intent_threshold is not None
        else profile.knowledge_intent_threshold
    )
    if (
        isinstance(resolved_limit, bool)
        or not isinstance(resolved_limit, int)
        or resolved_limit < 1
    ):
        raise ValueError("knowledge_limit must be positive")
    if resolved_retrieval not in {"cached_sparse", "hybrid"}:
        raise ValueError("unknown knowledge retrieval mode")
    if resolved_backend not in {"fastembed", "model2vec"}:
        raise ValueError("unknown knowledge dense backend")
    if resolved_voice_mode not in {"off", "intent", "always"}:
        raise ValueError("unknown voice knowledge mode")
    if resolved_threshold is not None and (
        isinstance(resolved_threshold, bool)
        or not isinstance(resolved_threshold, (int, float))
        or not 0 <= resolved_threshold <= 1
    ):
        raise ValueError("knowledge_intent_threshold must be a number between 0 and 1")
    if resolved_voice_mode == "intent" and resolved_threshold is None:
        raise ValueError("intent mode requires knowledge_intent_threshold")
    if resolved_voice_mode == "intent" and resolved_retrieval != "hybrid":
        raise ValueError("intent mode requires hybrid retrieval")

    return ResolvedVoiceRuntimeConfig(
        profile_key=profile_key,
        asr_model=resolved_asr_model,
        llm_model=resolved_llm_model,
        tts_voice=resolved_tts_voice,
        endpoint_silence_ms=(
            endpoint_silence_ms if endpoint_silence_ms is not None else profile.endpoint_silence_ms
        ),
        adaptive_endpoint=(
            adaptive_endpoint if adaptive_endpoint is not None else profile.adaptive_endpoint
        ),
        max_record_ms=(max_record_ms if max_record_ms is not None else profile.max_record_ms),
        max_tokens=max_tokens if max_tokens is not None else profile.max_tokens,
        temperature=(temperature if temperature is not None else profile.temperature),
        top_p=top_p if top_p is not None else profile.top_p,
        first_clause_enabled=(
            first_clause_enabled
            if first_clause_enabled is not None
            else profile.first_clause_enabled
        ),
        first_clause_min_chars=profile.first_clause_min_chars,
        first_clause_min_words=profile.first_clause_min_words,
        first_clause_max_scan_chars=profile.first_clause_max_scan_chars,
        pcm_crossfade_enabled=profile.pcm_crossfade_enabled,
        pcm_crossfade_ms=profile.pcm_crossfade_ms,
        vault=Path(vault or Path.home() / "KnowledgeVault").expanduser().resolve(),
        no_memory=no_memory,
        memory_chars=memory_chars,
        profile_chars=profile_chars,
        session_chars=session_chars,
        session_turns=session_turns,
        turn_chars=turn_chars,
        llm_threads=llm_threads,
        llm_gpu_layers=llm_gpu_layers,
        knowledge_limit=resolved_limit,
        knowledge_retrieval_mode=resolved_retrieval,
        knowledge_dense_backend=resolved_backend,
        voice_knowledge_mode=resolved_voice_mode,
        knowledge_intent_threshold=resolved_threshold,
    )


def build_voice_runtime(
    config: ResolvedVoiceRuntimeConfig,
    *,
    session_memory: SessionMemory | None = None,
) -> VoiceRuntimeBundle:
    detector = SpeechDetector()
    turn_detector = None
    if config.adaptive_endpoint:
        model_dir = _smart_turn_model_dir()
        turn_detector = SmartTurnDetector(model_dir=model_dir)

    confidence_calibration = load_confidence_guard_calibration(config.asr_model)
    if confidence_calibration is None:
        asr = RobustASR(
            asr=VietnameseASR(model_key=config.asr_model),
            vad=detector,
            confidence_guard_skip_reason=f"skipped:missing_for_model:{config.asr_model}",
        )
    else:
        asr = RobustASR(
            asr=VietnameseASR(model_key=config.asr_model),
            vad=detector,
            min_avg_logprob=confidence_calibration.min_avg_logprob,
            max_compression_ratio=confidence_calibration.max_compression_ratio,
            confidence_profile_model_key=confidence_calibration.model_key,
        )
    llm = LocalLlamaCppLLM(
        model_key=config.llm_model,
        n_threads=config.llm_threads,
        n_gpu_layers=config.llm_gpu_layers,
    )

    memory_status = "disabled"
    knowledge_limit = config.knowledge_limit
    knowledge_status = "disabled:not_found"
    memory_builder = None
    knowledge_builder = None
    tools = [LocalTimeTool()]

    knowledge_intent_gate = None
    effective_voice_mode = config.voice_knowledge_mode
    if config.vault.is_dir():
        knowledge = build_knowledge_runtime_setup(
            config.vault,
            knowledge_limit=knowledge_limit,
            retrieval_config=RetrievalConfig(
                mode=config.knowledge_retrieval_mode,
                dense_backend=config.knowledge_dense_backend,
            ),
        )
        knowledge_builder = knowledge.builder
        tools.extend([knowledge.search_tool, knowledge.read_tool])
        knowledge_status = knowledge.status
        if (
            effective_voice_mode == "intent"
            and isinstance(knowledge.source, HybridKnowledgeSource)
            and config.knowledge_intent_threshold is not None
        ):
            knowledge_intent_gate = RetrievalIntentGate(
                knowledge.source,
                threshold=config.knowledge_intent_threshold,
            )
        elif effective_voice_mode == "intent":
            effective_voice_mode = "off"
    else:
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
            knowledge_limit=knowledge_limit,
            voice_knowledge_mode=effective_voice_mode,
        ),
        knowledge_intent_gate=knowledge_intent_gate,
    )

    tts = create_tts_engine(voice=config.tts_voice)
    pipeline = VoicePipeline(
        asr=asr,
        llm=llm,
        tts=tts,
        assistant_runtime=assistant_runtime,
        repair_catalog=default_repair_catalog(),
        first_clause_enabled=config.first_clause_enabled,
        first_clause_min_chars=config.first_clause_min_chars,
        first_clause_min_words=config.first_clause_min_words,
        first_clause_max_scan_chars=config.first_clause_max_scan_chars,
        pcm_crossfade_ms=(config.pcm_crossfade_ms if config.pcm_crossfade_enabled else 0.0),
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
        turn_detector=turn_detector,
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
    if bundle.turn_detector is not None:
        results.append(_warm_up_smart_turn(bundle))
    return tuple(results)


def _smart_turn_model_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / "smart-turn-v3-onnx"


def _warm_up_asr(bundle: VoiceRuntimeBundle, *, seconds: float) -> VoiceRuntimeWarmupResult:
    t0 = time.perf_counter()
    try:
        sample_rate = getattr(bundle.asr.asr, "SAMPLING_RATE", 16000)
        audio = np.zeros(max(int(sample_rate * seconds), 1), dtype=np.float32)
        bundle.asr.asr.transcribe(audio, max_new_tokens=1)  # kernel warm
        # --- calibrate partial cadence: one REPRESENTATIVE decode (NOT max_new_tokens=1,
        #     since 1 token does not measure decoder cost) ---
        probe = (np.random.randn(sample_rate * 3) * 0.01).astype(np.float32)
        c0 = time.perf_counter()
        bundle.asr.asr.transcribe(probe)  # real decode
        per_call_ms = (time.perf_counter() - c0) * 1000
        interval, enabled = partial_interval_from_cost(per_call_ms, os.cpu_count())
        bundle.partial_interval_ms = interval
        bundle.partial_enabled = enabled
        detail = f"{bundle.config.asr_model} · partial={interval}ms{'' if enabled else ' (off)'}"
        return VoiceRuntimeWarmupResult(
            component="asr",
            ok=True,
            latency_ms=(time.perf_counter() - t0) * 1000,
            detail=detail,
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
            detail=f"{VALTEC_TTS_CONFIG.key}:{bundle.config.tts_voice}",
        )
    except Exception as exc:
        return VoiceRuntimeWarmupResult(
            component="tts",
            ok=False,
            latency_ms=(time.perf_counter() - t0) * 1000,
            detail=str(exc),
        )


def _warm_up_smart_turn(bundle: VoiceRuntimeBundle) -> VoiceRuntimeWarmupResult:
    t0 = time.perf_counter()
    bundle.turn_detector.warmup()
    return VoiceRuntimeWarmupResult(
        component="smart_turn",
        ok=True,
        latency_ms=(time.perf_counter() - t0) * 1000,
        detail="smart-turn-v3.2-cpu.onnx",
    )
