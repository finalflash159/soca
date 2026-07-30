from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text as RichText

from soca.app.style.palette import ACCENT, ALT, BAD, BORDER, ICON, MUTED, TEXT, st
from soca.app.usage_view import print_turn_usage
from soca.config import DEFAULT_MAX_TOKENS, LlmSettings, SecretStore, load_settings
from soca.core import AssistantRuntime, DefaultRuntimeToolRouter, RuntimeOptions
from soca.core.knowledge_setup import build_knowledge_runtime_setup
from soca.core.memory_setup import (
    MemoryMode,
    MemoryRuntimeConfig,
    build_memory_runtime_setup,
)
from soca.core.profiles import DEFAULT_VOICE_RUNTIME_PROFILE_KEY, get_voice_runtime_profile
from soca.core.router_setup import build_runtime_tool_router
from soca.core.tool_routing import (
    RouterResponseMode,
    SemanticRouterConfig,
    ToolRouterConfig,
    ToolRouterMode,
)
from soca.core.turn import RuntimeResult
from soca.core.usage import TurnUsage
from soca.core.workflow import ActiveGoalStore, GoalCheckpointStore
from soca.knowledge.factory import DenseBackend, RetrievalConfig, RetrievalMode
from soca.knowledge.retrievers.dense import EmbeddingModel, FastEmbedModel
from soca.llm import LLMEngine, LocalLlamaCppLLM
from soca.llm.factory import SecretReader, build_llm_engine
from soca.llm.registry import LLM_MODEL_REGISTRY
from soca.memory import (
    SessionCheckpointStore,
    SessionMemory,
    SessionPersistence,
    WorkingMemoryPolicy,
    default_session_checkpoint_home,
)
from soca.tools import LocalTimeTool, MemorySearchTool, Tool, ToolRuntime


def default_text_llm_model_key() -> str:
    """Return the product default LLM from the default runtime profile."""
    return get_voice_runtime_profile(DEFAULT_VOICE_RUNTIME_PROFILE_KEY).llm_model


def default_semantic_router_examples() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "eval"
        / "prompts"
        / "p0"
        / "turn_routing_vi.jsonl"
    )


@dataclass(frozen=True)
class TextRuntimeConfig:
    profile_key: str = DEFAULT_VOICE_RUNTIME_PROFILE_KEY
    llm_model: str = field(default_factory=default_text_llm_model_key)
    llm_model_is_override: bool = False
    vault: Path = Path.home() / "KnowledgeVault"
    no_memory: bool = False
    no_llm: bool = False
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.2
    top_p: float = 0.95
    knowledge_limit: int = 3
    knowledge_retrieval_mode: str = "hybrid"
    knowledge_dense_backend: str = "aiteamvn_v2"
    memory_chars: int = 64_000
    profile_chars: int = 900
    session_chars: int = 60_000
    session_turns: int = 6
    turn_chars: int = 500
    session_persistence: SessionPersistence = "ram_only"
    session_id: str = "default"
    session_resume: bool = False
    llm_threads: int = 8
    llm_gpu_layers: int = -1
    tool_router_mode: str = "cascade"
    tool_router_response_mode: str = "prompt_json"
    semantic_router_enabled: bool = True
    semantic_router_threshold: float = 0.58
    semantic_router_margin: float = 0.0
    semantic_router_examples: Path | None = field(default_factory=default_semantic_router_examples)
    memory_mode: MemoryMode = "retrieved"
    memory_limit: int = 3
    memory_retrieval_mode: str = "chunk_sparse"
    memory_dense_backend: str = "aiteamvn_v2"
    memory_recency_weight: float = 0.20
    memory_importance_weight: float = 0.10
    memory_recency_half_life_days: float = 30.0


def resolve_text_runtime_config(
    *,
    profile_key: str = DEFAULT_VOICE_RUNTIME_PROFILE_KEY,
    llm_model: str | None = None,
    vault: str | Path | None = None,
    no_memory: bool = False,
    no_llm: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.2,
    top_p: float = 0.95,
    knowledge_limit: int = 3,
    memory_chars: int = 64_000,
    profile_chars: int = 900,
    session_chars: int = 60_000,
    session_turns: int = 6,
    turn_chars: int = 500,
    session_persistence: SessionPersistence = "ram_only",
    session_id: str = "default",
    session_resume: bool = False,
    llm_threads: int = 8,
    llm_gpu_layers: int = -1,
    tool_router_mode: str = "cascade",
    tool_router_response_mode: str = "prompt_json",
    semantic_router_enabled: bool = True,
    semantic_router_threshold: float = 0.58,
    semantic_router_margin: float = 0.0,
    semantic_router_examples: str | Path | None = None,
    memory_mode: str = "retrieved",
    memory_limit: int = 3,
    memory_retrieval_mode: str = "chunk_sparse",
    memory_dense_backend: str = "aiteamvn_v2",
    memory_recency_weight: float = 0.20,
    memory_importance_weight: float = 0.10,
    memory_recency_half_life_days: float = 30.0,
) -> TextRuntimeConfig:
    """Resolve text-only runtime config from the same profile source as voice.

    Product commands (`soca ask/chat/ui`) should use runtime profiles as the
    source of truth. The low-level LLM registry can still keep historical
    benchmark defaults such as PhoGPT without leaking into app defaults.
    """
    profile = get_voice_runtime_profile(profile_key)
    return TextRuntimeConfig(
        profile_key=profile_key,
        llm_model=llm_model or profile.llm_model,
        llm_model_is_override=llm_model is not None,
        vault=Path(vault or Path.home() / "KnowledgeVault").expanduser().resolve(),
        no_memory=no_memory,
        no_llm=no_llm,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        knowledge_limit=knowledge_limit,
        knowledge_retrieval_mode=profile.knowledge_retrieval_mode,
        knowledge_dense_backend=profile.knowledge_dense_backend,
        memory_chars=memory_chars,
        profile_chars=profile_chars,
        session_chars=session_chars,
        session_turns=session_turns,
        turn_chars=turn_chars,
        session_persistence=session_persistence,
        session_id=session_id,
        session_resume=session_resume,
        llm_threads=llm_threads,
        llm_gpu_layers=llm_gpu_layers,
        tool_router_mode=tool_router_mode,
        tool_router_response_mode=tool_router_response_mode,
        semantic_router_enabled=semantic_router_enabled,
        semantic_router_threshold=semantic_router_threshold,
        semantic_router_margin=semantic_router_margin,
        semantic_router_examples=(
            Path(semantic_router_examples).expanduser().resolve()
            if semantic_router_examples is not None
            else default_semantic_router_examples()
        ),
        memory_mode=cast(MemoryMode, memory_mode),
        memory_limit=memory_limit,
        memory_retrieval_mode=memory_retrieval_mode,
        memory_dense_backend=memory_dense_backend,
        memory_recency_weight=memory_recency_weight,
        memory_importance_weight=memory_importance_weight,
        memory_recency_half_life_days=memory_recency_half_life_days,
    )


@dataclass(frozen=True)
class TextRuntimeBundle:
    runtime: AssistantRuntime
    session_memory: SessionMemory | None
    llm_status: str
    knowledge_status: str
    memory_status: str


class LLMFactory(Protocol):
    def __call__(
        self,
        *,
        model_key: str,
        n_threads: int = 8,
        n_gpu_layers: int = -1,
    ) -> LLMEngine: ...


class LLMEngineFactory(Protocol):
    def __call__(
        self,
        settings: LlmSettings,
        secrets: SecretReader,
        *,
        local_factory: LLMFactory | None,
        n_threads: int,
        n_gpu_layers: int,
    ) -> LLMEngine: ...


def build_text_runtime(
    config: TextRuntimeConfig,
    *,
    llm_factory: LLMFactory | None = None,
    session_memory: SessionMemory | None = None,
    llm_settings: LlmSettings | None = None,
    secret_store: SecretReader | None = None,
    engine_factory: LLMEngineFactory = build_llm_engine,
    embedding_model: EmbeddingModel | None = None,
    active_goal_store: ActiveGoalStore | None = None,
) -> TextRuntimeBundle:
    """Build text-only AssistantRuntime without ASR or TTS.

    This is the shared builder for `soca ask` and the later interactive
    `soca chat` command. It deliberately keeps tool routing explicit:
    natural-language domain routing is left to AssistantRuntime/LLM, while
    deterministic tools are only selected for clear commands such as `wiki:`.
    """
    vault = config.vault.expanduser()
    knowledge_builder = None
    knowledge_status = "disabled:not_found"
    tools: list[Tool] = [LocalTimeTool()]

    if vault.is_dir():
        knowledge = build_knowledge_runtime_setup(
            vault,
            knowledge_limit=config.knowledge_limit,
            retrieval_config=RetrievalConfig(
                mode=cast(RetrievalMode, config.knowledge_retrieval_mode),
                dense_backend=cast(DenseBackend, config.knowledge_dense_backend),
            ),
        )
        knowledge_builder = knowledge.builder
        tools.extend([knowledge.search_tool, knowledge.read_tool])
        knowledge_status = knowledge.status

    selected_settings = llm_settings or load_settings()
    if config.llm_model_is_override:
        # An explicit CLI model key is a local runtime override. This
        # prevents a persisted remote UI selection from silently turning
        # `soca ask --llm-model ...` (and its tests) into a paid network call.
        selected_settings = selected_settings.with_backend("local").with_model(config.llm_model)
    model_context_window = (
        LLM_MODEL_REGISTRY[selected_settings.model_id].context_window
        if selected_settings.backend == "local"
        and selected_settings.model_id in LLM_MODEL_REGISTRY
        else selected_settings.model_context_window
    )
    working_policy = WorkingMemoryPolicy.for_context_budget(
        context_window=model_context_window,
        output_reserve_tokens=selected_settings.effective_max_tokens,
        mode="background_summary" if not config.no_llm else "trim_only",
    )

    runtime_session_memory = None
    memory_builder = None
    if config.no_memory:
        memory_status = "disabled"
    else:
        # Session memory is RAM-only and must work even without a vault so
        # `soca chat` keeps multi-turn context. Long-term profile memory is the
        # only part that needs the vault on disk.
        runtime_session_memory = (
            session_memory
            if session_memory is not None
            else SessionMemory(
                thread_id=config.session_id,
                max_turns=config.session_turns,
                max_chars=config.session_chars,
                max_turn_chars=config.turn_chars,
                summary_enabled=not config.no_llm,
                working_policy=working_policy,
                summary_threads=config.llm_threads,
                summary_gpu_layers=config.llm_gpu_layers,
                persistence=config.session_persistence,
                checkpoint_store=(
                    SessionCheckpointStore(default_session_checkpoint_home())
                    if config.session_persistence == "local_resumable"
                    else None
                ),
                resume=config.session_resume,
            )
        )
        memory_setup = build_memory_runtime_setup(
            vault,
            session=runtime_session_memory,
            config=MemoryRuntimeConfig(
                mode=config.memory_mode,
                top_k=config.memory_limit,
                context_chars=config.memory_chars,
                profile_chars=config.profile_chars,
                retrieval_mode=cast(RetrievalMode, config.memory_retrieval_mode),
                dense_backend=cast(DenseBackend, config.memory_dense_backend),
                recency_weight=config.memory_recency_weight,
                importance_weight=config.memory_importance_weight,
                relevance_weight=1.0 - config.memory_recency_weight - config.memory_importance_weight,
                recency_half_life_days=config.memory_recency_half_life_days,
            ),
        )
        memory_builder = memory_setup.builder
        memory_status = memory_setup.status
        tools.append(MemorySearchTool(memory_builder, max_limit=config.knowledge_limit))

    if config.no_llm:
        llm = None
        llm_status = "disabled"
    else:
        llm = engine_factory(
            selected_settings,
            secret_store or SecretStore(),
            local_factory=llm_factory or LocalLlamaCppLLM,
            n_threads=config.llm_threads,
            n_gpu_layers=config.llm_gpu_layers,
        )
        if selected_settings.backend == "remote":
            llm_status = f"enabled:{selected_settings.provider_key}:{selected_settings.model_id}"
        else:
            llm_status = f"enabled:{selected_settings.model_id}"

    tool_runtime = ToolRuntime(tools)
    router_config = ToolRouterConfig(
        mode=cast(ToolRouterMode, config.tool_router_mode),
        response_mode=cast(RouterResponseMode, config.tool_router_response_mode),
        semantic=SemanticRouterConfig(
            enabled=config.semantic_router_enabled,
            threshold=config.semantic_router_threshold,
            margin=config.semantic_router_margin,
            examples_path=config.semantic_router_examples,
        ),
    )
    deterministic_router = DefaultRuntimeToolRouter(
        enable_memory_search=memory_builder is not None,
    )
    router_embedding_model = embedding_model
    if router_config.semantic.enabled and router_embedding_model is None:
        try:
            router_embedding_model = FastEmbedModel(allow_download=False)
        except (ImportError, FileNotFoundError, OSError, RuntimeError, ValueError):
            router_embedding_model = None
    tool_router = build_runtime_tool_router(
        llm=llm,
        tool_runtime=tool_runtime,
        deterministic=deterministic_router,
        config=router_config,
        embedding_model=router_embedding_model,
        voice=False,
    )
    runtime = AssistantRuntime(
        llm=llm,
        tool_runtime=tool_runtime,
        tool_router=tool_router,
        knowledge_builder=knowledge_builder,
        memory_builder=memory_builder,
        options=RuntimeOptions(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            knowledge_limit=config.knowledge_limit,
            model_context_window=model_context_window,
            model_max_output_tokens=selected_settings.model_max_output_tokens,
        ),
        active_goal_store=active_goal_store
        or (
            ActiveGoalStore(
                checkpoint_store=GoalCheckpointStore(default_session_checkpoint_home() / "goals"),
                session_id=config.session_id,
            )
            if config.session_persistence == "local_resumable"
            else None
        ),
    )
    return TextRuntimeBundle(
        runtime=runtime,
        session_memory=runtime_session_memory,
        llm_status=llm_status,
        knowledge_status=knowledge_status,
        memory_status=memory_status,
    )


def run_text_ask(
    text: str,
    config: TextRuntimeConfig,
    *,
    console: Console,
    show_trace: bool = False,
    show_usage: bool = False,
    runtime_builder=build_text_runtime,
) -> RuntimeResult:
    bundle = runtime_builder(config)

    header = RichText()
    header.append(f"{ICON.BIRD} ", style=st(f"bold {ACCENT}"))
    header.append("SoCa", style=st(f"bold {ACCENT}"))
    header.append(" · text", style=st(MUTED))
    console.print(header)
    console.print(
        RichText(
            f"    LLM {bundle.llm_status} {ICON.DOT} knowledge {bundle.knowledge_status}"
            f" {ICON.DOT} memory {bundle.memory_status}",
            style=st(MUTED),
        )
    )

    user_text, metadata = normalize_text_turn(text)
    result = bundle.runtime.run_text_turn(user_text, source="cli", metadata=metadata)
    render_text_result(console, result, show_trace=show_trace)
    if show_usage:
        print_turn_usage(console, TurnUsage.from_runtime_result(result))
    return result


def render_text_result(
    console: Console,
    result: RuntimeResult,
    *,
    show_trace: bool = False,
) -> None:
    reply = RichText()
    reply.append(f"{ICON.BIRD} ", style=st(f"bold {ACCENT}"))
    reply.append(result.response_text or "<empty>", style=st(TEXT))
    console.print(reply)
    note_style = BAD if result.blocked else MUTED
    console.print(RichText(f"  {ICON.DOT} Route: {result.route.value}", style=st(note_style)))

    if result.citations:
        citations = Table(title="Citations")
        citations.add_column("Ref", style=st(ALT) or "none")
        citations.add_column("Path")
        citations.add_column("Title")
        for index, citation in enumerate(result.citations, start=1):
            citations.add_row(f"K{index}", citation.path, citation.title)
        console.print(citations)

    if show_trace:
        render_trace(console, result)


def render_trace(console: Console, result: RuntimeResult) -> None:
    trace = result.trace
    if trace is None:
        console.print(Panel("<none>", title="Trace", border_style=st(BORDER) or "none"))
        return

    summary = Table(title="Trace Summary")
    summary.add_column("Field")
    summary.add_column("Value")
    summary.add_row("route", trace.route.value)
    summary.add_row("blocked", str(trace.blocked))
    summary.add_row("used_tool", str(trace.used_tool))
    summary.add_row("used_llm", str(trace.used_llm))
    summary.add_row("router_tier", trace.tool_router_tier)
    summary.add_row("router_reason", trace.tool_router_reason)
    summary.add_row("evidence_status", trace.evidence_status)
    summary.add_row("answer_policy", trace.answer_policy)
    summary.add_row("citation_count", str(trace.citation_count))
    if trace.memory_access_plan is not None:
        summary.add_row(
            "memory_access",
            f"{trace.memory_access_plan.archive_mode}:{trace.memory_access_plan.reason}",
        )
    console.print(summary)

    if trace.tool_calls:
        tools = Table(title="Tool Calls")
        tools.add_column("Tool")
        tools.add_column("Arguments")
        for call in trace.tool_calls:
            tools.add_row(call.name, str(call.arguments))
        console.print(tools)

    if trace.tool_results:
        results = Table(title="Tool Results")
        results.add_column("Tool")
        results.add_column("OK")
        results.add_column("Error")
        for item in trace.tool_results:
            results.add_row(item.name, str(item.ok), item.error or "-")
        console.print(results)

    guardrails = Table(title="Guardrail Events")
    guardrails.add_column("Stage")
    guardrails.add_column("Action")
    guardrails.add_column("Reason", overflow="fold")
    guardrails.add_column("Message", overflow="fold")
    for event in trace.guardrail_events:
        guardrails.add_row(
            event.stage.value,
            event.action.value,
            event.reason or "-",
            event.message or "-",
        )
    console.print(guardrails)
    # Wrap-safe raw line so long guardrail reasons stay greppable even when the
    # table folds them at narrow console widths.
    console.print(
        Panel(
            "\n".join(
                f"{event.stage.value}:{event.action.value}:{event.reason or '-'}"
                for event in trace.guardrail_events
            )
            or "<none>",
            title="Guardrail Raw",
            border_style=st(BORDER) or "none",
        )
    )

    if trace.stage_latencies_ms:
        latencies = Table(title="Stage Latencies")
        latencies.add_column("Stage")
        latencies.add_column("ms", justify="right")
        for stage, latency_ms in trace.stage_latencies_ms.items():
            latencies.add_row(stage, f"{latency_ms:.1f}")
        console.print(latencies)


def normalize_text_turn(text: str) -> tuple[str, dict[str, object]]:
    """Normalize a CLI text turn, extracting the `/k ` knowledge-context prefix.

    Shared by both `soca ask` and `soca chat` so the routing prefix behaves
    identically across the two entrypoints.
    """
    stripped = text.strip()
    metadata: dict[str, object] = {}
    if stripped.startswith("/k "):
        stripped = stripped[3:].strip()
        metadata["use_knowledge"] = True
    return stripped, metadata
