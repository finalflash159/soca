from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from soca.app.usage_view import print_turn_usage
from soca.core import AssistantRuntime, DefaultRuntimeToolRouter, RuntimeOptions
from soca.core.profiles import DEFAULT_VOICE_RUNTIME_PROFILE_KEY, get_voice_runtime_profile
from soca.core.turn import RuntimeResult
from soca.core.usage import TurnUsage
from soca.knowledge import KnowledgeContextBuilder, MarkdownVaultKnowledgeSource
from soca.llm import LLMEngine, LocalLlamaCppLLM
from soca.memory import MarkdownLongTermMemory, MemoryContextBuilder, SessionMemory
from soca.tools import KnowledgeReadTool, KnowledgeSearchTool, LocalTimeTool, ToolRuntime


def default_text_llm_model_key() -> str:
    """Return the product default LLM from the default runtime profile."""
    return get_voice_runtime_profile(DEFAULT_VOICE_RUNTIME_PROFILE_KEY).llm_model


@dataclass(frozen=True)
class TextRuntimeConfig:
    profile_key: str = DEFAULT_VOICE_RUNTIME_PROFILE_KEY
    llm_model: str = field(default_factory=default_text_llm_model_key)
    vault: Path = Path.home() / "KnowledgeVault"
    no_memory: bool = False
    no_llm: bool = False
    max_tokens: int = 160
    temperature: float = 0.2
    top_p: float = 0.95
    knowledge_limit: int = 3
    memory_chars: int = 2200
    profile_chars: int = 900
    session_chars: int = 1300
    session_turns: int = 6
    turn_chars: int = 500
    llm_threads: int = 8
    llm_gpu_layers: int = -1


def resolve_text_runtime_config(
    *,
    profile_key: str = DEFAULT_VOICE_RUNTIME_PROFILE_KEY,
    llm_model: str | None = None,
    vault: str | Path | None = None,
    no_memory: bool = False,
    no_llm: bool = False,
    max_tokens: int = 160,
    temperature: float = 0.2,
    top_p: float = 0.95,
    knowledge_limit: int = 3,
    memory_chars: int = 2200,
    profile_chars: int = 900,
    session_chars: int = 1300,
    session_turns: int = 6,
    turn_chars: int = 500,
    llm_threads: int = 8,
    llm_gpu_layers: int = -1,
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
        vault=Path(vault or Path.home() / "KnowledgeVault").expanduser().resolve(),
        no_memory=no_memory,
        no_llm=no_llm,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        knowledge_limit=knowledge_limit,
        memory_chars=memory_chars,
        profile_chars=profile_chars,
        session_chars=session_chars,
        session_turns=session_turns,
        turn_chars=turn_chars,
        llm_threads=llm_threads,
        llm_gpu_layers=llm_gpu_layers,
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
    ) -> LLMEngine:
        ...


def build_text_runtime(
    config: TextRuntimeConfig,
    *,
    llm_factory: LLMFactory | None = None,
    session_memory: SessionMemory | None = None,
) -> TextRuntimeBundle:
    """Build text-only AssistantRuntime without ASR or TTS.

    This is the shared builder for `soca ask` and the later interactive
    `soca chat` command. It deliberately keeps tool routing explicit:
    natural-language domain routing is left to AssistantRuntime/LLM, while
    deterministic tools are only selected for clear commands such as `wiki:`.
    """
    vault = config.vault.expanduser()
    knowledge_source = None
    knowledge_builder = None
    knowledge_status = f"disabled:vault_missing:{vault}"

    if vault.is_dir():
        knowledge_source = MarkdownVaultKnowledgeSource(
            vault,
            include_globs=("wiki/**/*.md",),
        )
        knowledge_builder = KnowledgeContextBuilder(
            knowledge_source,
            max_hits=config.knowledge_limit,
        )
        knowledge_status = f"enabled:{vault / 'wiki'}"

    tools = [LocalTimeTool()]
    if knowledge_source is not None:
        tools.extend(
            [
                KnowledgeSearchTool(knowledge_source, max_limit=config.knowledge_limit),
                KnowledgeReadTool(knowledge_source),
            ]
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
                max_turns=config.session_turns,
                max_chars=config.session_chars,
                max_turn_chars=config.turn_chars,
            )
        )
        long_term = (
            MarkdownLongTermMemory(vault, max_chars=config.profile_chars)
            if vault.is_dir()
            else None
        )
        memory_builder = MemoryContextBuilder(
            long_term=long_term,
            session=runtime_session_memory,
            max_chars=config.memory_chars,
        )
        if vault.is_dir():
            memory_status = f"enabled:{vault / 'memory' / 'profile.md'}"
        else:
            memory_status = f"session-only:vault_missing:{vault}"

    if config.no_llm:
        llm = None
        llm_status = "disabled"
    else:
        factory = llm_factory or LocalLlamaCppLLM
        llm = factory(
            model_key=config.llm_model,
            n_threads=config.llm_threads,
            n_gpu_layers=config.llm_gpu_layers,
        )
        llm_status = f"enabled:{config.llm_model}"

    runtime = AssistantRuntime(
        llm=llm,
        tool_runtime=ToolRuntime(tools),
        tool_router=DefaultRuntimeToolRouter(
            knowledge_search_prefixes=("wiki:", "knowledge:"),
        ),
        knowledge_builder=knowledge_builder,
        memory_builder=memory_builder,
        options=RuntimeOptions(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            knowledge_limit=config.knowledge_limit,
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

    console.print(
        f"[green]SoCa text runtime[/green] LLM={bundle.llm_status} "
        f"Knowledge={bundle.knowledge_status} Memory={bundle.memory_status}"
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
    border = "red" if result.blocked else "magenta"
    console.print(
        Panel(
            result.response_text or "<empty>",
            title=f"Route: {result.route.value}",
            border_style=border,
        )
    )

    if result.citations:
        citations = Table(title="Citations")
        citations.add_column("Ref", style="green")
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
        console.print(Panel("<none>", title="Trace", border_style="blue"))
        return

    summary = Table(title="Trace Summary")
    summary.add_column("Field")
    summary.add_column("Value")
    summary.add_row("route", trace.route.value)
    summary.add_row("blocked", str(trace.blocked))
    summary.add_row("used_tool", str(trace.used_tool))
    summary.add_row("used_llm", str(trace.used_llm))
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
            border_style="blue",
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
