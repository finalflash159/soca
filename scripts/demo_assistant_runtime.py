"""Interactive text demo for the Soca AssistantRuntime."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from soca.core import AssistantRuntime, RuntimeOptions
from soca.knowledge import KnowledgeContextBuilder, MarkdownVaultKnowledgeSource
from soca.llm import LocalLlamaCppLLM
from soca.llm.registry import DEFAULT_LLM_MODEL_KEY, LLM_MODEL_REGISTRY
from soca.memory import MarkdownLongTermMemory, MemoryContextBuilder, SessionMemory
from soca.tools import (
    KnowledgeReadTool,
    KnowledgeSearchTool,
    LocalTimeTool,
    ToolRuntime,
)

console = Console()
EXIT_COMMANDS = {"/exit", "/quit", ":q", "q"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the text-only Soca AssistantRuntime.")
    parser.add_argument(
        "--vault",
        type=Path,
        default=Path.home() / "KnowledgeVault",
        help="Knowledge vault root.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LLM_MODEL_KEY,
        choices=sorted(LLM_MODEL_REGISTRY),
        help="LLM registry key to load.",
    )
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--no-llm", action="store_true", help="Run tool-only runtime without loading LLM.")
    parser.add_argument("--no-memory", action="store_true", help="Disable profile/session memory.")
    parser.add_argument("--show-trace", action="store_true", help="Print runtime trace after every turn.")
    return parser


def print_help() -> None:
    console.print(
        Panel(
            "\n".join(
                [
                    "mấy giờ rồi                  -> local_time.now",
                    "wiki: chất đạm               -> knowledge.search",
                    "đọc wiki/path/note.md        -> knowledge.read",
                    "/k câu hỏi                   -> ask LLM with knowledge context",
                    "/trace                       -> toggle trace display",
                    "/clear                       -> clear session memory",
                    "/exit                        -> quit",
                ]
            ),
            title="AssistantRuntime Demo Commands",
            border_style="cyan",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    source = MarkdownVaultKnowledgeSource(args.vault, include_globs=("wiki/**/*.md",))
    knowledge_builder = KnowledgeContextBuilder(source)
    tool_runtime = ToolRuntime(
        [
            LocalTimeTool(),
            KnowledgeSearchTool(source),
            KnowledgeReadTool(source),
        ]
    )

    session_memory = SessionMemory()
    memory_builder = None
    if not args.no_memory:
        memory_builder = MemoryContextBuilder(
            long_term=MarkdownLongTermMemory(args.vault),
            session=session_memory,
        )

    llm = None
    if not args.no_llm:
        console.print(f"[bold]Loading LLM:[/bold] {args.model}")
        llm = LocalLlamaCppLLM(
            model_key=args.model,
            n_threads=args.n_threads,
            n_gpu_layers=args.n_gpu_layers,
        )

    runtime = AssistantRuntime(
        llm=llm,
        tool_runtime=tool_runtime,
        knowledge_builder=knowledge_builder,
        memory_builder=memory_builder,
        options=RuntimeOptions(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        ),
    )

    show_trace = args.show_trace
    console.print("[green]AssistantRuntime ready.[/green] Type /help for commands.")

    while True:
        user_text = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()
        if not user_text:
            continue
        if user_text in EXIT_COMMANDS:
            break
        if user_text == "/help":
            print_help()
            continue
        if user_text == "/trace":
            show_trace = not show_trace
            console.print(f"[yellow]Trace:[/yellow] {'on' if show_trace else 'off'}")
            continue
        if user_text == "/clear":
            session_memory.clear()
            console.print("[yellow]Session memory cleared.[/yellow]")
            continue

        metadata = {}
        if user_text.startswith("/k "):
            user_text = user_text[3:].strip()
            metadata["use_knowledge"] = True

        result = runtime.run_text_turn(user_text, metadata=metadata)
        border = "red" if result.blocked else "magenta"
        console.print(Panel(result.response_text or "<empty>", title=result.route.value, border_style=border))

        if result.citations:
            citations = "\n".join(f"- {item.path} :: {item.title}" for item in result.citations)
            console.print(Panel(citations, title="Citations", border_style="green"))

        if show_trace and result.trace is not None:
            guardrails = ", ".join(
                f"{event.stage.value}:{event.action.value}:{event.reason or '-'}"
                for event in result.trace.guardrail_events
            )
            tools = ", ".join(call.name for call in result.trace.tool_calls) or "<none>"
            console.print(
                Panel(
                    "\n".join(
                        [
                            f"blocked={result.trace.blocked}",
                            f"used_tool={result.trace.used_tool}",
                            f"used_llm={result.trace.used_llm}",
                            f"tools={tools}",
                            f"guardrails={guardrails}",
                        ]
                    ),
                    title="Trace",
                    border_style="blue",
                )
            )

    console.print("[green]Bye.[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
