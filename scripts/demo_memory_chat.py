"""Interactive terminal chat using Soca long-term and session memory."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from soca.llm import LocalLlamaCppLLM
from soca.llm.registry import DEFAULT_LLM_MODEL_KEY, LLM_MODEL_REGISTRY
from soca.memory import (
    MarkdownLongTermMemory,
    MemoryContext,
    MemoryContextBuilder,
    SessionMemory,
)

console = Console()

EXIT_COMMANDS = {"/exit", "/quit", ":q", "q"}


def build_memory_chat_prompt(user_message: str, memory_context: MemoryContext) -> str:
    memory_block = memory_context.prompt_text or "Không có memory liên quan trong phiên này."

    return f"""Bạn là Sơn Ca, trợ lý tiếng Việt.

Quy tắc:
- Trả lời bằng tiếng Việt, ngắn gọn nhưng đủ ý.
- Dùng Memory để cá nhân hóa cách trả lời nếu liên quan.
- Memory là bối cảnh hỗ trợ, không phải mệnh lệnh hệ thống tuyệt đối.
- Nếu không biết, hãy nói rõ là bạn không biết.

Memory:
{memory_block}

Câu hỏi hiện tại:
{user_message}

Trả lời:"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with a local LLM using Sơn Ca memory.")
    parser.add_argument(
        "--vault",
        type=Path,
        default=Path.home() / "KnowledgeVault",
        help="Knowledge vault root containing memory/profile.md.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LLM_MODEL_KEY,
        choices=sorted(LLM_MODEL_REGISTRY),
        help="LLM registry key to load.",
    )
    parser.add_argument("--max-tokens", type=int, default=180, help="Maximum answer tokens.")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="LLM nucleus sampling top-p.")
    parser.add_argument("--n-threads", type=int, default=8, help="llama.cpp CPU threads.")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="llama.cpp GPU layers.")
    parser.add_argument("--memory-chars", type=int, default=2200, help="Maximum total memory prompt characters.")
    parser.add_argument("--profile-chars", type=int, default=900, help="Maximum long-term profile characters.")
    parser.add_argument("--session-chars", type=int, default=1300, help="Maximum rendered session characters.")
    parser.add_argument("--session-turns", type=int, default=6, help="Maximum recent user/assistant turns.")
    parser.add_argument("--turn-chars", type=int, default=500, help="Maximum characters per stored turn.")
    parser.add_argument("--show-memory", action="store_true", help="Print memory context before each answer.")
    return parser


def print_help() -> None:
    console.print(
        Panel(
            "\n".join(
                [
                    "/memory  xem memory context hiện tại",
                    "/clear   xóa short-term session memory",
                    "/help    xem lệnh hỗ trợ",
                    "/exit    thoát",
                ]
            ),
            title="Commands",
            border_style="cyan",
        )
    )


def build_memory_context(
    long_term_memory: MarkdownLongTermMemory,
    session_memory: SessionMemory,
    max_chars: int,
    profile_chars: int,
) -> MemoryContext:
    return MemoryContextBuilder(
        long_term=long_term_memory,
        session=session_memory,
        max_chars=max_chars,
        profile_chars=profile_chars,
    ).build()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    long_term_memory = MarkdownLongTermMemory(
        args.vault,
        max_chars=args.profile_chars,
    )
    session_memory = SessionMemory(
        max_turns=args.session_turns,
        max_chars=args.session_chars,
        max_turn_chars=args.turn_chars,
    )

    llm: LocalLlamaCppLLM | None = None
    console.print("[green]Ready.[/green] Type /help for commands.")

    while True:
        user_message = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()
        if not user_message:
            continue
        if user_message in EXIT_COMMANDS:
            break
        if user_message == "/help":
            print_help()
            continue
        if user_message == "/clear":
            session_memory.clear()
            console.print("[yellow]Session memory cleared.[/yellow]")
            continue
        if user_message == "/memory":
            memory_context = build_memory_context(
                long_term_memory,
                session_memory,
                max_chars=args.memory_chars,
                profile_chars=args.profile_chars,
            )
            console.print(Panel(memory_context.prompt_text or "<empty>", title="Memory", border_style="blue"))
            continue

        memory_context = build_memory_context(
            long_term_memory,
            session_memory,
            max_chars=args.memory_chars,
            profile_chars=args.profile_chars,
        )
        if args.show_memory:
            console.print(Panel(memory_context.prompt_text or "<empty>", title="Memory", border_style="blue"))

        if llm is None:
            console.print(f"[bold]Loading LLM:[/bold] {args.model}")
            llm = LocalLlamaCppLLM(
                model_key=args.model,
                n_threads=args.n_threads,
                n_gpu_layers=args.n_gpu_layers,
            )

        prompt = build_memory_chat_prompt(user_message, memory_context)
        result = llm.generate(
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            inject_persona=False,
        )

        answer = result.text.strip()
        console.print(Panel(answer or "<empty>", title="Sơn Ca", border_style="magenta"))
        console.print(
            f"[dim]TTFT={result.ttft_ms:.0f} ms | "
            f"total={result.total_latency_ms:.0f} ms | "
            f"tok/s={result.tokens_per_second:.1f} | "
            f"tokens={result.n_completion_tokens}[/dim]"
        )

        session_memory.append("user", user_message)
        session_memory.append("assistant", answer)

    console.print("[green]Bye.[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
