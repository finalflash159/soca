"""Smoke test local Markdown knowledge retrieval and prompt context formatting."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.table import Table

from soca.knowledge import KnowledgeContextBuilder, MarkdownVaultKnowledgeSource

console = Console()

DEFAULT_QUERIES = (
    "bữa sáng nhanh lành mạnh",
    "bua sang nhanh lanh manh",
    "giảm muối đường chất béo",
    "ăn gì để đủ đạm",
    "táo bón nên ăn chất xơ",
)

DEFAULT_EXPECTED_TOP = {
    "bữa sáng nhanh lành mạnh": "wiki/dinh-duong/goi-y-bua-an.md",
    "bua sang nhanh lanh manh": "wiki/dinh-duong/goi-y-bua-an.md",
    "giảm muối đường chất béo": "wiki/dinh-duong/duong-muoi-chat-beo.md",
    "ăn gì để đủ đạm": "wiki/dinh-duong/chat-dam.md",
    "táo bón nên ăn chất xơ": "wiki/dinh-duong/chat-xo-rau-cu-trai-cay.md",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a quick local knowledge smoke test.")
    parser.add_argument(
        "--vault",
        type=Path,
        default=Path.home() / "KnowledgeVault",
        help="Knowledge vault root.",
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Query to test. Can be passed multiple times.",
    )
    parser.add_argument("--max-hits", type=int, default=3, help="Maximum hits per query.")
    parser.add_argument("--max-chars", type=int, default=1800, help="Maximum prompt-context characters.")
    parser.add_argument(
        "--check-defaults",
        action="store_true",
        help="Fail if default nutrition queries do not retrieve the expected top page.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    queries = tuple(args.queries) if args.queries else DEFAULT_QUERIES

    source = MarkdownVaultKnowledgeSource(
        args.vault,
        include_globs=("wiki/**/*.md",),
    )
    builder = KnowledgeContextBuilder(source, max_hits=args.max_hits, max_chars=args.max_chars)

    table = Table(title=f"Knowledge Smoke Test — {source.root}", show_lines=True)
    table.add_column("Query", style="cyan", width=28)
    table.add_column("Status", justify="center", width=8)
    table.add_column("Citations", style="green", width=48, overflow="fold")
    table.add_column("Prompt preview", style="white", width=72, overflow="fold")

    failed = False
    for query in queries:
        context = builder.build(query)
        citations = "\n".join(f"{item.path} :: {item.title}" for item in context.citations)
        expected_top = DEFAULT_EXPECTED_TOP.get(query) if args.check_defaults else None
        actual_top = context.citations[0].path if context.citations else ""
        status = "ok"
        if expected_top is not None and actual_top != expected_top:
            status = "fail"
            failed = True
        table.add_row(query, status, citations or "<none>", context.prompt_text)

    console.print(table)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
