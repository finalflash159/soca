from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.table import Table

from soca.knowledge.vault import ScaffoldResult, init_knowledge_vault

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialize a local knowledge vault.")
    parser.add_argument(
        "root",
        type=Path,
        help="Directory where the knowledge vault should be created.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing scaffold files.",
    )
    return parser


def print_result(result: ScaffoldResult) -> None:
    table = Table(title="Knowledge Vault")
    table.add_column("Item")
    table.add_column("Count", justify="right")

    table.add_row("Created directories", str(len(result.created_dirs)))
    table.add_row("Created files", str(len(result.created_files)))
    table.add_row("Skipped files", str(len(result.skipped_files)))
    console.print(table)

    console.print(f"[green]Vault ready:[/green] {result.root}")
    console.print("\nNext steps:")
    console.print(f"  1. Open this folder in your Markdown editor: {result.root}")
    console.print("  2. Put raw inputs under raw/sources/")
    console.print("  3. Write compiled notes under wiki/")
    console.print("  4. Point MarkdownVaultKnowledgeSource at this root with wiki/**/*.md")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = init_knowledge_vault(args.root, force=args.force)
    print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
