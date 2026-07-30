"""Interactively review a research-only BoH artifact."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()

DECISION_KEEP = "keep"
DECISION_REJECT = "reject"
DECISION_SKIP = "skip"
DECISION_QUIT = "quit"

PROMPT_HELP = (
    "[bold]y[/bold]=keep  [bold]n[/bold]=reject (flip to keep=false)  "
    "[bold]s[/bold]=skip (no change)  [bold]b[/bold]=back  [bold]q[/bold]=save & quit"
)


def render_phrase(item: dict, index: int, total: int) -> None:
    phrase = item["phrase"]
    count = item["count"]
    length = item["length"]
    keep = item.get("keep", True)
    reviewed = item.get("reviewed", False)
    examples = item.get("examples", [phrase])

    body_lines = [
        f"[bold]Count:[/bold]    {count}",
        f"[bold]Length:[/bold]   {length} chars",
        "[bold]Status:[/bold]   "
        + ("[green]keep=True[/green]" if keep else "[red]keep=False[/red]")
        + ("  [dim](reviewed)[/dim]" if reviewed else "  [dim](unreviewed)[/dim]"),
        "",
        "[bold]Phrase:[/bold]",
        f"  [cyan]{phrase}[/cyan]",
    ]
    if examples:
        body_lines.append("")
        body_lines.append(f"[bold]Examples (first {min(3, len(examples))}):[/bold]")
        for i, ex in enumerate(examples[:3], 1):
            ex_short = ex if len(ex) <= 200 else ex[:197] + "..."
            body_lines.append(f"  {i}. [dim]'{ex_short}'[/dim]")

    console.print(Rule(f"Phrase {index + 1}/{total}"))
    console.print("\n".join(body_lines))
    console.print()


def prompt_decision() -> str:
    while True:
        console.print(PROMPT_HELP)
        choice = click.prompt(">", default="", show_default=False).strip().lower()
        if choice in ("y", "yes", "k", "keep"):
            return DECISION_KEEP
        if choice in ("n", "no", "r", "reject"):
            return DECISION_REJECT
        if choice in ("s", "skip", ""):
            return DECISION_SKIP
        if choice in ("b", "back"):
            return "back"
        if choice in ("q", "quit", "save"):
            return DECISION_QUIT
        console.print("[red]Invalid input. Use y/n/s/b/q[/red]")


def save_data(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@click.command()
@click.option(
    "--boh-path",
    required=True,
    type=click.Path(),
    help="BoH JSON to review.",
)
@click.option(
    "--unreviewed-only", is_flag=True,
    help="Skip phrases that already have reviewed=true in metadata.",
)
@click.option(
    "--start", default=0, type=int, help="Start at this phrase index (0-based).",
)
@click.option(
    "--reset", is_flag=True,
    help="Flip all keep=True and mark unreviewed before starting (fresh review).",
)
def main(boh_path: str, unreviewed_only: bool, start: int, reset: bool) -> None:
    path = Path(boh_path)
    if not path.exists():
        raise click.ClickException(f"BoH file not found: {path}")

    # Backup once before any modification
    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    console.print(f"[dim]Backup → {backup}[/dim]\n")

    data = json.loads(path.read_text(encoding="utf-8"))
    boh = data.get("boh", [])
    if not boh:
        raise click.ClickException("BoH list is empty in this file.")

    if reset:
        for item in boh:
            item["keep"] = True
            item.pop("reviewed", None)
        save_data(data, path)
        console.print("[yellow]All phrases reset to keep=True, marked unreviewed.[/yellow]\n")

    # Build review index list
    indices = list(range(start, len(boh)))
    if unreviewed_only:
        indices = [i for i in indices if not boh[i].get("reviewed", False)]

    if not indices:
        console.print("[yellow]No phrases to review (try --reset or --start 0).[/yellow]")
        return

    console.print(
        Panel(
            f"BoH: [bold]{path}[/bold]\n"
            f"Total phrases: [bold]{len(boh)}[/bold]\n"
            f"To review:     [bold]{len(indices)}[/bold]"
            + (" (unreviewed only)" if unreviewed_only else "")
            + (f", starting at #{start + 1}" if start else "")
            + "\n\n"
            + PROMPT_HELP,
            title="Interactive BoH Manual Review",
            border_style="cyan",
        )
    )
    console.print()

    counters = {"kept": 0, "rejected": 0, "skipped": 0}
    cursor = 0
    quit_requested = False

    while cursor < len(indices):
        i = indices[cursor]
        item = boh[i]
        render_phrase(item, cursor, len(indices))
        decision = prompt_decision()

        if decision == DECISION_QUIT:
            quit_requested = True
            break
        if decision == "back":
            cursor = max(0, cursor - 1)
            console.print("[yellow]← Back[/yellow]\n")
            continue

        if decision == DECISION_KEEP:
            item["keep"] = True
            item["reviewed"] = True
            counters["kept"] += 1
            console.print("[green]✓ kept[/green]\n")
        elif decision == DECISION_REJECT:
            item["keep"] = False
            item["reviewed"] = True
            counters["rejected"] += 1
            console.print("[red]✗ rejected (keep=False)[/red]\n")
        elif decision == DECISION_SKIP:
            counters["skipped"] += 1
            console.print("[dim]→ skipped[/dim]\n")

        # Auto-save after every decision (Ctrl+C-safe)
        save_data(data, path)
        cursor += 1

    # Log the review session in metadata
    meta = data.setdefault("metadata", {})
    review_log = meta.setdefault("manual_review", [])
    review_log.append(
        {
            "applied_at_utc": datetime.now(UTC).isoformat(),
            "applied_by": "eval.experimental.asr_boh.review (interactive)",
            "n_decisions": counters["kept"] + counters["rejected"],
            "n_kept": counters["kept"],
            "n_rejected": counters["rejected"],
            "n_skipped": counters["skipped"],
            "quit_early": quit_requested,
        }
    )
    save_data(data, path)

    n_kept_total = sum(1 for item in boh if item.get("keep", True))
    n_rejected_total = sum(1 for item in boh if not item.get("keep", True))
    console.print(
        Panel(
            f"This session: kept={counters['kept']}, rejected={counters['rejected']}, "
            f"skipped={counters['skipped']}"
            + ("  [yellow](quit early)[/yellow]" if quit_requested else "")
            + f"\n\nFinal artifact totals: "
            f"[green]{n_kept_total} kept[/green] / "
            f"[red]{n_rejected_total} rejected[/red] "
            f"out of {len(boh)} phrases",
            title="Review complete",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
