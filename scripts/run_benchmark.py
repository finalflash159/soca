from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from eval.benchmark_run import run_logged_command

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "eval" / "results"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a benchmark while retaining stdout, stderr and provenance."
    )
    parser.add_argument("--family", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    command = tuple(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("benchmark command is required after --")
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / args.family / run_id
    return run_logged_command(
        command,
        run_dir=run_dir,
        family=args.family,
        cwd=REPO_ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
