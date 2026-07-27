"""Measure deterministic working-memory compaction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from soca.memory import CompactionConfig, WorkingMemory


def evaluate(turns: int, recent_turns: int, summary_chars: int) -> dict[str, int | float]:
    memory = WorkingMemory(config=CompactionConfig(recent_turns=recent_turns, summary_chars=summary_chars))
    for index in range(turns):
        memory.append("user", f"turn {index} decision and context")
        memory.append("assistant", f"answer {index}")
    snapshot = memory.snapshot
    return {
        "turns": turns,
        "recent_turn_count": len(snapshot.recent_turns),
        "compacted_turn_count": snapshot.compacted_turn_count,
        "summary_chars": len(snapshot.summary),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--recent-turns", type=int, default=8)
    parser.add_argument("--summary-chars", type=int, default=1_600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evaluate(args.turns, args.recent_turns, args.summary_chars), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
