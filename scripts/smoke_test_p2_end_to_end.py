"""Run a provider-free P2 smoke flow and print a machine-readable result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from soca.core.tool_routing import parse_tool_decision
from soca.memory import CompactionConfig, WorkingMemory


def run() -> dict[str, object]:
    decision = parse_tool_decision('{"tool":"none","arguments":{}}', max_chars=256)
    memory = WorkingMemory(config=CompactionConfig(recent_turns=2, summary_chars=128))
    for text in ("first decision", "second context", "third decision"):
        memory.append("user", text)
    return {
        "router": decision.tool,
        "compacted_turn_count": memory.snapshot.compacted_turn_count,
        "recent_turn_count": len(memory.snapshot.recent_turns),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
