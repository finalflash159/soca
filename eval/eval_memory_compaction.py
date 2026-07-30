"""Measure the canonical working-memory compaction contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from soca.core.text_budget import truncate
from soca.memory.working import WorkingMemory, WorkingMemoryPolicy, WorkingSummaryArtifact


def evaluate(turns: int, recent_turns: int, summary_chars: int) -> dict[str, int | float]:
    policy = WorkingMemoryPolicy(
        hard_limit_tokens=16_384,
        high_watermark_tokens=15_000,
        target_tokens=12_000,
        summary_budget_tokens=2_048,
        recent_budget_tokens=512,
        minimum_recent_complete_turns=2,
        preferred_recent_complete_turns=recent_turns,
        manual_compaction_minimum_complete_turns=5,
        mode="background_summary",
    )
    memory = WorkingMemory(policy=policy)
    for index in range(turns):
        turn = memory.begin_turn(f"turn {index} decision and context")
        memory.finish_turn(turn.sequence, f"answer {index}")
    job = memory.prepare_compaction(force=True)
    if job is None:
        raise ValueError("canonical compaction requires at least five complete turns")
    summary = truncate(
        "\n".join(
            f"User: {turn.user_text}\nAssistant: {turn.assistant_text}"
            for turn in job.frozen_turns
        ),
        summary_chars,
    )
    memory.publish_summary(
        job,
        WorkingSummaryArtifact(
            version=1,
            generation=job.generation,
            source_through_sequence=job.frozen_turns[-1].sequence,
            summary=summary,
        ),
    )
    snapshot = memory.snapshot
    return {
        "turns": turns,
        "recent_turn_count": len(snapshot.turns),
        "compacted_turn_count": len(job.frozen_turns),
        "summary_chars": len(snapshot.summary.summary) if snapshot.summary else 0,
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
