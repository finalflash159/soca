"""Exercise the consent-gated episode and proposal lifecycle."""

from __future__ import annotations

import argparse
import json
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

from eval.experimental.memory_lifecycle import EpisodeStore, MemoryEpisode
from soca.memory import MemoryCommands, MemoryProposal, ProposalStore


def evaluate(root: Path) -> dict[str, str | bool]:
    episode_store = EpisodeStore(root / "episodes")
    now = datetime.now(UTC)
    episode = MemoryEpisode(str(uuid.uuid4()), now, now, "A bounded episode summary", ("stable fact",))
    episode_store.persist(episode)
    proposals = ProposalStore(root / "proposals")
    proposal = MemoryProposal(
        id=str(uuid.uuid4()),
        kind="stable_fact",
        statement="The project uses a bounded memory lifecycle.",
        evidence_excerpt="A reviewed episode summary.",
        confidence=0.9,
        source_episode_id=episode.id,
        created_at=now,
    )
    proposals.put(proposal)
    result = MemoryCommands(root / "vault", proposals).approve(proposal.id)
    return {
        "episode_round_trip": episode_store.get(episode.id) == episode,
        "proposal_approved": result.status == "approved",
        "captured_note": result.note_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="soca-memory-eval-") as directory:
        result = evaluate(Path(directory))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
