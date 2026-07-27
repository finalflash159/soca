"""Run the local consent-gated memory lifecycle smoke test."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from eval.eval_memory_lifecycle import evaluate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="soca-memory-smoke-") as directory:
        result = evaluate(Path(directory))
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0 if result["episode_round_trip"] and result["proposal_approved"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
