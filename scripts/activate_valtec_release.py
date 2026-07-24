# scripts/activate_valtec_release.py
from __future__ import annotations

import argparse

from soca.tts.valtec.artifacts import activate_valtec_release


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-id", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    activate_valtec_release(args.artifact_id)
    print(f"Activated Valtec release: {args.artifact_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
