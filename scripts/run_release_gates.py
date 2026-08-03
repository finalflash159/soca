from __future__ import annotations

import argparse
from pathlib import Path

from eval.release_runner import run_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run explicit SoCa release-gate commands.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report, report_path = run_manifest(
        args.manifest.expanduser().resolve(),
        repo_root=args.repo_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    print(report_path)
    return 0 if report["decision"]["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

