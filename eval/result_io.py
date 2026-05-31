from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalRunPaths:
    run_dir: Path
    json_path: Path
    md_path: Path
    latest_json_path: Path
    latest_md_path: Path


def make_eval_run_paths(output_dir: Path, family: str, run_id: str) -> EvalRunPaths:
    family_dir = output_dir / family
    run_dir = family_dir / run_id
    return EvalRunPaths(
        run_dir=run_dir,
        json_path=run_dir / "report.json",
        md_path=run_dir / "report.md",
        latest_json_path=family_dir / "latest.json",
        latest_md_path=family_dir / "latest.md",
    )


def update_latest_eval_report(paths: EvalRunPaths) -> None:
    paths.latest_json_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(paths.json_path, paths.latest_json_path)
    shutil.copyfile(paths.md_path, paths.latest_md_path)
