"""Build provenance manifests for the controlled-workflow baseline suites."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eval.baseline_cases import assert_quality_eligible, load_cases
from eval.result_io import make_eval_artifact_metadata, write_json_artifact

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = (
    REPO_ROOT / "eval" / "prompts" / "remediation_workflow_vi.jsonl",
    REPO_ROOT / "eval" / "prompts" / "remediation_rag_vi.jsonl",
)


def build_dataset_manifest(paths: Sequence[Path]) -> dict[str, Any]:
    """Validate and describe all quality datasets used by remediation eval."""

    if not paths:
        raise ValueError("at least one dataset is required")
    all_cases = []
    datasets: list[dict[str, Any]] = []
    for path in paths:
        cases = load_cases(path, quality_suite=True)
        assert_quality_eligible(cases)
        datasets.append(
            {
                "path": str(path),
                "case_count": len(cases),
                "categories": dict(sorted(Counter(case.category for case in cases).items())),
                "splits": dict(sorted(Counter(case.split for case in cases).items())),
                "dataset_classes": sorted({case.dataset_class for case in cases}),
            }
        )
        all_cases.extend(cases)

    case_ids = [case.case_id for case in all_cases]
    duplicates = sorted(case_id for case_id, count in Counter(case_ids).items() if count > 1)
    if duplicates:
        raise ValueError("duplicate case ids across datasets: " + ", ".join(duplicates))

    metadata = make_eval_artifact_metadata(
        suite="remediation_baseline",
        data_files=tuple(paths),
        config={"quality_suite": True, "dataset_count": len(paths)},
    )
    return {
        "schema_version": "soca-remediation-dataset-manifest-v1",
        "suite": "remediation_baseline",
        "artifact": metadata.to_dict(),
        "case_count": len(all_cases),
        "datasets": datasets,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=Path, dest="datasets")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "eval" / "results" / "remediation_baseline" / "manifest.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = tuple(args.datasets or DEFAULT_DATASETS)
    manifest = build_dataset_manifest(paths)
    write_json_artifact(args.output, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
