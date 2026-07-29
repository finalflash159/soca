"""Build provenance manifests for the controlled-workflow baseline suites."""

from __future__ import annotations

import argparse
import hashlib
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
DEFAULT_CORPUS_ROOT = REPO_ROOT / "eval" / "fixtures" / "real_rag_vault"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_corpus(corpus_root: Path) -> tuple[Path, ...]:
    """Validate the corpus manifest and return every file in the corpus."""

    manifest_path = corpus_root / "SOURCE_MANIFEST.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("sources")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{manifest_path}: missing non-empty sources")
    for entry in entries:
        relative_path = entry.get("path") if isinstance(entry, dict) else None
        expected_hash = entry.get("sha256") if isinstance(entry, dict) else None
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            raise ValueError(f"{manifest_path}: invalid source entry")
        source_path = corpus_root / relative_path
        if not source_path.is_file() or _sha256(source_path) != expected_hash:
            raise ValueError(f"{manifest_path}: source hash mismatch for {relative_path}")
    files = tuple(sorted(path for path in corpus_root.rglob("*") if path.is_file()))
    return files


def build_dataset_manifest(
    paths: Sequence[Path],
    *,
    corpus_roots: Sequence[Path] | None = None,
) -> dict[str, Any]:
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

    resolved_corpus_roots = tuple(corpus_roots or (DEFAULT_CORPUS_ROOT,))
    corpus_files: list[Path] = []
    corpus_descriptions: list[dict[str, Any]] = []
    for corpus_root in resolved_corpus_roots:
        files = _validate_corpus(corpus_root)
        corpus_files.extend(files)
        corpus_descriptions.append(
            {
                "root": str(corpus_root),
                "manifest": str(corpus_root / "SOURCE_MANIFEST.json"),
                "file_count": len(files),
            }
        )

    metadata = make_eval_artifact_metadata(
        suite="remediation_baseline",
        data_files=tuple(paths) + tuple(corpus_files),
        config={
            "quality_suite": True,
            "dataset_count": len(paths),
            "corpus_count": len(resolved_corpus_roots),
        },
    )
    return {
        "schema_version": "soca-remediation-dataset-manifest-v1",
        "suite": "remediation_baseline",
        "artifact": metadata.to_dict(),
        "case_count": len(all_cases),
        "datasets": datasets,
        "corpora": corpus_descriptions,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=Path, dest="datasets")
    parser.add_argument("--corpus", action="append", type=Path, dest="corpora")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "eval" / "results" / "remediation_baseline" / "manifest.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = tuple(args.datasets or DEFAULT_DATASETS)
    manifest = build_dataset_manifest(paths, corpus_roots=tuple(args.corpora or (DEFAULT_CORPUS_ROOT,)))
    write_json_artifact(args.output, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
