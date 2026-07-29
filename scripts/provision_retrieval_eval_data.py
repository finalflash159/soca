from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from eval.retrieval_sources import RetrievalSource, load_source_lock, write_provision_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "benchmarks" / "retrieval"
DEFAULT_LOCK = DEFAULT_DATA_ROOT / "sources.lock.json"


def _git_source(source: RetrievalSource, destination: Path) -> None:
    if destination.exists():
        head = subprocess.run(
            ["git", "-C", str(destination), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head != source.revision:
            raise RuntimeError(
                f"{source.name}: existing checkout is {head}, expected {source.revision}"
            )
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            source.source,
            str(destination),
        ],
        check=True,
    )
    if source.sparse_paths and source.sparse_paths != (".",):
        subprocess.run(
            ["git", "-C", str(destination), "sparse-checkout", "init", "--no-cone"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(destination),
                "sparse-checkout",
                "set",
                *source.sparse_paths,
            ],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", source.revision],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != source.revision:
        raise RuntimeError(
            f"{source.name}: checkout resolved to {head}, expected {source.revision}"
        )


def _huggingface_source(source: RetrievalSource, destination: Path) -> None:
    from huggingface_hub import hf_hub_download

    for filename in source.files:
        hf_hub_download(
            repo_id=source.source,
            repo_type="dataset",
            filename=filename,
            revision=source.revision,
            local_dir=destination,
        )


def provision(source: RetrievalSource, *, data_root: Path) -> None:
    destination = data_root / source.destination
    print(f"[{source.name}] {source.kind} -> {destination}", flush=True)
    if source.kind == "git":
        _git_source(source, destination)
    elif source.kind == "huggingface":
        destination.mkdir(parents=True, exist_ok=True)
        _huggingface_source(source, destination)
    else:
        raise ValueError(f"{source.name}: unsupported source kind {source.kind!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Provision pinned retrieval benchmark data.")
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    lock = load_source_lock(args.lock)
    selected = set(args.source)
    if not args.all and not selected:
        raise SystemExit("pass --all or at least one --source")
    unknown = selected - {source.name for source in lock.sources}
    if unknown:
        raise SystemExit("unknown sources: " + ", ".join(sorted(unknown)))
    for source in lock.sources:
        if args.all or source.name in selected:
            provision(source, data_root=args.data_root)
    output = args.data_root / "provisioned-manifest.json"
    write_provision_manifest(
        lock,
        data_root=args.data_root,
        output=output,
        selected=None if args.all else selected,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
