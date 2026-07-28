"""Explicit, verified provisioning for local-only summary benchmark candidates.

The runtime never imports this module and never downloads weights. This script
stores artifacts below the repository's ignored ``models/summary`` directory,
one immutable Hugging Face revision at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
from collections.abc import Iterable, Sequence
from pathlib import Path

# This must be set before importing huggingface_hub, which reads its constants
# at import time. Callers may still set it to "0" explicitly to opt into Xet.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import hf_hub_download
from rich.console import Console

from soca.memory.summary import (
    SUMMARY_MODEL_REGISTRY,
    SummaryModelSpec,
    default_summary_model_root,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
_DISK_HEADROOM_BYTES = 512 * 1024 * 1024
console = Console()


def default_model_root() -> Path:
    return default_summary_model_root()


def load_hf_token_from_dotenv(path: Path = REPO_ROOT / ".env") -> bool:
    """Load only the HF token needed by this explicit provisioning command."""
    if os.environ.get("HF_TOKEN") or not path.is_file():
        return False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        key, separator, value = line.partition("=")
        if separator and key.strip() in {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"}:
            os.environ["HF_TOKEN"] = value.strip().strip("\"'")
            return True
    return False


def select_specs(model_keys: Iterable[str], *, all_models: bool) -> list[SummaryModelSpec]:
    keys = list(model_keys)
    if all_models:
        keys = list(SUMMARY_MODEL_REGISTRY)
    if not keys:
        raise ValueError("select at least one --model or pass --all")
    return [SUMMARY_MODEL_REGISTRY[key] for key in keys]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_download(path: Path, *, expected_bytes: int, expected_sha256: str) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == expected_bytes
        and _sha256(path) == expected_sha256
    )


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(stat.S_IRWXU)


def _ensure_disk_headroom(destination: Path, expected_bytes: int) -> None:
    free = shutil.disk_usage(destination).free
    required = expected_bytes + _DISK_HEADROOM_BYTES
    if free < required:
        raise RuntimeError(f"insufficient free disk: need {required} bytes, have {free} bytes")


def provision(spec: SummaryModelSpec, *, root: Path) -> Path:
    destination = spec.path(root)
    _ensure_private_directory(destination.parent)
    if verify_download(destination, expected_bytes=spec.expected_bytes, expected_sha256=spec.expected_sha256):
        destination.chmod(stat.S_IRUSR | stat.S_IWUSR)
        console.print(f"[green]Verified existing[/green] {spec.key}: {destination}")
        return destination
    _ensure_disk_headroom(destination.parent, spec.expected_bytes)
    console.print(
        f"[bold]Downloading[/bold] {spec.key}\n"
        f"  repo: {spec.hf_repo}@{spec.revision}\n"
        f"  file: {spec.filename}\n"
        f"  destination: {destination}"
    )
    downloaded = Path(
        hf_hub_download(
            repo_id=spec.hf_repo,
            filename=spec.filename,
            revision=spec.revision,
            local_dir=str(destination.parent),
        )
    )
    if downloaded != destination:
        raise RuntimeError(f"unexpected download path: {downloaded}")
    if not verify_download(destination, expected_bytes=spec.expected_bytes, expected_sha256=spec.expected_sha256):
        raise RuntimeError(f"integrity verification failed for {spec.key}")
    destination.chmod(stat.S_IRUSR | stat.S_IWUSR)
    console.print(f"[green]Verified[/green] {spec.key}: {destination}")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Provision verified summary GGUFs; runtime never downloads them.")
    parser.add_argument("--model", action="append", default=[], choices=sorted(SUMMARY_MODEL_REGISTRY))
    parser.add_argument("--all", action="store_true", help="Provision every benchmark candidate sequentially.")
    parser.add_argument("--list", action="store_true", help="Print pinned candidate metadata without downloading.")
    parser.add_argument("--model-root", type=Path, default=default_model_root())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_hf_token_from_dotenv()
    specs = select_specs(args.model, all_models=args.all)
    if args.list:
        for spec in specs:
            console.print(f"{spec.key}\t{spec.expected_bytes}\t{spec.hf_repo}@{spec.revision}\t{spec.filename}")
        return 0
    root = args.model_root.resolve()
    _ensure_private_directory(root)
    failures: list[tuple[str, str]] = []
    for spec in specs:
        try:
            provision(spec, root=root)
        except Exception as exc:  # noqa: BLE001 - continue independent candidate provisioning
            failures.append((spec.key, type(exc).__name__))
            console.print(f"[red]Failed[/red] {spec.key}: {type(exc).__name__}")
    if failures:
        console.print("[yellow]Provisioning finished with failures:[/yellow]")
        for key, error in failures:
            console.print(f"  {key}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
