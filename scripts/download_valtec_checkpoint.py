"""Download a pinned Valtec checkpoint/config as immutable build inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "valtecAI-team/valtec-tts-pretrained"
PINNED_REVISION = "d58e99132232e58c9e156a334fece8f546aa7d40"
REQUIRED_FILES = ("G.pth", "config.json")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = (
    REPO_ROOT
    / "models/tts/valtec_multispeaker/source/upstream"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(destination: Path, *, revision: str) -> dict:
    revision = revision.lower()
    manifest_path = destination / "source.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Valtec source manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported Valtec source manifest schema")
    if payload.get("repo_id") != REPO_ID or payload.get("revision") != revision:
        raise ValueError("Valtec source repo/revision does not match the requested pin")
    files = payload.get("files")
    if not isinstance(files, dict) or set(files) != set(REQUIRED_FILES):
        raise ValueError("Valtec source manifest has an invalid file allow-list")
    for name in REQUIRED_FILES:
        path = destination / name
        if not path.is_file() or sha256_file(path) != files[name]:
            raise ValueError(f"Valtec source checksum mismatch: {name}")
    return payload


def download_checkpoint(destination: Path, *, revision: str) -> Path:
    revision = revision.lower()
    destination = destination.expanduser().resolve()
    if destination.exists():
        verify_source(destination, revision=revision)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )
    try:
        snapshot_download(
            repo_id=REPO_ID,
            revision=revision,
            allow_patterns=list(REQUIRED_FILES),
            local_dir=staging,
        )
        missing = [name for name in REQUIRED_FILES if not (staging / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Valtec pretrained repo is missing: {missing}")
        shutil.rmtree(staging / ".cache", ignore_errors=True)
        payload = {
            "schema_version": 1,
            "repo_id": REPO_ID,
            "revision": revision,
            "files": {
                name: sha256_file(staging / name)
                for name in REQUIRED_FILES
            },
        }
        (staging / "source.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    verify_source(destination, revision=revision)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--revision", default=PINNED_REVISION)
    args = parser.parse_args()
    if len(args.revision) != 40 or any(
        char not in "0123456789abcdef" for char in args.revision.lower()
    ):
        parser.error("--revision must be a full 40-character Git commit")
    print(download_checkpoint(args.destination, revision=args.revision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
