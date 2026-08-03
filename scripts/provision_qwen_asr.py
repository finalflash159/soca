from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import soundfile as sf

from soca.asr.qwen_artifacts import (
    QWEN_REFERENCE_ARTIFACT,
    QWEN_RELEASE_ARTIFACT,
    QwenArtifactError,
    QwenASRArtifactSpec,
    default_asr_model_root,
)
from soca.asr.qwen_service_client import QwenASRServiceClient
from soca.asr.qwen_service_identity import QwenServiceLaunch
from soca.asr.qwen_store import (
    ArtifactPreflight,
    ArtifactSourceKind,
    QwenArtifactStore,
    QwenSnapshotResolver,
    QwenStoreError,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PROJECT = REPO_ROOT / "runtime" / "qwen-asr"
RUNTIME_LOCK = RUNTIME_PROJECT / "uv.lock"
RUNTIME_RECEIPT = RUNTIME_PROJECT / ".runtime-receipt.json"
RUNTIME_PYTHON = RUNTIME_PROJECT / ".venv" / "bin" / "python"
EXPECTED_PYTHON = "3.11.14"
EXPECTED_UV = "0.11.16"
TOKEN_ENVIRONMENT = "HF_TOKEN"


class QwenProvisionCommandError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(name: str) -> QwenASRArtifactSpec:
    if name == "release":
        return QWEN_RELEASE_ARTIFACT
    if name == "reference":
        return QWEN_REFERENCE_ARTIFACT
    raise QwenProvisionCommandError(f"unsupported artifact selector: {name}")


def verify_worker_runtime() -> Mapping[str, object]:
    if shutil.which("uvx") is None:
        raise QwenProvisionCommandError("uvx is required for the locked Qwen runtime")
    try:
        metadata = RUNTIME_RECEIPT.lstat()
        payload = json.loads(RUNTIME_RECEIPT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QwenProvisionCommandError(
            "Qwen worker receipt is missing; run scripts/provision_qwen_runtime.py"
        ) from exc
    expected_fields = {
        "schema_version",
        "python",
        "uv",
        "lock_sha256",
        "soca_wheel_sha256",
        "environment",
    }
    if (
        RUNTIME_RECEIPT.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or not isinstance(payload, dict)
        or set(payload) != expected_fields
    ):
        raise QwenProvisionCommandError("Qwen worker receipt is invalid or not private")
    lock_digest = _sha256(RUNTIME_LOCK)
    if (
        payload["schema_version"] != 1
        or payload["python"] != EXPECTED_PYTHON
        or payload["uv"] != EXPECTED_UV
        or payload["lock_sha256"] != lock_digest
        or Path(str(payload["environment"])) != RUNTIME_PROJECT / ".venv"
    ):
        raise QwenProvisionCommandError("Qwen worker receipt does not match the locked runtime")
    if not RUNTIME_PYTHON.is_file():
        raise QwenProvisionCommandError("Qwen worker Python executable is missing")
    try:
        completed = subprocess.run(
            [str(RUNTIME_PYTHON), "-c", "import platform; print(platform.python_version())"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise QwenProvisionCommandError("Qwen worker Python verification failed") from exc
    if completed.returncode != 0 or completed.stdout.strip() != EXPECTED_PYTHON:
        raise QwenProvisionCommandError("Qwen worker Python version is not the pinned version")
    return {
        "python": EXPECTED_PYTHON,
        "uv": EXPECTED_UV,
        "lock_sha256": lock_digest,
        "environment": str(RUNTIME_PROJECT / ".venv"),
    }


def build_health_probe(
    audio_path: Path,
    spec: QwenASRArtifactSpec,
    *,
    client_factory: Callable[..., QwenASRServiceClient] = QwenASRServiceClient,
) -> Callable[[Path], Mapping[str, object]]:
    resolved_audio = audio_path.expanduser().resolve()
    try:
        audio, sample_rate = sf.read(resolved_audio, dtype="float32", always_2d=False)
    except (OSError, RuntimeError) as exc:
        raise QwenProvisionCommandError("health audio is unreadable") from exc
    samples = np.asarray(audio, dtype=np.float32)
    if sample_rate != 16_000 or samples.ndim != 1 or samples.size == 0:
        raise QwenProvisionCommandError("health audio must be non-empty mono PCM at 16 kHz")
    if not np.isfinite(samples).all():
        raise QwenProvisionCommandError("health audio contains non-finite samples")

    def probe(model_path: Path) -> Mapping[str, object]:
        started = time.monotonic()
        client = client_factory(
            launch=QwenServiceLaunch.for_provisioning(spec, model_path),
            python_executable=RUNTIME_PYTHON,
            startup_timeout_s=180.0,
            request_timeout_s=90.0,
            shutdown_timeout_s=10.0,
            process_environment={
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
        )
        try:
            result = client.transcribe(samples, max_new_tokens=128)
        finally:
            client.close()
        if not result.avg_logprob_reliable:
            raise QwenProvisionCommandError("health transcription has unreliable logprob")
        return {
            "transcript": result.text,
            "audio_sha256": _sha256(resolved_audio),
            "audio_samples": int(samples.size),
            "latency_ms": result.latency_ms,
            "rtf": result.rtf,
            "avg_logprob": result.avg_logprob,
            "elapsed_ms": (time.monotonic() - started) * 1_000,
        }

    return probe


def _progress(path: str, copied: int, total: int) -> None:
    percent = 100.0 if total == 0 else copied * 100.0 / total
    print(f"\r{path}: {percent:6.2f}%", end="", file=sys.stderr, flush=True)
    if copied >= total:
        print(file=sys.stderr)


def _preflight_payload(report: ArtifactPreflight) -> dict[str, object]:
    return {
        "artifact_key": report.artifact_key,
        "platform": f"{report.platform}/{report.architecture}",
        "final_bytes": report.final_bytes,
        "staging_bytes": report.staging_bytes,
        "reusable_bytes": report.reusable_bytes,
        "required_free_bytes": report.required_free_bytes,
        "free_bytes": report.free_bytes,
        "runtime_lock_digest": report.runtime_lock_digest,
    }


def execute(args: argparse.Namespace) -> Mapping[str, object]:
    store = QwenArtifactStore(args.store_root.expanduser().resolve())
    specs = (QWEN_RELEASE_ARTIFACT, QWEN_REFERENCE_ARTIFACT)
    if args.command == "inspect":
        inspections = store.inspect(_artifact(args.artifact)) if args.artifact else None
        selected = (
            (inspections,)
            if inspections is not None
            else tuple(store.inspect(spec) for spec in specs)
        )
        return {
            "artifacts": [
                {
                    "artifact_key": item.artifact_key,
                    "state": item.state.value,
                    "model_path": str(item.model_path),
                    "detail": item.detail,
                }
                for item in selected
            ]
        }
    if args.command == "gc":
        removed = store.gc(specs, dry_run=args.dry_run, generation=args.generation)
        return {"dry_run": args.dry_run, "generations": [str(path) for path in removed]}

    spec = _artifact(args.artifact)
    runtime = verify_worker_runtime()
    if args.command == "verify":
        if args.deep and args.health_audio is None:
            raise QwenProvisionCommandError(
                "deep verification requires --health-audio with real 16 kHz mono speech"
            )
        health = build_health_probe(args.health_audio, spec) if args.deep else None
        receipt = store.verify(spec, deep=args.deep, health_probe=health)
        return {
            "artifact_key": spec.key,
            "state": "provisioned",
            "deep": args.deep,
            "model_path": receipt.model_path,
            "runtime": runtime,
        }
    if args.command == "refresh":
        receipt = store.refresh_receipt(
            spec,
            source_kind=ArtifactSourceKind(args.source),
            health_probe=build_health_probe(args.health_audio, spec),
            runtime_lock=RUNTIME_LOCK,
        )
        return {
            "artifact_key": spec.key,
            "state": "provisioned",
            "receipt_refreshed": True,
            "model_path": receipt.model_path,
            "runtime": runtime,
        }
    if args.health_audio is None:
        raise QwenProvisionCommandError("install requires --health-audio with real 16 kHz mono speech")
    source_kind = ArtifactSourceKind(args.source)
    token = os.environ.get(TOKEN_ENVIRONMENT)
    source = QwenSnapshotResolver().resolve(
        spec,
        source_kind=source_kind,
        cache_only=args.cache_only,
        token=token,
    )
    preflight = store.preflight(spec, source, runtime_lock=RUNTIME_LOCK)
    receipt = store.install_from_snapshot(
        spec,
        source,
        source_kind=source_kind,
        health_probe=build_health_probe(args.health_audio, spec),
        runtime_lock=RUNTIME_LOCK,
        progress=_progress,
    )
    return {
        "artifact_key": spec.key,
        "state": "provisioned",
        "source_kind": source_kind.value,
        "model_path": receipt.model_path,
        "preflight": _preflight_payload(preflight),
        "runtime": runtime,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Provision immutable Qwen ASR artifacts")
    parser.add_argument("--store-root", type=Path, default=default_asr_model_root())
    subparsers = parser.add_subparsers(dest="command", required=True)

    install = subparsers.add_parser("install")
    install.add_argument("--artifact", choices=("release", "reference"), required=True)
    install.add_argument("--source", choices=("mirror", "upstream"), default="mirror")
    install.add_argument("--cache-only", action="store_true")
    install.add_argument("--health-audio", type=Path)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--artifact", choices=("release", "reference"), required=True)
    verify.add_argument("--deep", action="store_true")
    verify.add_argument("--health-audio", type=Path)

    refresh = subparsers.add_parser(
        "refresh",
        help="re-issue a receipt for existing bytes after a device/dtype change",
    )
    refresh.add_argument("--artifact", choices=("release", "reference"), required=True)
    refresh.add_argument("--source", choices=("mirror", "upstream"), default="upstream")
    refresh.add_argument("--health-audio", type=Path, required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--artifact", choices=("release", "reference"))
    inspect_parser.add_argument("--json", action="store_true")

    gc_parser = subparsers.add_parser("gc")
    selection = gc_parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--dry-run", action="store_true")
    selection.add_argument("--generation")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = execute(args)
    except (QwenProvisionCommandError, QwenStoreError, QwenArtifactError) as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "message": str(exc)}))
        return 2
    if args.command == "inspect" and not args.json:
        for item in payload["artifacts"]:
            print(
                f"{item['artifact_key']}\t{item['state']}\t"
                f"{item['model_path']}\t{item['detail']}"
            )
        return 0
    print(json.dumps({"ok": True, **payload}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
