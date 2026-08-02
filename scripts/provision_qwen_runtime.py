from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

UV_VERSION = "0.11.16"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PROJECT = REPO_ROOT / "runtime" / "qwen-asr"
DEFAULT_ENVIRONMENT = RUNTIME_PROJECT / ".venv"
RECEIPT_PATH = RUNTIME_PROJECT / ".runtime-receipt.json"


class QwenRuntimeProvisionError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise QwenRuntimeProvisionError(
            f"Command failed with exit {completed.returncode}: {' '.join(command)}"
        )


def _python_path(environment_path: Path) -> Path:
    return environment_path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _write_private_receipt(payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(RECEIPT_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        remaining = memoryview(encoded)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise QwenRuntimeProvisionError("Could not finish writing runtime receipt")
            remaining = remaining[written:]
    finally:
        os.close(descriptor)
    os.chmod(RECEIPT_PATH, 0o600)


def provision(environment_path: Path = DEFAULT_ENVIRONMENT) -> dict[str, object]:
    lock_path = RUNTIME_PROJECT / "uv.lock"
    process_environment = os.environ.copy()
    process_environment.pop("VIRTUAL_ENV", None)
    process_environment["UV_PROJECT_ENVIRONMENT"] = str(environment_path)
    uv = ["uvx", f"uv@{UV_VERSION}"]
    _run(
        [*uv, "sync", "--project", str(RUNTIME_PROJECT), "--frozen", "--no-dev"],
        environment=process_environment,
    )

    with tempfile.TemporaryDirectory(prefix="soca-qwen-worker-") as temporary:
        wheel_directory = Path(temporary)
        build_environment = os.environ.copy()
        build_environment["SOURCE_DATE_EPOCH"] = "315532800"
        _run(
            [*uv, "build", "--wheel", "--out-dir", str(wheel_directory), str(REPO_ROOT)],
            environment=build_environment,
        )
        wheels = tuple(wheel_directory.glob("soca-*.whl"))
        if len(wheels) != 1:
            raise QwenRuntimeProvisionError(f"Expected one SoCa wheel, found {len(wheels)}")
        wheel = wheels[0]
        wheel_digest = _sha256(wheel)
        _run(
            [
                *uv,
                "pip",
                "install",
                "--python",
                str(_python_path(environment_path)),
                "--no-deps",
                "--reinstall",
                str(wheel),
            ]
        )

    verify_script = """
import importlib.metadata as metadata
import json
import soca.asr.qwen_service_server

distribution = metadata.distribution("soca")
direct_url = distribution.read_text("direct_url.json")
if direct_url and json.loads(direct_url).get("dir_info", {}).get("editable"):
    raise RuntimeError("Qwen worker must not use an editable SoCa install")
import qwen_asr
print(metadata.version("qwen-asr"))
"""
    _run([str(_python_path(environment_path)), "-c", verify_script])
    receipt: dict[str, object] = {
        "schema_version": 1,
        "python": "3.11.14",
        "uv": UV_VERSION,
        "lock_sha256": _sha256(lock_path),
        "soca_wheel_sha256": wheel_digest,
        "environment": str(environment_path.resolve()),
    }
    _write_private_receipt(receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=Path, default=DEFAULT_ENVIRONMENT)
    args = parser.parse_args()
    receipt = provision(args.environment.resolve())
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
