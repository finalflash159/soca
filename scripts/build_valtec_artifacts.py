from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from soca.tts.valtec.artifacts import resolve_valtec_onnx_artifacts
from soca.tts.valtec.manifest import load_acceptance_report, sha256_file, write_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / "models/tts/valtec_multispeaker"
SOURCE_ROOT = REPO_ROOT / "external/valtec-tts"
EXPORT_SCRIPT = REPO_ROOT / "scripts/export_valtec_onnx.py"
QUANTIZE_SCRIPT = REPO_ROOT / "scripts/quantize_valtec_onnx.py"
SOURCE_REPO_ID = "valtecAI-team/valtec-tts-pretrained"


def _variant_payloads(*, quantized: bool, eligible: bool, selected: str) -> dict:
    variants = {
        "fp32": {
            "precision": "fp32",
            "release_eligible": eligible and selected == "fp32",
            "runtime_graphs": {
                "text_encoder": "fp32/text_encoder.onnx",
                "duration_predictor": "fp32/duration_predictor.onnx",
                "flow": "fp32/flow.onnx",
                "decoder": "fp32/decoder.onnx",
            },
        }
    }
    if quantized:
        variants["int8"] = {
            "precision": "mixed-dynamic-int8-fp32-decoder",
            "release_eligible": eligible and selected == "int8",
            "runtime_graphs": {
                "text_encoder": "int8/text_encoder.onnx",
                "duration_predictor": "int8/duration_predictor.onnx",
                "flow": "int8/flow.onnx",
                "decoder": "fp32/decoder.onnx",
            },
        }
    return variants


def _run(command: list[str], *, source_root: Path | None = None) -> None:
    environment = os.environ.copy()
    cwd = REPO_ROOT
    if source_root is not None:
        old_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(source_root)
            if not old_pythonpath
            else os.pathsep.join((str(source_root), old_pythonpath))
        )
        cwd = source_root
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def _validate_source_manifest(
    path: Path,
    *,
    checkpoint: Path,
    config: Path,
) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    revision = str(payload.get("revision", ""))
    if payload.get("schema_version") != 1 or payload.get("repo_id") != SOURCE_REPO_ID:
        raise ValueError("Invalid Valtec checkpoint source manifest")
    if len(revision) != 40 or any(
        char not in "0123456789abcdef" for char in revision.lower()
    ):
        raise ValueError("Valtec checkpoint source revision must be a full Git commit")
    expected_paths = {
        "G.pth": checkpoint.resolve(),
        "config.json": config.resolve(),
    }
    files = payload.get("files")
    if not isinstance(files, dict) or set(files) != set(expected_paths):
        raise ValueError("Valtec checkpoint source file allow-list is invalid")
    for name, source_path in expected_paths.items():
        if source_path != (path.parent / name).resolve():
            raise ValueError(f"Valtec build input must be the manifested source file: {name}")
        if not source_path.is_file() or sha256_file(source_path) != files[name]:
            raise ValueError(f"Valtec checkpoint source checksum mismatch: {name}")
    return revision


def _sha256_tree(root: Path) -> str:
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    files = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    ]
    if not files:
        raise ValueError(f"Valtec source tree is empty: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def build_candidate(args: argparse.Namespace) -> None:
    if not args.trust_checkpoint:
        raise ValueError("build requires --trust-checkpoint")
    model_root = args.model_root.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    config = args.config.expanduser().resolve()
    source_manifest = args.source_manifest.expanduser().resolve()
    for path in (checkpoint, config, source_manifest, source_root / "src"):
        if not path.exists():
            raise FileNotFoundError(path)
    checkpoint_revision = _validate_source_manifest(
        source_manifest,
        checkpoint=checkpoint,
        config=config,
    )
    source_tree_sha256 = _sha256_tree(source_root / "src")
    export_script_sha256 = sha256_file(EXPORT_SCRIPT)
    if args.checkpoint_sha256 and sha256_file(checkpoint) != args.checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 mismatch")
    if args.config_sha256 and sha256_file(config) != args.config_sha256:
        raise ValueError("config SHA-256 mismatch")

    candidate = model_root / "candidates" / args.artifact_id
    if candidate.exists() or (model_root / "releases" / args.artifact_id).exists():
        raise FileExistsError(f"Valtec artifact id already exists: {args.artifact_id}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.mkdir()
    try:
        (candidate / "reference").mkdir()
        shutil.copy2(
            source_manifest,
            candidate / "reference/checkpoint_source.json",
        )
        shutil.copy2(config, candidate / "source_config.json")
        _run(
            [
                sys.executable,
                str(EXPORT_SCRIPT),
                "--checkpoint", str(checkpoint),
                "--config", str(config),
                "--output-dir", str(candidate / "fp32"),
                "--opset", str(args.opset),
                "--trust-checkpoint",
            ],
            source_root=source_root,
        )
        if args.quantize:
            _run(
                [
                    sys.executable,
                    str(QUANTIZE_SCRIPT),
                    "--fp32-dir", str(candidate / "fp32"),
                    "--output-dir", str(candidate / "int8"),
                ]
            )
        variants = _variant_payloads(quantized=args.quantize, eligible=False, selected="fp32")
        write_manifest(
            candidate,
            artifact_id=args.artifact_id,
            role="candidate",
            checkpoint=checkpoint,
            source_config=config,
            checkpoint_revision=checkpoint_revision,
            source_tree_sha256=source_tree_sha256,
            export_script_sha256=export_script_sha256,
            variants=variants,
            active_variant="fp32",
            acceptance=None,
        )
        resolve_valtec_onnx_artifacts(
            candidate,
            allow_candidate=True,
            verify_checksums=True,
        )
    except BaseException:
        shutil.rmtree(candidate, ignore_errors=True)
        raise


def finalize_candidate(args: argparse.Namespace) -> None:
    candidate = args.model_root / "candidates" / args.artifact_id
    release = args.model_root / "releases" / args.artifact_id
    if release.exists():
        raise FileExistsError(f"Immutable Valtec release already exists: {release}")
    current = resolve_valtec_onnx_artifacts(
        candidate,
        allow_candidate=True,
        verify_checksums=True,
    )
    old_manifest = json.loads(current.manifest.read_text(encoding="utf-8"))
    acceptance = load_acceptance_report(args.acceptance_report)
    selected = str(acceptance["selected_variant"])
    if selected not in old_manifest["variants"]:
        raise ValueError(f"Acceptance selected unavailable variant: {selected}")
    reference = candidate / "reference"
    reference.mkdir(exist_ok=True)
    raw_report = Path(str(acceptance["raw_report"]))
    if not raw_report.is_absolute():
        raw_report = args.acceptance_report.parent / raw_report
    shutil.copy2(raw_report, reference / "raw_report.json")
    portable_acceptance = {
        **acceptance,
        "raw_report": "raw_report.json",
    }
    (reference / "acceptance.json").write_text(
        json.dumps(portable_acceptance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    variants = _variant_payloads(
        quantized="int8" in old_manifest["variants"],
        eligible=True,
        selected=selected,
    )
    write_manifest(
        candidate,
        artifact_id=args.artifact_id,
        role="release",
        checkpoint=Path(old_manifest["checkpoint"]["path"]),
        source_config=Path(old_manifest["config"]["source_path"]),
        checkpoint_revision=str(old_manifest["source"]["checkpoint_revision"]),
        source_tree_sha256=str(old_manifest["source"]["vendored_tree_sha256"]),
        export_script_sha256=str(old_manifest["export"]["script_sha256"]),
        variants=variants,
        active_variant=selected,
        acceptance=portable_acceptance,
    )
    resolve_valtec_onnx_artifacts(candidate, verify_checksums=True)
    release.parent.mkdir(parents=True, exist_ok=True)
    candidate.rename(release)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build/finalize SoCa-owned Valtec artifacts.")
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--artifact-id", required=True)
    build.add_argument("--checkpoint", type=Path, required=True)
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--source-manifest", type=Path, required=True)
    build.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    build.add_argument("--checkpoint-sha256")
    build.add_argument("--config-sha256")
    build.add_argument("--opset", type=int, default=17)
    build.add_argument("--quantize", action="store_true")
    build.add_argument("--trust-checkpoint", action="store_true")
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--artifact-id", required=True)
    finalize.add_argument("--acceptance-report", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "build":
        build_candidate(args)
    else:
        finalize_candidate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
