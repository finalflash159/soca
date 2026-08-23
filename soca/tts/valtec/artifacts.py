# soca/tts/valtec/artifacts.py
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from soca.model_paths import default_model_root

VALTEC_MODEL_ROOT = default_model_root() / "tts" / "valtec_multispeaker"
ARTIFACT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


@dataclass(frozen=True)
class ValtecOnnxArtifacts:
    root: Path
    role: str
    artifact_id: str
    variant: str
    precision: str
    text_encoder: Path
    duration_predictor: Path
    flow: Path
    decoder: Path
    config: Path
    manifest: Path
    manifest_sha256: str
    sample_rate: int
    hop_length: int
    noise_scale: float
    length_scale: float
    tone_offset_vi: int
    language_id_vi: int
    add_blank: bool
    voice_map: dict[str, int]
    default_voice: str


def resolve_valtec_onnx_artifacts(
    root: Path,
    *,
    variant: str | None = None,
    allow_reference: bool = False,
    allow_candidate: bool = False,
    verify_checksums: bool = False,
) -> ValtecOnnxArtifacts:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Valtec manifest: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported Valtec manifest schema_version")
    if payload.get("model_key") != "valtec_multispeaker":
        raise ValueError("Valtec manifest model_key mismatch")
    artifact_id = str(payload.get("artifact_id", ""))
    if ARTIFACT_ID_RE.fullmatch(artifact_id) is None:
        raise ValueError("Valtec manifest artifact_id is invalid")

    role = str(payload.get("role", ""))
    if role == "reference":
        if not allow_reference:
            raise ValueError(
                "Reference Valtec artifact is not allowed in production runtime"
            )
    elif role == "candidate":
        if not allow_candidate:
            raise ValueError(
                "Candidate Valtec artifact is not allowed in production runtime"
            )
    elif role != "release":
        raise ValueError(f"Unsupported Valtec artifact role: {role!r}")

    runtime_defaults = payload.get("runtime_defaults")
    if not isinstance(runtime_defaults, dict):
        raise ValueError("Valtec manifest runtime_defaults must be an object")
    sample_rate = int(runtime_defaults.get("sample_rate", 0))
    hop_length = int(runtime_defaults.get("hop_length", 0))
    noise_scale = float(runtime_defaults.get("noise_scale", -1.0))
    length_scale = float(runtime_defaults.get("length_scale", 0.0))
    tone_offset_vi = int(runtime_defaults.get("tone_offset_vi", -1))
    language_id_vi = int(runtime_defaults.get("language_id_vi", -1))
    add_blank = runtime_defaults.get("add_blank")
    if sample_rate <= 0 or hop_length <= 0:
        raise ValueError("Valtec sample_rate and hop_length must be positive")
    if noise_scale < 0.0 or length_scale <= 0.0:
        raise ValueError("Valtec noise_scale/length_scale are invalid")
    if tone_offset_vi < 0 or language_id_vi < 0 or not isinstance(add_blank, bool):
        raise ValueError("Valtec tone/language/add_blank defaults are invalid")

    voices = payload.get("voices")
    if not isinstance(voices, dict) or not isinstance(voices.get("map"), dict):
        raise ValueError("Valtec manifest voices.map must be an object")
    voice_map = {str(name): int(speaker_id) for name, speaker_id in voices["map"].items()}
    if not voice_map or any(speaker_id < 0 for speaker_id in voice_map.values()):
        raise ValueError("Valtec manifest voice map is empty or contains a negative id")
    if len(set(voice_map.values())) != len(voice_map):
        raise ValueError("Valtec manifest speaker ids must be unique")
    default_voice = str(voices.get("default", ""))
    if default_voice not in voice_map:
        raise ValueError("Valtec manifest default voice is not present in voices.map")

    variants = payload.get("variants")
    if not isinstance(variants, dict):
        raise ValueError("Valtec manifest variants must be an object")
    selected_variant = variant or payload.get("active_variant")
    if not isinstance(selected_variant, str) or selected_variant not in variants:
        raise ValueError(f"Unknown Valtec artifact variant: {selected_variant!r}")
    variant_payload = variants[selected_variant]
    if not isinstance(variant_payload, dict):
        raise ValueError("Valtec variant payload must be an object")
    if role == "release" and selected_variant == payload.get("active_variant"):
        if variant_payload.get("release_eligible") is not True:
            raise ValueError("Active Valtec variant has not passed release gates")

    runtime_graphs = variant_payload.get("runtime_graphs")
    expected_graphs = {"text_encoder", "duration_predictor", "flow", "decoder"}
    if not isinstance(runtime_graphs, dict) or set(runtime_graphs) != expected_graphs:
        raise ValueError("Valtec runtime_graphs must define exactly four graphs")
    runtime_files = payload.get("runtime_files")
    expected_files = {"config"}
    if not isinstance(runtime_files, dict) or set(runtime_files) != expected_files:
        raise ValueError("Valtec runtime_files is incomplete")

    def resolve_relative(value: object) -> tuple[str, Path]:
        if not isinstance(value, str) or not value:
            raise ValueError("Valtec manifest file path must be a non-empty string")
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe Valtec manifest path: {value!r}")
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Valtec manifest path escaped artifact root: {value!r}")
        return relative.as_posix(), resolved

    graph_paths = {name: resolve_relative(path) for name, path in runtime_graphs.items()}
    shared_paths = {name: resolve_relative(path) for name, path in runtime_files.items()}
    all_paths = [*graph_paths.values(), *shared_paths.values()]
    missing = [relative for relative, path in all_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Valtec ONNX artifacts missing from "
            f"{root}: {', '.join(missing)}. Run scripts/build_valtec_artifacts.py."
        )

    checksums = payload.get("files")
    if not isinstance(checksums, dict):
        raise ValueError("Valtec manifest files/checksums must be an object")
    missing_checksums = [relative for relative, _ in all_paths if relative not in checksums]
    if missing_checksums:
        raise ValueError(f"Valtec manifest missing checksums: {missing_checksums}")
    if verify_checksums:
        checksum_paths: list[tuple[str, Path]] = []
        for relative, expected in checksums.items():
            if re.fullmatch(r"[0-9a-f]{64}", str(expected)) is None:
                raise ValueError(f"Invalid Valtec artifact checksum: {relative}")
            checksum_paths.append(resolve_relative(relative))
        missing_verified = [relative for relative, path in checksum_paths if not path.is_file()]
        if missing_verified:
            raise FileNotFoundError(f"Valtec checksummed files are missing: {missing_verified}")
        for relative, path in checksum_paths:
            actual = _sha256_file(path)
            if actual != checksums[relative]:
                raise ValueError(f"Valtec artifact checksum mismatch: {relative}")

    return ValtecOnnxArtifacts(
        root=root,
        role=role,
        artifact_id=artifact_id,
        variant=selected_variant,
        precision=str(variant_payload.get("precision", "unknown")),
        text_encoder=graph_paths["text_encoder"][1],
        duration_predictor=graph_paths["duration_predictor"][1],
        flow=graph_paths["flow"][1],
        decoder=graph_paths["decoder"][1],
        config=shared_paths["config"][1],
        manifest=manifest_path,
        manifest_sha256=_sha256_file(manifest_path),
        sample_rate=sample_rate,
        hop_length=hop_length,
        noise_scale=noise_scale,
        length_scale=length_scale,
        tone_offset_vi=tone_offset_vi,
        language_id_vi=language_id_vi,
        add_blank=add_blank,
        voice_map=voice_map,
        default_voice=default_voice,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_current_valtec_release(
    model_root: Path = VALTEC_MODEL_ROOT,
) -> Path:
    pointer_path = model_root / "current.json"
    payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    artifact_id = str(payload.get("artifact_id", ""))
    if ARTIFACT_ID_RE.fullmatch(artifact_id) is None:
        raise ValueError(f"Invalid Valtec artifact_id in {pointer_path}")
    expected_manifest_sha256 = str(payload.get("manifest_sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256) is None:
        raise ValueError(f"Invalid Valtec manifest_sha256 in {pointer_path}")

    release_root = (model_root / "releases" / artifact_id).resolve()
    expected_parent = (model_root / "releases").resolve()
    if release_root.parent != expected_parent:
        raise ValueError("Valtec release path escaped the releases directory")
    actual_manifest_sha256 = _sha256_file(release_root / "manifest.json")
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError("Active Valtec manifest checksum does not match current.json")
    resolve_valtec_onnx_artifacts(release_root)  # Validate before returning.
    return release_root


def activate_valtec_release(artifact_id: str, model_root: Path = VALTEC_MODEL_ROOT) -> None:
    if ARTIFACT_ID_RE.fullmatch(artifact_id) is None:
        raise ValueError(f"Invalid Valtec artifact_id: {artifact_id!r}")
    release_root = model_root / "releases" / artifact_id
    resolve_valtec_onnx_artifacts(release_root, verify_checksums=True)
    manifest_sha256 = _sha256_file(release_root / "manifest.json")

    pointer_path = model_root / "current.json"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=model_root,
            prefix=".current.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(
                {
                    "artifact_id": artifact_id,
                    "manifest_sha256": manifest_sha256,
                },
                handle,
                indent=2,
            )
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, pointer_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
