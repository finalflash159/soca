from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REQUIRED_ISSUES = {"#1", "#2", "#3", "#4", "#6"}
REQUIRED_GATES = {
    "pytest",
    "onnx_smoke",
    "tts_eval",
    "g2p_golden",
    "voice_listening",
    "asr_loopback",
    "license_verified",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_acceptance_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported Valtec acceptance report schema")
    gates = payload.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("Valtec acceptance report gates must be an object")
    missing = REQUIRED_GATES - set(gates)
    failed = sorted(name for name in REQUIRED_GATES if gates.get(name) is not True)
    if missing or failed:
        raise ValueError(f"Valtec acceptance gates incomplete/failed: missing={sorted(missing)}, failed={failed}")
    issue_coverage = set(payload.get("issue_coverage", []))
    if not REQUIRED_ISSUES <= issue_coverage:
        raise ValueError(f"Valtec acceptance report lacks issue coverage: {sorted(REQUIRED_ISSUES - issue_coverage)}")
    selected_variant = payload.get("selected_variant")
    if selected_variant not in {"fp32", "int8"}:
        raise ValueError("Valtec acceptance selected_variant must be fp32 or int8")
    if payload.get("voices_passed") != ["NF", "SF", "NM1", "SM", "NM2"]:
        raise ValueError("Valtec acceptance must pass all five voices in canonical order")
    metrics = payload.get("metrics", {})
    if (
        float(metrics.get("fp32_tts_p50_ms", float("inf"))) > 250
        or float(metrics.get("fp32_tts_p95_ms", float("inf"))) > 450
        or float(metrics.get("fp32_rtf_p50", float("inf"))) > 0.06
        or float(metrics.get("asr_loopback_cer", float("inf"))) > 0.15
    ):
        raise ValueError("Valtec acceptance metrics miss release thresholds")
    reviewers = payload.get("reviewers", {})
    if not reviewers.get("voice_listening") or not reviewers.get("license"):
        raise ValueError("Valtec acceptance reviewer identities are missing")
    raw_report = Path(str(payload.get("raw_report", "")))
    if not raw_report.is_absolute():
        raw_report = path.parent / raw_report
    if not raw_report.is_file() or sha256_file(raw_report) != payload.get("raw_report_sha256"):
        raise ValueError("Valtec raw eval report is missing or checksum mismatch")
    return payload


def _checksums(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def write_manifest(
    root: Path,
    *,
    artifact_id: str,
    role: str,
    checkpoint: Path,
    source_config: Path,
    checkpoint_revision: str,
    source_tree_sha256: str,
    export_script_sha256: str,
    variants: dict[str, dict[str, Any]],
    active_variant: str,
    acceptance: dict[str, Any] | None,
) -> Path:
    if role == "candidate" and acceptance is not None:
        raise ValueError("Candidate manifest must not claim acceptance")
    if role == "release" and acceptance is None:
        raise ValueError("Release manifest requires validated acceptance")
    if role not in {"candidate", "release"}:
        raise ValueError(f"Manifest writer does not support role: {role}")
    if re.fullmatch(r"[0-9a-f]{40}", checkpoint_revision) is None:
        raise ValueError("Checkpoint revision must be a full Git commit")
    for label, value in {
        "source_tree_sha256": source_tree_sha256,
        "export_script_sha256": export_script_sha256,
    }.items():
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"Invalid {label}")
    runtime = json.loads((root / "tts_config.json").read_text(encoding="utf-8"))
    checkpoint_source_manifest = root / "reference/checkpoint_source.json"
    if not checkpoint_source_manifest.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint source manifest: {checkpoint_source_manifest}"
        )
    voices = runtime.get("speaker_id_map")
    if not isinstance(voices, dict) or set(voices) != {"NF", "SF", "NM1", "SM", "NM2"}:
        raise ValueError("Runtime config must contain the verified five-voice speaker_id_map")
    payload = {
        "schema_version": 1,
        "artifact_kind": "soca-valtec-onnx",
        "artifact_id": artifact_id,
        "role": role,
        "model_key": "valtec_multispeaker",
        "active_variant": active_variant,
        "variants": variants,
        "runtime_files": {"config": "tts_config.json"},
        "runtime_defaults": {
            "sample_rate": int(runtime["sample_rate"]),
            "hop_length": int(runtime["hop_length"]),
            "noise_scale": 0.667,
            "length_scale": 1.0,
            "tone_offset_vi": int(runtime["tone_offset_vi"]),
            "language_id_vi": int(runtime["language_id_map"]["VI"]),
            "add_blank": bool(runtime["add_blank"]),
        },
        "voices": {
            "source": "source_config.data.spk2id",
            "map": {name: int(speaker_id) for name, speaker_id in voices.items()},
            "default": "NF",
        },
        "created_at": datetime.now(UTC).isoformat(),
        "source": {
            "code_repo": "https://github.com/tronghieuit/valtec-tts",
            "vendored_tree_sha256": source_tree_sha256,
            "checkpoint_repo": "valtecAI-team/valtec-tts-pretrained",
            "checkpoint_revision": checkpoint_revision,
            "checkpoint_source_manifest": "reference/checkpoint_source.json",
            "license": "CC BY-NC; redistribution requires explicit release review",
        },
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": sha256_file(checkpoint),
            "trusted_local_file": True,
        },
        "config": {
            "source_path": str(source_config.resolve()),
            "runtime_path": "tts_config.json",
            "sha256": sha256_file(source_config),
        },
        "export": {
            "opset": 17,
            "split_graphs": ["text_encoder", "duration_predictor", "flow", "decoder"],
            "dynamic_axes": True,
            "script": "scripts/export_valtec_onnx.py",
            "script_sha256": export_script_sha256,
        },
        "quantization": {
            "mode": "dynamic_qint8_selective" if "int8" in variants else "none",
            "decoder": "fp32",
        },
        "acceptance": acceptance,
    }
    payload["files"] = _checksums(root)
    path = root / "manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
