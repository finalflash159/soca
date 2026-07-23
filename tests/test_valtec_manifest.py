from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.build_valtec_acceptance import build_acceptance
from soca.tts.valtec.artifacts import resolve_valtec_onnx_artifacts
from soca.tts.valtec.manifest import load_acceptance_report, sha256_file, write_manifest


def _raw_report(*, int8_speedup: float = 10.0, int8_quality: bool = True) -> dict:
    return {
        "pytest": True,
        "onnx_smoke": True,
        "g2p_golden": True,
        "voices_passed": ["NF", "SF", "NM1", "SM", "NM2"],
        "issue_coverage": ["#1", "#2", "#3", "#4", "#6"],
        "asr_loopback_cer": 0.08,
        "variants": {
            "fp32": {"tts_p50_ms": 180.0, "tts_p95_ms": 320.0, "rtf_p50": 0.04},
            "int8": {
                "speedup_percent_vs_fp32": int8_speedup,
                "quality_passed": int8_quality,
            },
        },
    }


def _write_runtime_files(root: Path) -> tuple[Path, Path]:
    for relative in (
        "fp32/text_encoder.onnx", "fp32/duration_predictor.onnx",
        "fp32/flow.onnx", "fp32/decoder.onnx",
        "int8/text_encoder.onnx", "int8/duration_predictor.onnx", "int8/flow.onnx",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    runtime = {
        "sample_rate": 24000,
        "hop_length": 512,
        "add_blank": True,
        "symbol_to_id": {"_": 0, "UNK": 1},
        "language_id_map": {"VI": 7},
        "tone_offset_vi": 16,
        "speaker_id_map": {"NF": 0, "SF": 1, "NM1": 2, "SM": 3, "NM2": 4},
    }
    (root / "tts_config.json").write_text(json.dumps(runtime), encoding="utf-8")
    checkpoint = root.parent / "G.pth"
    source_config = root.parent / "config.json"
    checkpoint.write_bytes(b"trusted-checkpoint")
    source_config.write_text("{}", encoding="utf-8")
    return checkpoint, source_config


def _variants(selected: str) -> dict:
    return {
        "fp32": {
            "precision": "fp32",
            "release_eligible": selected == "fp32",
            "runtime_graphs": {
                "text_encoder": "fp32/text_encoder.onnx",
                "duration_predictor": "fp32/duration_predictor.onnx",
                "flow": "fp32/flow.onnx",
                "decoder": "fp32/decoder.onnx",
            },
        },
        "int8": {
            "precision": "mixed-dynamic-int8-fp32-decoder",
            "release_eligible": selected == "int8",
            "runtime_graphs": {
                "text_encoder": "int8/text_encoder.onnx",
                "duration_predictor": "int8/duration_predictor.onnx",
                "flow": "int8/flow.onnx",
                "decoder": "fp32/decoder.onnx",
            },
        },
    }


def test_acceptance_selects_int8_only_after_speed_and_quality_gate():
    assert build_acceptance(
        _raw_report(int8_speedup=19.9), listening_reviewer="alice", license_reviewer="bob"
    )["selected_variant"] == "fp32"
    assert build_acceptance(
        _raw_report(int8_speedup=20.0), listening_reviewer="alice", license_reviewer="bob"
    )["selected_variant"] == "int8"
    assert build_acceptance(
        _raw_report(int8_speedup=30.0, int8_quality=False),
        listening_reviewer="alice",
        license_reviewer="bob",
    )["selected_variant"] == "fp32"


def test_load_acceptance_resolves_relative_raw_report_and_checks_checksum(tmp_path):
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(_raw_report()), encoding="utf-8")
    payload = build_acceptance(
        _raw_report(), listening_reviewer="alice", license_reviewer="bob"
    )
    payload.update(raw_report="raw.json", raw_report_sha256=sha256_file(raw_path))
    acceptance_path = tmp_path / "acceptance.json"
    acceptance_path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_acceptance_report(acceptance_path)["selected_variant"] == "fp32"
    raw_path.write_text("corrupted", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_acceptance_report(acceptance_path)


def test_release_manifest_resolves_mixed_graph_variant(tmp_path):
    root = tmp_path / "release-1"
    root.mkdir()
    (root / "reference").mkdir()
    (root / "reference/checkpoint_source.json").write_text(
        "{}",
        encoding="utf-8",
    )
    checkpoint, config = _write_runtime_files(root)
    acceptance = build_acceptance(
        _raw_report(int8_speedup=25.0), listening_reviewer="alice", license_reviewer="bob"
    )
    write_manifest(
        root,
        artifact_id="release-1",
        role="release",
        checkpoint=checkpoint,
        source_config=config,
        checkpoint_revision="a" * 40,
        source_tree_sha256="b" * 64,
        export_script_sha256="c" * 64,
        variants=_variants("int8"),
        active_variant="int8",
        acceptance=acceptance,
    )
    artifacts = resolve_valtec_onnx_artifacts(root, verify_checksums=True)
    assert artifacts.text_encoder.parent.name == "int8"
    assert artifacts.decoder.parent.name == "fp32"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"]["checkpoint_revision"] == "a" * 40
    assert manifest["source"]["vendored_tree_sha256"] == "b" * 64
    assert manifest["export"]["script_sha256"] == "c" * 64
    assert "revision" not in manifest["source"]