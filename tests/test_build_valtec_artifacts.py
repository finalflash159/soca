from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts import build_valtec_artifacts as builder
from soca.tts.valtec.artifacts import resolve_valtec_onnx_artifacts
from soca.tts.valtec.manifest import sha256_file


def _write_source(root: Path) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "G.pth"
    config = root / "config.json"
    source_manifest = root / "source.json"
    checkpoint.write_bytes(b"checkpoint")
    config.write_text("{}", encoding="utf-8")
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repo_id": builder.SOURCE_REPO_ID,
                "revision": "a" * 40,
                "files": {
                    "G.pth": sha256_file(checkpoint),
                    "config.json": sha256_file(config),
                },
            }
        ),
        encoding="utf-8",
    )
    return checkpoint, config, source_manifest


def _write_fake_export(output_dir: Path) -> None:
    output_dir.mkdir()
    for name in ("text_encoder", "duration_predictor", "flow", "decoder"):
        (output_dir / f"{name}.onnx").write_bytes(name.encode())
    runtime = {
        "sample_rate": 24000,
        "hop_length": 512,
        "add_blank": True,
        "symbol_to_id": {"_": 0},
        "language_id_map": {"VI": 7},
        "tone_offset_vi": 16,
        "speaker_id_map": {"NF": 0, "SF": 1, "NM1": 2, "SM": 3, "NM2": 4},
    }
    (output_dir.parent / "tts_config.json").write_text(
        json.dumps(runtime),
        encoding="utf-8",
    )


def test_build_requires_explicit_checkpoint_trust() -> None:
    with pytest.raises(ValueError, match="--trust-checkpoint"):
        builder.build_candidate(Namespace(trust_checkpoint=False))


def test_mixed_variant_always_keeps_decoder_fp32() -> None:
    variants = builder._variant_payloads(
        quantized=True,
        eligible=True,
        selected="int8",
    )

    assert variants["int8"]["release_eligible"]
    assert variants["int8"]["runtime_graphs"]["text_encoder"] == (
        "int8/text_encoder.onnx"
    )
    assert variants["int8"]["runtime_graphs"]["decoder"] == "fp32/decoder.onnx"
    assert not variants["fp32"]["release_eligible"]


def test_candidate_variants_are_never_release_eligible() -> None:
    variants = builder._variant_payloads(
        quantized=True,
        eligible=False,
        selected="int8",
    )

    assert not any(payload["release_eligible"] for payload in variants.values())


def test_source_manifest_pins_exact_build_inputs(tmp_path: Path) -> None:
    checkpoint, config, source_manifest = _write_source(tmp_path)

    assert builder._validate_source_manifest(
        source_manifest,
        checkpoint=checkpoint,
        config=config,
    ) == "a" * 40

    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        builder._validate_source_manifest(
            source_manifest,
            checkpoint=checkpoint,
            config=config,
        )


def test_source_tree_hash_changes_with_vendored_export_code(tmp_path: Path) -> None:
    source_tree = tmp_path / "src"
    source_tree.mkdir()
    model = source_tree / "synthesizer.py"
    model.write_text("VERSION = 1\n", encoding="utf-8")
    first = builder._sha256_tree(source_tree)

    model.write_text("VERSION = 2\n", encoding="utf-8")
    second = builder._sha256_tree(source_tree)

    assert len(first) == len(second) == 64
    assert first != second


def test_build_resolves_paths_before_changing_subprocess_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    checkpoint, config, source_manifest = _write_source(tmp_path / "source")
    source_root = tmp_path / "external"
    (source_root / "src").mkdir(parents=True)
    (source_root / "src/model.py").write_text("MODEL = 1\n", encoding="utf-8")

    def fake_run(command: list[str], *, source_root: Path | None = None) -> None:
        assert source_root is not None and source_root.is_absolute()
        for flag in ("--checkpoint", "--config", "--output-dir"):
            assert Path(command[command.index(flag) + 1]).is_absolute()
        _write_fake_export(Path(command[command.index("--output-dir") + 1]))

    monkeypatch.setattr(builder, "_run", fake_run)
    args = Namespace(
        trust_checkpoint=True,
        model_root=Path("models"),
        source_root=Path("external"),
        checkpoint=checkpoint.relative_to(tmp_path),
        config=config.relative_to(tmp_path),
        source_manifest=source_manifest.relative_to(tmp_path),
        checkpoint_sha256=None,
        config_sha256=None,
        artifact_id="candidate-1",
        opset=17,
        quantize=False,
    )

    builder.build_candidate(args)

    candidate = tmp_path / "models/candidates/candidate-1"
    artifacts = resolve_valtec_onnx_artifacts(
        candidate,
        allow_candidate=True,
        verify_checksums=True,
    )
    assert artifacts.role == "candidate"
    assert artifacts.hop_length == 512
