from __future__ import annotations

import inspect
import json
import tomllib
from pathlib import Path

import soca.asr
from soca.asr.robust_asr import RobustASR


def test_production_asr_does_not_expose_or_reference_boh() -> None:
    production_sources = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(Path("soca").rglob("*.py"))
    )

    assert not hasattr(soca.asr, "VietnameseBoH")
    assert "VietnameseBoH" not in inspect.getsource(RobustASR)
    assert "ahocorasick" not in inspect.getsource(RobustASR)
    assert "VietnameseBoH" not in production_sources
    assert "ahocorasick" not in production_sources


def test_boh_dependency_and_tooling_are_experimental_only() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    core_dependencies = project["dependencies"]
    eval_dependencies = project["optional-dependencies"]["eval"]

    assert not any("ahocorasick" in dependency for dependency in core_dependencies)
    assert any("ahocorasick" in dependency for dependency in eval_dependencies)
    assert not Path("local/build_boh.py").exists()
    assert not Path("local/boh_manual_review.py").exists()
    assert "RUNTIME_BOH_PATH" not in Path("local/config.py").read_text(encoding="utf-8")


def test_boh_ablation_manifest_records_current_paired_measurement() -> None:
    manifest_path = Path("eval/experimental/asr_boh/baseline_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "current_paired_ablation_complete"
    assert manifest["production"]["boh_auto_loaded"] is False
    assert manifest["run"]["source_dirty"] is False
    assert manifest["run"]["results"]["changed_predictions"] == 0
    assert manifest["decision"]["production_boh"] == "removed"
