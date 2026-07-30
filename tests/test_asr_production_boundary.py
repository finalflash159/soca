from __future__ import annotations

import inspect
import json
from pathlib import Path

import soca.asr
from soca.asr.robust_asr import RobustASR


def test_production_asr_does_not_expose_or_reference_boh() -> None:
    assert not hasattr(soca.asr, "VietnameseBoH")
    assert "VietnameseBoH" not in inspect.getsource(RobustASR)
    assert "ahocorasick" not in inspect.getsource(RobustASR)


def test_boh_ablation_manifest_records_current_paired_measurement() -> None:
    manifest_path = Path("eval/experimental/asr_boh/baseline_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "current_paired_ablation_complete"
    assert manifest["production"]["boh_auto_loaded"] is False
    assert manifest["run"]["source_dirty"] is False
    assert manifest["run"]["results"]["changed_predictions"] == 0
    assert manifest["decision"]["production_boh"] == "removed"
