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


def test_boh_ablation_manifest_marks_measurements_as_pending() -> None:
    manifest_path = Path("eval/experimental/asr_boh/baseline_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "historical_reference_pending_no_boh_rerun"
    assert manifest["production_after"]["boh_auto_loaded"] is False
    assert "no_boh_vs_experimental_boh WER and CER on Vietnamese speech" in manifest[
        "required_release_measurements"
    ]
