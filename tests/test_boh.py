"""Unit tests for soca.asr.boh."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soca.asr.boh import VietnameseBoH


@pytest.fixture
def boh_artifact(tmp_path: Path) -> Path:
    """Create a minimal BoH JSON for testing."""
    payload = {
        "metadata": {"model_key": "test"},
        "boh": [
            {"phrase": "cảm ơn các bạn đã xem video", "count": 12, "keep": True},
            {"phrase": "hẹn gặp lại các bạn", "count": 7, "keep": True},
            {"phrase": "đăng ký kênh", "count": 5, "keep": True},
            {"phrase": "phrase to skip", "count": 3, "keep": False},  # explicitly skipped
        ],
    }
    artifact = tmp_path / "vi_boh_test.json"
    artifact.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return artifact


class TestVietnameseBoH:
    def test_load_artifact(self, boh_artifact):
        boh = VietnameseBoH(boh_path=boh_artifact)
        # Skipped phrase should not be loaded
        assert len(boh) == 3
        assert "phrase to skip" not in boh.boh_phrases
        assert boh.model_key == "test"
        assert boh.is_compatible_with("test") is True
        assert boh.is_compatible_with("phowhisper_tiny") is False

    def test_model_artifact_path_uses_model_key(self):
        path = VietnameseBoH.path_for_model("phowhisper_base")
        assert path.name == "phowhisper_base_vi_boh_v1.json"
        assert path.parent.name == "boh"

    def test_missing_artifact_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            VietnameseBoH(boh_path=tmp_path / "missing.json")

    def test_match_single_phrase(self, boh_artifact):
        boh = VietnameseBoH(boh_path=boh_artifact)
        result = boh.match_and_clean("cảm ơn các bạn đã xem video xin chào mọi người")
        assert "cảm ơn các bạn đã xem video" in result.matched_phrases
        assert "xin chào mọi người" in result.cleaned_text
        assert "cảm ơn các bạn" not in result.cleaned_text

    def test_match_case_insensitive(self, boh_artifact):
        boh = VietnameseBoH(boh_path=boh_artifact)
        # Capitalized input should still match lowercase BoH entries
        result = boh.match_and_clean("Cảm Ơn Các Bạn Đã Xem Video.")
        assert len(result.matched_phrases) == 1

    def test_no_match(self, boh_artifact):
        boh = VietnameseBoH(boh_path=boh_artifact)
        result = boh.match_and_clean("hôm nay trời rất đẹp")
        assert result.matched_phrases == ()
        assert result.cleaned_text == "hôm nay trời rất đẹp"
        assert result.n_chars_removed == 0

    def test_empty_text(self, boh_artifact):
        boh = VietnameseBoH(boh_path=boh_artifact)
        result = boh.match_and_clean("")
        assert result.matched_phrases == ()
        assert result.cleaned_text == ""

    def test_multiple_disjoint_matches(self, boh_artifact):
        boh = VietnameseBoH(boh_path=boh_artifact)
        result = boh.match_and_clean("cảm ơn các bạn đã xem video và đăng ký kênh nhé")
        assert len(result.matched_phrases) == 2

    def test_empty_boh_loads_safely(self, tmp_path):
        payload = {"metadata": {}, "boh": []}
        artifact = tmp_path / "empty.json"
        artifact.write_text(json.dumps(payload), encoding="utf-8")
        boh = VietnameseBoH(boh_path=artifact)
        assert len(boh) == 0
        # Match should pass through unchanged
        result = boh.match_and_clean("bất kỳ text nào")
        assert result.cleaned_text == "bất kỳ text nào"

    def test_repr(self, boh_artifact):
        boh = VietnameseBoH(boh_path=boh_artifact)
        assert "VietnameseBoH" in repr(boh)
        assert "n_phrases=3" in repr(boh)
