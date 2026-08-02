from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.tts_intelligibility.manifest import (
    SynthManifest,
    SynthRecord,
    read_manifest,
    write_manifest,
)


class TestManifestRoundTrip:
    def _manifest(self) -> SynthManifest:
        return SynthManifest(
            engine="valtec-onnx",
            voice="NF",
            records=(
                SynthRecord(
                    item_id="lexicon-word-000",
                    corpus="lexicon",
                    text_in="Mô hình dùng cosine để so sánh kết quả",
                    expected="cosine",
                    mode="term",
                    wav_path="wavs/lexicon-word-000.wav",
                    sample_rate=24000,
                    tts_latency_ms=12.5,
                    audio_duration_ms=1500.0,
                ),
            ),
        )

    def test_round_trip_preserves_records(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        write_manifest(self._manifest(), path)
        loaded = read_manifest(path)
        assert loaded.engine == "valtec-onnx"
        assert loaded.records[0].expected == "cosine"
        assert loaded.records[0].sample_rate == 24000

    def test_diacritics_survive_the_json_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        write_manifest(self._manifest(), path)
        assert "Mô hình" in path.read_text(encoding="utf-8")

    def test_missing_file_raises_a_readable_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Cannot read synth manifest"):
            read_manifest(tmp_path / "nope.json")

    def test_manifest_without_records_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"engine": "x"}), encoding="utf-8")
        with pytest.raises(ValueError, match="no 'records' list"):
            read_manifest(path)

    def test_malformed_record_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        path.write_text(
            json.dumps({"engine": "x", "records": [{"item_id": "only"}]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Malformed record"):
            read_manifest(path)
