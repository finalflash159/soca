from __future__ import annotations

from pathlib import Path

import pytest

from scripts import download_valtec_checkpoint as module


def test_download_checkpoint_is_staged_hashed_and_idempotent(tmp_path, monkeypatch):
    destination = tmp_path / "source"
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        (local_dir / "G.pth").write_bytes(b"trusted-checkpoint")
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(module, "snapshot_download", fake_snapshot_download)
    first = module.download_checkpoint(destination, revision="a" * 40)
    second = module.download_checkpoint(destination, revision="a" * 40)

    assert first == second == destination.resolve()
    assert len(calls) == 1
    assert calls[0]["repo_id"] == module.REPO_ID
    assert calls[0]["revision"] == "a" * 40
    assert set(calls[0]["allow_patterns"]) == {"G.pth", "config.json"}
    payload = module.verify_source(destination, revision="a" * 40)
    assert payload["files"]["G.pth"] == module.sha256_file(destination / "G.pth")


def test_missing_download_never_publishes_destination(tmp_path, monkeypatch):
    destination = tmp_path / "source"

    def incomplete_snapshot_download(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(module, "snapshot_download", incomplete_snapshot_download)

    with pytest.raises(FileNotFoundError, match="G.pth"):
        module.download_checkpoint(destination, revision="b" * 40)
    assert not destination.exists()


def test_existing_source_with_changed_bytes_fails_closed(tmp_path, monkeypatch):
    destination = tmp_path / "source"

    def fake_snapshot_download(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        (local_dir / "G.pth").write_bytes(b"checkpoint")
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(module, "snapshot_download", fake_snapshot_download)
    module.download_checkpoint(destination, revision="c" * 40)
    (destination / "G.pth").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="checksum mismatch"):
        module.download_checkpoint(destination, revision="c" * 40)
