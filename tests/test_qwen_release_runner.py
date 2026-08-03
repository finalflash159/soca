from pathlib import Path

import click
import pytest

from eval.run_qwen_asr_release import _snapshot_config


def test_snapshot_config_preserves_exact_release_config(tmp_path: Path) -> None:
    source = tmp_path / "config.json"
    source.write_bytes(b'{"schema_version":1}\n')
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    import hashlib

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    _snapshot_config(source, run_dir, expected_digest=digest)

    assert (run_dir / "benchmark_config.json").read_bytes() == source.read_bytes()


def test_snapshot_config_rejects_identity_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "config.json"
    source.write_bytes(b'{"schema_version":1}\n')
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(click.ClickException, match="benchmark config changed"):
        _snapshot_config(source, run_dir, expected_digest="0" * 64)
