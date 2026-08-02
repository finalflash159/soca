from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.asr_release_config import (
    ASRReleaseConfig,
    ASRReleaseConfigError,
    SplitPolicy,
    load_release_config,
)
from eval.asr_release_runner import load_release_items

REPO_ROOT = Path(__file__).resolve().parents[1]


def _require_release_manifests(config: ASRReleaseConfig) -> None:
    missing = [
        str(REPO_ROOT / contract.manifest)
        for contract in config.datasets.values()
        if not (REPO_ROOT / contract.manifest).is_file()
    ]
    if missing:
        pytest.skip("release corpora are local-only; missing manifests: " + ", ".join(missing))


def test_release_config_is_strict_and_verifies_local_manifests() -> None:
    config = load_release_config(REPO_ROOT / "eval/gates/qwen_asr_release.json")
    _require_release_manifests(config)

    assert config.artifacts == ("qwen3_asr_0_6b", "qwen3_asr_1_7b")
    assert config.verify_datasets(REPO_ROOT) == {
        name: contract.expected_sha256 for name, contract in config.datasets.items()
    }
    assert config.release_gates["fallback_attempt_count_max"] == 0
    assert config.decode["final_budget_candidates"] == [128, 256, 512]


def test_release_config_rejects_unknown_fields(tmp_path: Path) -> None:
    source = json.loads((REPO_ROOT / "eval/gates/qwen_asr_release.json").read_text())
    source["unexpected"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(source))

    with pytest.raises(ASRReleaseConfigError, match="fields do not match"):
        load_release_config(path)


def test_dataset_digest_mismatch_is_typed(tmp_path: Path) -> None:
    source = json.loads((REPO_ROOT / "eval/gates/qwen_asr_release.json").read_text())
    source["datasets"]["fleurs_vi"]["manifest"] = "manifest.jsonl"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(source))
    (tmp_path / "manifest.jsonl").write_text("changed\n")
    config = load_release_config(path)

    with pytest.raises(ASRReleaseConfigError, match="digest mismatch"):
        config.datasets["fleurs_vi"].verify(tmp_path)


def test_split_policy_is_deterministic_and_disjoint() -> None:
    policy = SplitPolicy("sha256_id_modulo_v1", calibration_buckets=60, total_buckets=100)

    first = [policy.cohort(f"item-{index}", seed=42) for index in range(100)]
    second = [policy.cohort(f"item-{index}", seed=42) for index in range(100)]

    assert first == second
    assert set(first) == {"calibration", "holdout"}


def test_fleurs_recordings_have_unique_ids_without_splitting_duplicate_transcripts() -> None:
    config = load_release_config(REPO_ROOT / "eval/gates/qwen_asr_release.json")
    _require_release_manifests(config)

    rows = load_release_items(config, REPO_ROOT)["fleurs_vi"]
    repeated = [row for row in rows if row.item_id.startswith("1884:")]

    assert len({row.item_id for row in rows}) == len(rows)
    assert len(repeated) == 2
    assert len({row.cohort for row in repeated}) == 1
