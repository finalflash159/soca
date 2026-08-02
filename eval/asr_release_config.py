from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ASRReleaseConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DatasetContract:
    manifest: Path
    expected_sha256: str
    usage: str
    repo_id: str | None = None
    revision: str | None = None
    config: str | None = None
    split: str | None = None

    def verify(self, root: Path) -> str:
        path = (root / self.manifest).resolve()
        if not path.is_file():
            raise ASRReleaseConfigError(f"dataset manifest is missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != self.expected_sha256:
            raise ASRReleaseConfigError(
                f"dataset manifest digest mismatch for {self.manifest}: {digest}"
            )
        return digest


@dataclass(frozen=True, slots=True)
class SplitPolicy:
    algorithm: str
    calibration_buckets: int
    total_buckets: int

    def __post_init__(self) -> None:
        if self.algorithm != "sha256_id_modulo_v1":
            raise ASRReleaseConfigError(f"unsupported split algorithm: {self.algorithm}")
        if not 0 < self.calibration_buckets < self.total_buckets:
            raise ASRReleaseConfigError("calibration buckets must be inside total buckets")

    def cohort(self, item_id: str, *, seed: int) -> str:
        if not item_id:
            raise ASRReleaseConfigError("split item id must not be empty")
        digest = hashlib.sha256(f"{seed}:{item_id}".encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") % self.total_buckets
        return "calibration" if bucket < self.calibration_buckets else "holdout"


@dataclass(frozen=True, slots=True)
class ASRReleaseConfig:
    schema_version: int
    run_name: str
    seed: int
    artifacts: tuple[str, ...]
    datasets: dict[str, DatasetContract]
    split_policy: SplitPolicy
    context_variants: tuple[str, ...]
    decode: dict[str, Any]
    repetitions: dict[str, int]
    threshold_selection: dict[str, float]
    context_echo_candidates: dict[str, Any]
    release_gates: dict[str, float | int]
    source_path: Path
    digest: str

    def verify_datasets(self, root: Path) -> dict[str, str]:
        return {name: contract.verify(root) for name, contract in self.datasets.items()}


def load_release_config(path: Path) -> ASRReleaseConfig:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ASRReleaseConfigError(f"cannot read release config: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ASRReleaseConfigError("unsupported ASR release config schema")
    expected = {
        "schema_version",
        "run_name",
        "seed",
        "artifacts",
        "datasets",
        "split_policy",
        "context_variants",
        "decode",
        "repetitions",
        "threshold_selection",
        "context_echo_candidates",
        "release_gates",
    }
    if set(payload) != expected:
        raise ASRReleaseConfigError("ASR release config fields do not match schema")

    datasets_raw = _mapping(payload, "datasets")
    datasets = {name: _dataset_contract(name, value) for name, value in datasets_raw.items()}
    required_datasets = {"private_codeswitch", "fleurs_vi", "non_speech"}
    if set(datasets) != required_datasets:
        raise ASRReleaseConfigError("release dataset contracts are incomplete")

    split_raw = _mapping(payload, "split_policy")
    split_policy = SplitPolicy(
        algorithm=_string(split_raw, "algorithm"),
        calibration_buckets=_integer(split_raw, "calibration_buckets"),
        total_buckets=_integer(split_raw, "total_buckets"),
    )
    artifacts = _string_tuple(payload, "artifacts")
    if len(artifacts) != len(set(artifacts)) or len(artifacts) != 2:
        raise ASRReleaseConfigError("release config requires two unique artifact keys")
    variants = _string_tuple(payload, "context_variants")
    if set(variants) != {"empty", "production_catalog"}:
        raise ASRReleaseConfigError("context variants must cover empty and production catalog")

    repetitions = _numeric_mapping(payload, "repetitions", integer=True)
    if any(value < 1 for value in repetitions.values()):
        raise ASRReleaseConfigError("benchmark repetitions must be positive")
    threshold_selection = _numeric_mapping(payload, "threshold_selection")
    context_echo_candidates = dict(_mapping(payload, "context_echo_candidates"))
    if set(context_echo_candidates) != {
        "minimum_unique_tokens",
        "token_overlap_thresholds",
        "contiguous_token_thresholds",
        "selected_min_contiguous_tokens",
    }:
        raise ASRReleaseConfigError("context echo candidate fields do not match schema")
    minimum_unique_tokens = _integer(context_echo_candidates, "minimum_unique_tokens")
    overlap_thresholds = context_echo_candidates.get("token_overlap_thresholds")
    contiguous_thresholds = context_echo_candidates.get("contiguous_token_thresholds")
    selected_contiguous = _integer(context_echo_candidates, "selected_min_contiguous_tokens")
    if (
        minimum_unique_tokens < 1
        or not isinstance(overlap_thresholds, list)
        or not overlap_thresholds
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1
            for value in overlap_thresholds
        )
        or not isinstance(contiguous_thresholds, list)
        or not contiguous_thresholds
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in contiguous_thresholds
        )
        or selected_contiguous not in contiguous_thresholds
    ):
        raise ASRReleaseConfigError("context echo candidate sweep is invalid")
    release_gates = _numeric_mapping(payload, "release_gates")
    decode = dict(_mapping(payload, "decode"))
    if set(decode) != {
        "language",
        "max_new_tokens",
        "final_budget_candidates",
        "budget_probe_timeout_s",
    }:
        raise ASRReleaseConfigError("decode fields do not match schema")
    final_budget_candidates = decode.get("final_budget_candidates")
    if (
        _string(decode, "language") != "Vietnamese"
        or _integer(decode, "max_new_tokens") < 1
        or not isinstance(final_budget_candidates, list)
        or not final_budget_candidates
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in final_budget_candidates
        )
        or final_budget_candidates != sorted(set(final_budget_candidates))
        or decode["max_new_tokens"] not in final_budget_candidates
        or not isinstance(decode["budget_probe_timeout_s"], (int, float))
        or isinstance(decode["budget_probe_timeout_s"], bool)
        or decode["budget_probe_timeout_s"] <= 0
    ):
        raise ASRReleaseConfigError("decode contract is invalid")

    return ASRReleaseConfig(
        schema_version=1,
        run_name=_string(payload, "run_name"),
        seed=_integer(payload, "seed"),
        artifacts=artifacts,
        datasets=datasets,
        split_policy=split_policy,
        context_variants=variants,
        decode=decode,
        repetitions={key: int(value) for key, value in repetitions.items()},
        threshold_selection=threshold_selection,
        context_echo_candidates=context_echo_candidates,
        release_gates=release_gates,
        source_path=path.resolve(),
        digest=hashlib.sha256(raw).hexdigest(),
    )


def _dataset_contract(name: str, value: object) -> DatasetContract:
    if not isinstance(name, str) or not name or not isinstance(value, dict):
        raise ASRReleaseConfigError("dataset contracts must be named objects")
    required = {"manifest", "expected_sha256", "usage"}
    optional = {"repo_id", "revision", "config", "split"}
    if not required <= set(value) or set(value) - required - optional:
        raise ASRReleaseConfigError(f"dataset contract fields are invalid: {name}")
    digest = _string(value, "expected_sha256")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ASRReleaseConfigError(f"dataset digest is invalid: {name}")
    return DatasetContract(
        manifest=Path(_string(value, "manifest")),
        expected_sha256=digest,
        usage=_string(value, "usage"),
        repo_id=_optional_string(value, "repo_id"),
        revision=_optional_string(value, "revision"),
        config=_optional_string(value, "config"),
        split=_optional_string(value, "split"),
    )


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ASRReleaseConfigError(f"{key} must be an object")
    return value


def _string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ASRReleaseConfigError(f"{key} must be a non-empty string")
    return value


def _optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is not None and (not isinstance(value, str) or not value):
        raise ASRReleaseConfigError(f"{key} must be null or a non-empty string")
    return value


def _integer(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ASRReleaseConfigError(f"{key} must be an integer")
    return value


def _string_tuple(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise ASRReleaseConfigError(f"{key} must be a non-empty list")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ASRReleaseConfigError(f"{key} must contain non-empty strings")
    return result


def _numeric_mapping(
    payload: dict[str, Any], key: str, *, integer: bool = False
) -> dict[str, float | int]:
    value = _mapping(payload, key)
    result: dict[str, float | int] = {}
    for name, item in value.items():
        if not isinstance(name, str) or not name:
            raise ASRReleaseConfigError(f"{key} keys must be non-empty strings")
        if isinstance(item, bool) or not isinstance(item, int if integer else (int, float)):
            raise ASRReleaseConfigError(f"{key}.{name} must be numeric")
        result[name] = item
    return result


__all__ = [
    "ASRReleaseConfig",
    "ASRReleaseConfigError",
    "DatasetContract",
    "SplitPolicy",
    "load_release_config",
]
