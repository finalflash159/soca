"""Evaluate natural Vietnamese disfluency audio without committing private speech.

The repository stores reviewed scenario specifications. A local materialization
maps those IDs to hashed 16 kHz mono WAV files, while endpoint and tool receipts
prove the actual production paths were exercised. Missing evidence is an error,
never a synthetic pass.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import soundfile as sf

from eval.result_io import make_eval_artifact_metadata, write_json_artifact

DisfluencyType = Literal[
    "filler",
    "pause",
    "hesitation",
    "false_start",
    "self_correction",
]
DISFLUENCY_TYPES = frozenset(
    {"filler", "pause", "hesitation", "false_start", "self_correction"}
)


@dataclass(frozen=True)
class ScenarioSpec:
    case_id: str
    language: str
    disfluency_type: DisfluencyType
    transcript: str
    expected_tool: str
    expected_terminal: str


@dataclass(frozen=True)
class MaterializedCase:
    spec: ScenarioSpec
    audio_path: Path
    audio_sha256: str
    true_end_ms: float
    hold_spans_ms: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class EndpointReceipt:
    stopped: bool
    stop_ms: float | None
    policy: str = "production"
    model_revision: str = ""


@dataclass(frozen=True)
class ToolReceipt:
    tool_name: str
    terminal: str
    provider: str
    model: str
    route: str = "controlled"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scenario_specs(path: Path) -> tuple[ScenarioSpec, ...]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    specs: list[ScenarioSpec] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"scenario row {index} must be an object")
        case_id = str(row.get("id", "")).strip()
        kind = str(row.get("disfluency_type", "")).strip()
        if not case_id or case_id in seen:
            raise ValueError(f"scenario row {index} has an empty or duplicate id")
        if kind not in DISFLUENCY_TYPES:
            raise ValueError(f"scenario {case_id} has an invalid disfluency type")
        language = str(row.get("language", "")).strip()
        transcript = str(row.get("transcript", "")).strip()
        expected_tool = str(row.get("expected_tool", "")).strip()
        expected_terminal = str(row.get("expected_terminal", "")).strip()
        if language != "vi" or not transcript or not expected_tool or not expected_terminal:
            raise ValueError(f"scenario {case_id} has an incomplete contract")
        seen.add(case_id)
        specs.append(
            ScenarioSpec(
                case_id=case_id,
                language=language,
                disfluency_type=kind,  # type: ignore[arg-type]
                transcript=transcript,
                expected_tool=expected_tool,
                expected_terminal=expected_terminal,
            )
        )
    observed = {spec.disfluency_type for spec in specs}
    if observed != DISFLUENCY_TYPES:
        missing = sorted(DISFLUENCY_TYPES - observed)
        raise ValueError(f"scenario suite is missing disfluency types: {missing}")
    return tuple(specs)


def load_materialized_cases(
    specs: tuple[ScenarioSpec, ...],
    manifest_path: Path,
) -> tuple[MaterializedCase, ...]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "soca-disfluency-audio-v1":
        raise ValueError("unsupported disfluency audio manifest schema")
    if payload.get("source") != "reviewed_private_vietnamese_speech":
        raise ValueError("disfluency audio must be reviewed private Vietnamese speech")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("disfluency audio manifest cases must be a list")
    by_id = {spec.case_id: spec for spec in specs}
    materialized: list[MaterializedCase] = []
    seen: set[str] = set()
    for row in raw_cases:
        if not isinstance(row, dict):
            raise ValueError("disfluency audio case must be an object")
        case_id = str(row.get("id", "")).strip()
        if case_id not in by_id or case_id in seen:
            raise ValueError(f"unexpected or duplicate materialized case: {case_id}")
        raw_path = str(row.get("wav_path", "")).strip()
        audio_path = (manifest_path.parent / raw_path).resolve()
        if not audio_path.is_file():
            raise ValueError(f"missing disfluency audio: {audio_path}")
        expected_hash = str(row.get("wav_sha256", "")).strip()
        actual_hash = _sha256(audio_path)
        if expected_hash != actual_hash:
            raise ValueError(f"audio hash mismatch for {case_id}")
        info = sf.info(audio_path)
        if info.samplerate != 16_000 or info.channels != 1:
            raise ValueError(f"{case_id}: audio must be 16 kHz mono")
        true_end_ms = float(row.get("true_end_ms", 0.0))
        duration_ms = info.frames / info.samplerate * 1000.0
        if true_end_ms <= 0 or true_end_ms > duration_ms:
            raise ValueError(f"{case_id}: true_end_ms is outside the audio")
        raw_spans = row.get("hold_spans_ms", [])
        if not isinstance(raw_spans, list) or not raw_spans:
            raise ValueError(f"{case_id}: at least one hold span is required")
        spans: list[tuple[float, float]] = []
        for raw_span in raw_spans:
            if not isinstance(raw_span, list) or len(raw_span) != 2:
                raise ValueError(f"{case_id}: invalid hold span")
            start, end = float(raw_span[0]), float(raw_span[1])
            if start < 0 or end <= start or end > true_end_ms:
                raise ValueError(f"{case_id}: hold span is outside the active turn")
            spans.append((start, end))
        seen.add(case_id)
        materialized.append(
            MaterializedCase(
                spec=by_id[case_id],
                audio_path=audio_path,
                audio_sha256=actual_hash,
                true_end_ms=true_end_ms,
                hold_spans_ms=tuple(spans),
            )
        )
    if seen != set(by_id):
        raise ValueError(f"materialization is missing cases: {sorted(set(by_id) - seen)}")
    return tuple(materialized)


def evaluate_disfluency(
    cases: tuple[MaterializedCase, ...],
    *,
    endpoint_receipts: dict[str, EndpointReceipt],
    tool_receipts: dict[str, ToolReceipt],
    max_over_wait_ms: float = 2_000.0,
) -> dict[str, Any]:
    expected_ids = {case.spec.case_id for case in cases}
    if set(endpoint_receipts) != expected_ids or set(tool_receipts) != expected_ids:
        raise ValueError("endpoint and tool receipts must cover the exact scenario set")
    rows: list[dict[str, Any]] = []
    for case in cases:
        endpoint = endpoint_receipts[case.spec.case_id]
        tool = tool_receipts[case.spec.case_id]
        stopped_in_hold = bool(
            endpoint.stop_ms is not None
            and any(start <= endpoint.stop_ms < end for start, end in case.hold_spans_ms)
        )
        endpoint_pass = bool(
            endpoint.stopped
            and endpoint.stop_ms is not None
            and endpoint.stop_ms >= case.true_end_ms
            and endpoint.stop_ms - case.true_end_ms <= max_over_wait_ms
            and not stopped_in_hold
        )
        tool_pass = bool(
            tool.tool_name == case.spec.expected_tool
            and tool.terminal == case.spec.expected_terminal
            and tool.route == "controlled"
            and tool.provider.strip()
            and tool.model.strip()
        )
        rows.append(
            {
                "id": case.spec.case_id,
                "disfluency_type": case.spec.disfluency_type,
                "audio_sha256": case.audio_sha256,
                "endpoint": asdict(endpoint),
                "tool": asdict(tool),
                "stopped_in_hold": stopped_in_hold,
                "endpoint_pass": endpoint_pass,
                "tool_terminal_pass": tool_pass,
            }
        )
    n = len(rows)
    endpoint_accuracy = sum(bool(row["endpoint_pass"]) for row in rows) / n
    tool_accuracy = sum(bool(row["tool_terminal_pass"]) for row in rows) / n
    by_type = Counter(str(row["disfluency_type"]) for row in rows)
    gate_pass = endpoint_accuracy >= 0.95 and tool_accuracy >= 0.95
    return {
        "schema_version": "soca-disfluency-eval-v1",
        "gate_status": "pass" if gate_pass else "fail",
        "scenario_count": n,
        "scenario_count_by_type": dict(sorted(by_type.items())),
        "endpoint_hold_accuracy": endpoint_accuracy,
        "tool_terminal_accuracy": tool_accuracy,
        "thresholds": {
            "min_endpoint_hold_accuracy": 0.95,
            "min_tool_terminal_accuracy": 0.95,
            "max_over_wait_ms": max_over_wait_ms,
        },
        "cases": rows,
    }


def _load_endpoint_receipts(path: Path) -> dict[str, EndpointReceipt]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "soca-disfluency-endpoint-receipts-v1":
        raise ValueError("unsupported endpoint receipt schema")
    return {
        str(case_id): EndpointReceipt(**receipt)
        for case_id, receipt in payload.get("receipts", {}).items()
    }


def _load_tool_receipts(path: Path) -> dict[str, ToolReceipt]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "soca-disfluency-tool-receipts-v1":
        raise ValueError("unsupported tool receipt schema")
    return {
        str(case_id): ToolReceipt(**receipt)
        for case_id, receipt in payload.get("receipts", {}).items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--audio-manifest", type=Path, required=True)
    parser.add_argument("--endpoint-receipts", type=Path, required=True)
    parser.add_argument("--tool-receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    specs = load_scenario_specs(args.scenarios)
    cases = load_materialized_cases(specs, args.audio_manifest)
    report = evaluate_disfluency(
        cases,
        endpoint_receipts=_load_endpoint_receipts(args.endpoint_receipts),
        tool_receipts=_load_tool_receipts(args.tool_receipts),
    )
    report["metadata"] = make_eval_artifact_metadata(
        suite="disfluency_vi",
        run_type="benchmark",
        data_files=(
            args.scenarios,
            args.audio_manifest,
            args.endpoint_receipts,
            args.tool_receipts,
            *(case.audio_path for case in cases),
        ),
        config={"scenario_ids": [case.spec.case_id for case in cases]},
        ignored_untracked_paths=(args.output,),
    ).to_dict()
    write_json_artifact(args.output, report)
    return 0 if report["gate_status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
