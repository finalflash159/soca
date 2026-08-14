"""Profile declared SoCa workloads with a pinned py-spy and auditable receipts."""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from eval.result_io import make_eval_artifact_metadata, write_json_artifact

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_SPY_VERSION = "0.4.2"
Scenario = Literal["voice", "text_retrieval", "turn_taking"]
SCENARIOS: tuple[Scenario, ...] = ("voice", "text_retrieval", "turn_taking")


def parse_py_spy_version(output: str) -> str:
    match = re.fullmatch(r"\s*py-spy\s+(\d+\.\d+\.\d+)\s*", output)
    if match is None:
        raise ValueError("unexpected py-spy version output")
    return match.group(1)


def native_sampling_supported(*, system: str, machine: str) -> bool:
    del machine
    return system == "Linux"


def _is_python_frame(frame: Mapping[str, object]) -> bool:
    file_name = frame.get("file")
    if not isinstance(file_name, str):
        return False
    lowered = file_name.lower()
    return lowered.endswith((".py", ".pyc")) or ".py:" in lowered


def classify_speedscope(
    path: Path,
    *,
    native_available: bool,
) -> dict[str, Any]:
    """Classify exclusive sampled time by the leaf frame's implementation."""
    if not native_available:
        return {
            "status": "blocked",
            "reason": "native_sampling_unsupported",
            "method": "exclusive_leaf_frame",
            "weighted_samples": None,
            "python_percent": None,
            "native_percent": None,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    shared = payload.get("shared")
    profiles = payload.get("profiles")
    if not isinstance(shared, dict) or not isinstance(profiles, list):
        raise ValueError("invalid speedscope document")
    frames = shared.get("frames")
    if not isinstance(frames, list):
        raise ValueError("speedscope document has no shared frames")
    counts: Counter[str] = Counter()
    for profile_entry in profiles:
        if not isinstance(profile_entry, dict) or profile_entry.get("type") != "sampled":
            continue
        samples = profile_entry.get("samples")
        weights = profile_entry.get("weights")
        if not isinstance(samples, list):
            raise ValueError("sampled speedscope profile has no samples")
        if weights is None:
            weights = [1.0] * len(samples)
        if not isinstance(weights, list) or len(weights) != len(samples):
            raise ValueError("speedscope sample weights do not align")
        for sample, weight in zip(samples, weights, strict=True):
            if (
                not isinstance(sample, list)
                or not sample
                or not isinstance(weight, int | float)
                or isinstance(weight, bool)
            ):
                raise ValueError("invalid speedscope sample")
            leaf_index = sample[-1]
            if not isinstance(leaf_index, int) or not 0 <= leaf_index < len(frames):
                raise ValueError("invalid speedscope frame index")
            leaf = frames[leaf_index]
            if not isinstance(leaf, dict):
                raise ValueError("invalid speedscope frame")
            counts["python" if _is_python_frame(leaf) else "native"] += float(weight)
    total = counts.total()
    if total <= 0:
        raise ValueError("speedscope profile has no weighted samples")
    return {
        "status": "measured",
        "reason": None,
        "method": "exclusive_leaf_frame",
        "weighted_samples": total,
        "python_percent": counts["python"] / total * 100,
        "native_percent": counts["native"] / total * 100,
    }


def validate_workload_receipt(scenario: Scenario, payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("status") != "pass":
        raise ValueError("workload receipt must have status=pass")
    if payload.get("scenario") != scenario:
        raise ValueError("workload receipt scenario mismatch")
    if scenario == "turn_taking" and payload.get("scenario_count") != 120:
        raise ValueError("turn-taking receipt must prove exactly 120 scenarios")
    if scenario == "text_retrieval":
        if not isinstance(payload.get("case_count"), int) or payload["case_count"] < 1:
            raise ValueError("text retrieval receipt must prove at least one case")
        if payload.get("retrieval_called") is not True:
            raise ValueError("text retrieval receipt must prove retrieval execution")
    if scenario == "voice":
        required = {
            "microphone_capture": True,
            "speaker_playback": True,
            "turn_count": 1,
        }
        if any(payload.get(key) != value for key, value in required.items()):
            raise ValueError("voice receipt must prove one live microphone-to-speaker turn")
    return dict(payload)


def run_profile(
    *,
    scenario: Scenario,
    command: Sequence[str],
    raw_profile: Path,
    workload_receipt_path: Path,
) -> dict[str, Any]:
    py_spy = shutil.which("py-spy")
    if py_spy is None:
        return _blocked_report(scenario, "py_spy_not_installed", command)
    version_result = subprocess.run(
        [py_spy, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if version_result.returncode != 0:
        return _blocked_report(scenario, "py_spy_version_failed", command)
    try:
        version = parse_py_spy_version(version_result.stdout or version_result.stderr)
    except ValueError:
        return _blocked_report(scenario, "py_spy_version_invalid", command)
    if version != PY_SPY_VERSION:
        return _blocked_report(scenario, "py_spy_revision_mismatch", command, version=version)
    if not command:
        return _blocked_report(scenario, "workload_command_missing", command, version=version)
    raw_profile.parent.mkdir(parents=True, exist_ok=True)
    native_available = native_sampling_supported(
        system=platform.system(), machine=platform.machine()
    )
    if not native_available:
        return _blocked_report(
            scenario,
            "native_sampling_unsupported",
            command,
            version=version,
            system=platform.system(),
            machine=platform.machine(),
        )
    invocation = [
        py_spy,
        "record",
        "--format",
        "speedscope",
        "--rate",
        "100",
        "--output",
        str(raw_profile),
    ]
    if native_available:
        invocation.append("--native")
    invocation.extend(("--", *command))
    completed = subprocess.run(invocation, check=False)
    if completed.returncode != 0 or not raw_profile.is_file():
        return _blocked_report(
            scenario,
            "profile_command_failed",
            command,
            version=version,
            return_code=completed.returncode,
        )
    try:
        receipt = validate_workload_receipt(
            scenario,
            json.loads(workload_receipt_path.read_text(encoding="utf-8")),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return _blocked_report(
            scenario,
            "workload_receipt_invalid",
            command,
            version=version,
            detail=str(exc),
        )
    classification = classify_speedscope(
        raw_profile,
        native_available=native_available,
    )
    reasons = [] if classification["status"] == "measured" else [classification["reason"]]
    return {
        "schema_version": "soca-hot-path-profile-v1",
        "scenario": scenario,
        "status": "pass" if not reasons else "blocked",
        "reasons": reasons,
        "py_spy": {
            "version": version,
            "rate_hz": 100,
            "native_requested": native_available,
            "native_support_platform": native_available,
        },
        "command": list(command),
        "workload_receipt": receipt,
        "classification": classification,
        "artifact": make_eval_artifact_metadata(
            suite=f"profile_hot_paths_{scenario}",
            run_type="benchmark",
            data_files=(workload_receipt_path,),
            config={
                "py_spy_version": version,
                "rate_hz": 100,
                "native": native_available,
                "classification_method": "exclusive_leaf_frame",
                "command": list(command),
            },
            ignored_untracked_paths=(raw_profile,),
        ).to_dict(),
    }


def _blocked_report(
    scenario: Scenario,
    reason: str,
    command: Sequence[str],
    **detail: Any,
) -> dict[str, Any]:
    return {
        "schema_version": "soca-hot-path-profile-v1",
        "scenario": scenario,
        "status": "blocked",
        "reasons": [reason],
        "command": list(command),
        "detail": detail,
        "classification": {
            "status": "blocked",
            "reason": reason,
            "python_percent": None,
            "native_percent": None,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--raw-profile", type=Path, required=True)
    parser.add_argument("--workload-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    report = run_profile(
        scenario=args.scenario,
        command=command,
        raw_profile=args.raw_profile,
        workload_receipt_path=args.workload_receipt,
    )
    write_json_artifact(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
