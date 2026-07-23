from __future__ import annotations

import argparse
import json
from pathlib import Path

from soca.tts.valtec.manifest import REQUIRED_ISSUES, sha256_file

VOICES = ["NF", "SF", "NM1", "SM", "NM2"]


def build_acceptance(raw: dict, *, listening_reviewer: str, license_reviewer: str) -> dict:
    fp32 = raw["variants"]["fp32"]
    if fp32["tts_p50_ms"] > 250 or fp32["tts_p95_ms"] > 450 or fp32["rtf_p50"] > 0.12:
        raise ValueError("FP32 candidate misses latency/RTF release thresholds")
    if raw["asr_loopback_cer"] > 0.15:
        raise ValueError("Valtec ASR loopback CER exceeds 0.15")
    if raw["voices_passed"] != VOICES:
        raise ValueError("Listening/eval must pass all five voices in canonical order")
    if set(raw["issue_coverage"]) < REQUIRED_ISSUES:
        raise ValueError("Raw report does not cover all required upstream issues")
    selected = "fp32"
    int8 = raw["variants"].get("int8")
    if int8 is not None:
        speedup = float(int8["speedup_percent_vs_fp32"])
        if speedup >= 20.0 and int8["quality_passed"] is True:
            selected = "int8"
    return {
        "schema_version": 1,
        "selected_variant": selected,
        "voices_passed": VOICES,
        "gates": {
            "pytest": raw["pytest"] is True,
            "onnx_smoke": raw["onnx_smoke"] is True,
            "tts_eval": True,
            "g2p_golden": raw["g2p_golden"] is True,
            "voice_listening": bool(listening_reviewer.strip()),
            "asr_loopback": True,
            "license_verified": bool(license_reviewer.strip()),
        },
        "issue_coverage": sorted(REQUIRED_ISSUES),
        "metrics": {
            "fp32_tts_p50_ms": float(fp32["tts_p50_ms"]),
            "fp32_tts_p95_ms": float(fp32["tts_p95_ms"]),
            "fp32_rtf_p50": float(fp32["rtf_p50"]),
            "int8_speedup_percent": None if int8 is None else float(int8["speedup_percent_vs_fp32"]),
            "asr_loopback_cer": float(raw["asr_loopback_cer"]),
        },
        "reviewers": {
            "voice_listening": listening_reviewer.strip(),
            "license": license_reviewer.strip(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a fail-closed Valtec acceptance report.")
    parser.add_argument("--raw-report", type=Path, required=True)
    parser.add_argument("--listening-approved-by", required=True)
    parser.add_argument("--license-approved-by", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.raw_report.read_text(encoding="utf-8"))
    acceptance = build_acceptance(
        raw,
        listening_reviewer=args.listening_approved_by,
        license_reviewer=args.license_approved_by,
    )
    acceptance["raw_report"] = str(args.raw_report.resolve())
    acceptance["raw_report_sha256"] = sha256_file(args.raw_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())