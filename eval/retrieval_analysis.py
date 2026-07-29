from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MeasurementPair:
    query_id: str
    baseline: float
    candidate: float


@dataclass(frozen=True)
class PairedResult:
    count: int
    baseline_mean: float
    candidate_mean: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float


def paired_bootstrap(
    pairs: tuple[MeasurementPair, ...],
    *,
    samples: int = 10_000,
    seed: int = 20_260_729,
) -> PairedResult:
    if not pairs:
        raise ValueError("paired analysis requires measurements")
    if samples < 100:
        raise ValueError("bootstrap requires at least 100 samples")
    differences = np.asarray(
        [item.candidate - item.baseline for item in pairs],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(pairs), size=(samples, len(pairs)))
    deltas = differences[indices].mean(axis=1)
    observed = float(differences.mean())
    ci_low, ci_high = np.quantile(deltas, (0.025, 0.975)).tolist()
    if observed == 0:
        p_value = 1.0
    else:
        opposite = deltas <= 0 if observed > 0 else deltas >= 0
        p_value = min(1.0, 2 * (int(opposite.sum()) + 1) / (samples + 1))
    return PairedResult(
        count=len(pairs),
        baseline_mean=fmean(item.baseline for item in pairs),
        candidate_mean=fmean(item.candidate for item in pairs),
        delta=observed,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=p_value,
    )


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in p_values.values()):
        raise ValueError("p-values must be finite values in [0, 1]")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for position, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (count - position)))
        adjusted[name] = running
    return adjusted


METRIC_FIELDS = (
    "recall_at_5",
    "reciprocal_rank_at_10",
    "ndcg_at_10",
    "precision_at_3",
)


def analyze_reports(
    reports: tuple[dict[str, Any], ...],
    *,
    baseline: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    by_dataset: dict[str, dict[str, dict[str, Any]]] = {}
    for report in reports:
        for result in report.get("results", ()):
            if result.get("status", "ok") != "ok":
                continue
            measurements = result.get("measurements")
            if not isinstance(measurements, list) or not measurements:
                continue
            by_dataset.setdefault(str(result["dataset"]), {})[
                str(result["candidate"])
            ] = result

    comparisons: list[dict[str, Any]] = []
    for dataset, candidates in sorted(by_dataset.items()):
        baseline_result = candidates.get(baseline)
        if baseline_result is None:
            continue
        baseline_rows = {
            str(item["query_id"]): item
            for item in baseline_result["measurements"]
        }
        dataset_results: list[dict[str, Any]] = []
        for candidate, result in sorted(candidates.items()):
            if candidate == baseline:
                continue
            candidate_rows = {
                str(item["query_id"]): item for item in result["measurements"]
            }
            common = tuple(sorted(set(baseline_rows) & set(candidate_rows)))
            if not common:
                continue
            metrics: dict[str, Any] = {}
            for metric in METRIC_FIELDS:
                pairs = tuple(
                    MeasurementPair(
                        query_id,
                        float(baseline_rows[query_id][metric]),
                        float(candidate_rows[query_id][metric]),
                    )
                    for query_id in common
                )
                metrics[metric] = asdict(
                    paired_bootstrap(
                        pairs,
                        samples=samples,
                        seed=seed,
                    )
                )
            dataset_results.append(
                {
                    "dataset": dataset,
                    "baseline": baseline,
                    "candidate": candidate,
                    "shared_queries": len(common),
                    "metrics": metrics,
                }
            )

        for metric in METRIC_FIELDS:
            adjusted = holm_adjust(
                {
                    item["candidate"]: item["metrics"][metric]["p_value"]
                    for item in dataset_results
                }
            )
            for item in dataset_results:
                metric_result = item["metrics"][metric]
                metric_result["holm_p_value"] = adjusted[item["candidate"]]
                metric_result["practical_tie"] = (
                    abs(metric_result["delta"]) < 0.005
                    and metric_result["ci_low"] <= 0 <= metric_result["ci_high"]
                )
        comparisons.extend(dataset_results)
    return {
        "schema_version": 1,
        "baseline": baseline,
        "bootstrap_samples": samples,
        "seed": seed,
        "comparisons": comparisons,
    }


def _output_path(argument: Path | None) -> Path:
    if argument is not None:
        return argument
    run_dir = os.environ.get("SOCA_BENCHMARK_RUN_DIR")
    if not run_dir:
        raise ValueError("--output or SOCA_BENCHMARK_RUN_DIR is required")
    return Path(run_dir) / "analysis.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired analysis for retrieval reports.")
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--baseline", default="bm25")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_729)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.report
    )
    analysis = analyze_reports(
        reports,
        baseline=args.baseline,
        samples=args.samples,
        seed=args.seed,
    )
    output = _output_path(args.output)
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(output)
    print(f"analysis: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
