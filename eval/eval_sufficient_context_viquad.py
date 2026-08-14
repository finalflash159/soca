"""Evaluate the sufficient-context autorater on pinned UIT-ViQuAD 2.0 labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.result_io import make_eval_artifact_metadata, write_json_artifact
from soca.config import SecretStore, load_settings
from soca.core.sufficient_context import (
    ContextSufficiencyAssessor,
    RetrievedContext,
    SufficiencyAssessmentError,
    SufficiencyStatus,
    SufficientContextAutorater,
)
from soca.llm.base import StructuredLLMEngine
from soca.llm.providers import RemoteOpenAILLM, get_provider

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_REPO = "taidng/UIT-ViQuAD2.0"
DATASET_REVISION = "406f09a45cc106a8f7b7fd0c25078883fe58cb1f"
DATASET_SPLIT = "validation"
DEFAULT_PER_CLASS = 115
DEFAULT_SEED = 73
DEFAULT_FALSE_SUFFICIENT_MAX = 0.05
DEFAULT_SUFFICIENT_RECALL_MIN = 0.90
DEFAULT_PUBLIC_OUTPUT = REPO_ROOT / "eval" / "results" / "sufficient_context_viquad.json"
DEFAULT_RAW_OUTPUT = REPO_ROOT / "artifacts" / "local" / "sufficient_context_viquad.jsonl"


@dataclass(frozen=True)
class LabeledContext:
    case_id: str
    question: str
    context: str
    expected_sufficient: bool


@dataclass(frozen=True)
class CaseResult:
    case_id_sha256: str
    expected_sufficient: bool
    predicted_sufficient: bool | None
    confidence: float | None
    reason_code: str | None
    model_id: str
    prompt_sha256: str
    provider_trace: Mapping[str, Any]
    usage: Mapping[str, int | float]
    error_code: str | None
    elapsed_ms: float
    question: str = ""
    context: str = ""

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "case_id_sha256": self.case_id_sha256,
            "expected_sufficient": self.expected_sufficient,
            "predicted_sufficient": self.predicted_sufficient,
            "confidence": self.confidence,
            "reason_code": self.reason_code,
            "model_id": self.model_id,
            "prompt_sha256": self.prompt_sha256,
            "provider_trace": dict(self.provider_trace),
            "usage": dict(self.usage),
            "error_code": self.error_code,
            "elapsed_ms": self.elapsed_ms,
        }

    def as_private_dict(self) -> dict[str, Any]:
        return {
            **self.as_public_dict(),
            "question": self.question,
            "context": self.context,
        }


def load_labeled_contexts(rows: Iterable[Mapping[str, object]]) -> tuple[LabeledContext, ...]:
    contexts: list[LabeledContext] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        case_id = row.get("id")
        question = row.get("question")
        context = row.get("context")
        is_impossible = row.get("is_impossible")
        if (
            not isinstance(case_id, str)
            or not case_id.strip()
            or not isinstance(question, str)
            or not question.strip()
            or not isinstance(context, str)
            or not context.strip()
            or not isinstance(is_impossible, bool)
        ):
            raise ValueError(
                f"row {index} requires non-empty id/question/context and boolean is_impossible"
            )
        normalized_id = case_id.strip()
        if normalized_id in seen_ids:
            raise ValueError(f"duplicate UIT-ViQuAD id: {normalized_id}")
        seen_ids.add(normalized_id)
        contexts.append(
            LabeledContext(
                case_id=normalized_id,
                question=question.strip(),
                context=context.strip(),
                expected_sufficient=not is_impossible,
            )
        )
    if not contexts:
        raise ValueError("UIT-ViQuAD split is empty")
    return tuple(contexts)


def _sample_key(item: LabeledContext, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{item.case_id}".encode()).hexdigest()


def select_balanced_contexts(
    contexts: Sequence[LabeledContext],
    *,
    per_class: int,
    seed: int,
) -> tuple[LabeledContext, ...]:
    if per_class < 1:
        raise ValueError("per_class must be positive")
    selected: list[LabeledContext] = []
    for expected in (True, False):
        candidates = sorted(
            (item for item in contexts if item.expected_sufficient is expected),
            key=lambda item: _sample_key(item, seed),
        )
        if len(candidates) < per_class:
            raise ValueError(
                f"expected_sufficient={expected} has {len(candidates)} rows; "
                f"need {per_class}"
            )
        selected.extend(candidates[:per_class])
    return tuple(sorted(selected, key=lambda item: _sample_key(item, seed + 1)))


def evaluate_contexts(
    contexts: Sequence[LabeledContext],
    assessor: ContextSufficiencyAssessor,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[CaseResult, ...]:
    results: list[CaseResult] = []
    for index, item in enumerate(contexts, start=1):
        evidence = RetrievedContext(
            evidence_id=item.case_id,
            text=item.context,
            provenance={"dataset": DATASET_REPO, "split": DATASET_SPLIT},
        )
        started = time.perf_counter()
        try:
            decision = assessor.assess(item.question, (evidence,))
        except SufficiencyAssessmentError as exc:
            results.append(
                CaseResult(
                    case_id_sha256=hashlib.sha256(item.case_id.encode()).hexdigest(),
                    expected_sufficient=item.expected_sufficient,
                    predicted_sufficient=None,
                    confidence=None,
                    reason_code=None,
                    model_id="",
                    prompt_sha256="",
                    provider_trace={},
                    usage={},
                    error_code=exc.code,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    question=item.question,
                    context=item.context,
                )
            )
            if progress is not None:
                progress(index, len(contexts))
            continue
        results.append(
            CaseResult(
                case_id_sha256=hashlib.sha256(item.case_id.encode()).hexdigest(),
                expected_sufficient=item.expected_sufficient,
                predicted_sufficient=decision.status is SufficiencyStatus.SUFFICIENT,
                confidence=decision.confidence,
                reason_code=decision.reason_code,
                model_id=decision.model_id,
                prompt_sha256=decision.prompt_sha256,
                provider_trace=decision.provider_trace,
                usage=decision.usage,
                error_code=None,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                question=item.question,
                context=item.context,
            )
        )
        if progress is not None:
            progress(index, len(contexts))
    return tuple(results)


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    if successes < 0 or successes > total:
        raise ValueError("successes must be between zero and total")
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    spread = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total)
    return max(0.0, (centre - spread) / denominator), min(
        1.0, (centre + spread) / denominator
    )


def aggregate_results(
    results: Sequence[CaseResult],
    *,
    false_sufficient_max: float,
    sufficient_recall_min: float,
) -> dict[str, Any]:
    if not results:
        raise ValueError("at least one evaluation result is required")
    if not 0 <= false_sufficient_max <= 1 or not 0 <= sufficient_recall_min <= 1:
        raise ValueError("gate thresholds must be between zero and one")
    evaluated = [item for item in results if item.predicted_sufficient is not None]
    tp = sum(item.expected_sufficient and item.predicted_sufficient is True for item in evaluated)
    fn = sum(item.expected_sufficient and item.predicted_sufficient is False for item in evaluated)
    fp = sum(not item.expected_sufficient and item.predicted_sufficient is True for item in evaluated)
    tn = sum(not item.expected_sufficient and item.predicted_sufficient is False for item in evaluated)
    sufficient_total = tp + fn
    insufficient_total = fp + tn
    false_sufficient_rate = fp / insufficient_total if insufficient_total else None
    sufficient_recall = tp / sufficient_total if sufficient_total else None
    accuracy = (tp + tn) / len(evaluated) if evaluated else None
    false_interval = wilson_interval(fp, insufficient_total)
    recall_interval = wilson_interval(tp, sufficient_total)
    failures = Counter(item.error_code for item in results if item.error_code)
    reasons: list[str] = []
    if failures:
        reasons.append("assessment_failures")
    if false_sufficient_rate is None:
        reasons.append("missing_insufficient_class")
    elif false_sufficient_rate > false_sufficient_max:
        reasons.append("false_sufficient_rate")
    if sufficient_recall is None:
        reasons.append("missing_sufficient_class")
    elif sufficient_recall < sufficient_recall_min:
        reasons.append("sufficient_recall")
    usage: Counter[str] = Counter()
    latencies: list[float] = []
    for item in evaluated:
        for key in ("prompt_tokens", "completion_tokens"):
            value = item.usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                usage[key] += value
        latency = item.usage.get("total_latency_ms")
        if isinstance(latency, int | float) and not isinstance(latency, bool):
            latencies.append(float(latency))
    return {
        "case_count": len(results),
        "evaluated_count": len(evaluated),
        "assessment_failures": dict(sorted(failures.items())),
        "confusion_matrix": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "accuracy": accuracy,
        "false_sufficient_rate": false_sufficient_rate,
        "false_sufficient_rate_wilson_95": list(false_interval),
        "sufficient_recall": sufficient_recall,
        "sufficient_recall_wilson_95": list(recall_interval),
        "usage": dict(sorted(usage.items())),
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p95": _percentile(latencies, 0.95),
        },
        "gate": {
            "passed": not reasons,
            "reasons": reasons,
            "false_sufficient_max": false_sufficient_max,
            "sufficient_recall_min": sufficient_recall_min,
        },
    }


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(quantile * len(ordered)) - 1
    return ordered[max(0, min(rank, len(ordered) - 1))]


def _load_dataset_rows() -> Sequence[Mapping[str, object]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the eval extra to load UIT-ViQuAD") from exc
    dataset = load_dataset(
        DATASET_REPO,
        revision=DATASET_REVISION,
        split=DATASET_SPLIT,
    )
    return dataset


def _build_remote_assessor(model: str | None) -> SufficientContextAutorater:
    settings = load_settings()
    if settings.backend != "remote":
        raise RuntimeError("UIT-ViQuAD release evaluation requires a configured remote model")
    provider = get_provider(settings.provider_key)
    key = SecretStore(dotenv_path=REPO_ROOT / ".env").get_key(provider.key)
    if not key:
        raise RuntimeError(f"missing API key for {provider.key}")
    model_id = model or settings.model_id
    engine = RemoteOpenAILLM(
        provider,
        model_id,
        key,
        reasoning_enabled=settings.effective_reasoning_enabled,
        reasoning_parameter=settings.model_reasoning_parameter,
        max_output_tokens=settings.effective_max_tokens,
    )
    if not isinstance(engine, StructuredLLMEngine):
        raise RuntimeError("configured remote model does not support structured output")
    return SufficientContextAutorater(engine, model_id=model_id)


def _write_private_results(path: Path, results: Sequence[CaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for result in results:
            output.write(json.dumps(result.as_private_dict(), ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--false-sufficient-max", type=float, default=DEFAULT_FALSE_SUFFICIENT_MAX)
    parser.add_argument("--sufficient-recall-min", type=float, default=DEFAULT_SUFFICIENT_RECALL_MIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_PUBLIC_OUTPUT)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    all_contexts = load_labeled_contexts(_load_dataset_rows())
    selected = select_balanced_contexts(
        all_contexts,
        per_class=args.per_class,
        seed=args.seed,
    )
    assessor = _build_remote_assessor(args.model)
    results = evaluate_contexts(
        selected,
        assessor,
        progress=lambda completed, total: (
            print(f"evaluated {completed}/{total} sufficient-context cases", flush=True)
            if completed == total or completed % 10 == 0
            else None
        ),
    )
    _write_private_results(args.raw_output, results)
    aggregate = aggregate_results(
        results,
        false_sufficient_max=args.false_sufficient_max,
        sufficient_recall_min=args.sufficient_recall_min,
    )
    inventory = Counter(item.expected_sufficient for item in all_contexts)
    report = {
        "schema_version": "soca-sufficient-context-viquad-v1",
        "artifact": make_eval_artifact_metadata(
            suite="sufficient_context_viquad",
            run_type="benchmark",
            data_files=(),
            config={
                "dataset_repo": DATASET_REPO,
                "dataset_revision": DATASET_REVISION,
                "dataset_split": DATASET_SPLIT,
                "sample_seed": args.seed,
                "per_class": args.per_class,
                "model": assessor.model_id,
            },
            ignored_untracked_paths=(args.raw_output, args.output),
        ).to_dict(),
        "dataset": {
            "repo": DATASET_REPO,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "row_count": len(all_contexts),
            "answerable_count": inventory[True],
            "unanswerable_count": inventory[False],
        },
        "sample": {"seed": args.seed, "per_class": args.per_class},
        "aggregate": aggregate,
    }
    write_json_artifact(args.output, report)
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    return int(not aggregate["gate"]["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
