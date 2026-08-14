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
from soca.config import LlmSettings, SecretStore, load_settings
from soca.config.llm_settings import ReasoningParameter
from soca.core.sufficient_context import (
    ContextSufficiencyAssessor,
    RetrievedContext,
    SufficiencyAssessmentError,
    SufficiencyPromptVariant,
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
RELEASE_MINIMUM_REVIEWED_PER_CLASS = 20
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


def apply_reviewed_labels(
    contexts: Sequence[LabeledContext],
    manifest: Mapping[str, object],
) -> tuple[LabeledContext, ...]:
    """Replace extractive-MRC proxy labels with reviewed semantic labels."""
    if manifest.get("dataset_revision") != DATASET_REVISION:
        raise ValueError("reviewed labels use the wrong dataset revision")
    if manifest.get("label_definition") != "sufficient_context_semantic_v1":
        raise ValueError("reviewed labels use an unknown label definition")
    reviewers = manifest.get("reviewers")
    if (
        not isinstance(reviewers, list)
        or len({item.strip() for item in reviewers if isinstance(item, str) and item.strip()}) < 2
    ):
        raise ValueError("reviewed labels require at least two reviewers")
    reviewer_ids = {
        item.strip() for item in reviewers if isinstance(item, str) and item.strip()
    }
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("reviewed labels require a cases list")
    labels: dict[str, bool] = {}
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ValueError(f"reviewed label {index} must be an object")
        case_id = raw.get("case_id")
        expected = raw.get("expected_sufficient")
        reviewer_labels = raw.get("reviewer_labels")
        if (
            not isinstance(case_id, str)
            or not case_id.strip()
            or not isinstance(expected, bool)
        ):
            raise ValueError(
                f"reviewed label {index} requires case_id and boolean label"
            )
        if (
            not isinstance(reviewer_labels, dict)
            or set(reviewer_labels) != reviewer_ids
            or any(
                not isinstance(label, bool) or label is not expected
                for label in reviewer_labels.values()
            )
        ):
            raise ValueError(
                f"reviewed label {index} requires unanimous per-reviewer labels"
            )
        normalized_id = case_id.strip()
        if normalized_id in labels:
            raise ValueError(f"duplicate reviewed label: {normalized_id}")
        labels[normalized_id] = expected
    expected_ids = {item.case_id for item in contexts}
    if set(labels) != expected_ids:
        missing = len(expected_ids - set(labels))
        extra = len(set(labels) - expected_ids)
        raise ValueError(f"reviewed labels require exact coverage; missing={missing}, extra={extra}")
    return tuple(
        LabeledContext(
            case_id=item.case_id,
            question=item.question,
            context=item.context,
            expected_sufficient=labels[item.case_id],
        )
        for item in contexts
    )


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
    release_labels_reviewed: bool = False,
    minimum_class_count: int = 20,
) -> dict[str, Any]:
    if not results:
        raise ValueError("at least one evaluation result is required")
    if not 0 <= false_sufficient_max <= 1 or not 0 <= sufficient_recall_min <= 1:
        raise ValueError("gate thresholds must be between zero and one")
    if isinstance(minimum_class_count, bool) or minimum_class_count < 1:
        raise ValueError("minimum_class_count must be positive")
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
    if not release_labels_reviewed:
        reasons.append("proxy_labels_not_release_evidence")
    if sufficient_total < minimum_class_count:
        reasons.append("underpowered_sufficient_class")
    if insufficient_total < minimum_class_count:
        reasons.append("underpowered_insufficient_class")
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
            "minimum_class_count": minimum_class_count,
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


def release_overrides(
    *,
    false_sufficient_max: float,
    sufficient_recall_min: float,
    per_class: int,
    minimum_reviewed_per_class: int,
    reviewed_labels_present: bool,
) -> tuple[str, ...]:
    """Name every deviation that makes this run easier to pass than the release one.

    The knobs exist because diagnosing a failing autorater needs small, cheap runs.
    The danger is that a diagnostic run emits the same schema, the same
    ``gate.passed`` field, and the same exit code as a release run, so six months
    later nobody can tell which is which. Tightening a threshold is fine; only
    loosening is recorded, because only loosening can manufacture a pass.
    """
    overrides: list[str] = []
    if false_sufficient_max > DEFAULT_FALSE_SUFFICIENT_MAX:
        overrides.append("relaxed_false_sufficient_max")
    if sufficient_recall_min < DEFAULT_SUFFICIENT_RECALL_MIN:
        overrides.append("relaxed_sufficient_recall_min")
    if per_class < DEFAULT_PER_CLASS:
        overrides.append("reduced_per_class")
    if minimum_reviewed_per_class < RELEASE_MINIMUM_REVIEWED_PER_CLASS:
        overrides.append("reduced_minimum_reviewed_per_class")
    if not reviewed_labels_present:
        overrides.append("proxy_labels_not_release_evidence")
    return tuple(overrides)


def demote_non_release_run(
    report: dict[str, Any],
    *,
    overrides: tuple[str, ...],
    dirty: bool,
) -> dict[str, Any]:
    """Stamp the run class and refuse a passing gate for anything unreproducible.

    A dirty tree cannot be reproduced by anyone,
    including us, so it is never evidence regardless of how good the numbers look.
    """
    reasons = list(report["gate"].get("reasons", ()))
    for override in overrides:
        if override not in reasons:
            reasons.append(override)
    if dirty and "evaluation_source_dirty" not in reasons:
        reasons.append("evaluation_source_dirty")
    disqualified = bool(overrides) or dirty
    report["gate"] = {
        **report["gate"],
        "reasons": reasons,
        "passed": report["gate"]["passed"] and not disqualified,
    }
    report["run_class"] = "diagnostic" if disqualified else "release"
    return report


def reasoning_options_for_provider(
    settings: LlmSettings,
    provider_key: str,
    model_id: str,
) -> tuple[bool | None, ReasoningParameter | None]:
    """Keep persisted model capabilities scoped to their selected provider."""
    if provider_key != settings.provider_key or model_id != settings.model_id:
        return None, None
    return settings.effective_reasoning_enabled, settings.model_reasoning_parameter


def _build_remote_assessor(
    model: str | None,
    provider_key: str | None,
    prompt_variant: SufficiencyPromptVariant | str,
    max_tokens: int = 96,
) -> SufficientContextAutorater:
    settings = load_settings()
    if settings.backend != "remote":
        raise RuntimeError("UIT-ViQuAD release evaluation requires a configured remote model")
    provider = get_provider(provider_key or settings.provider_key)
    key = SecretStore(dotenv_path=REPO_ROOT / ".env").get_key(provider.key)
    if not key:
        raise RuntimeError(f"missing API key for {provider.key}")
    if provider.key != settings.provider_key and model is None:
        raise RuntimeError("an explicit model is required when overriding the provider")
    model_id = model or settings.model_id
    reasoning_enabled, reasoning_parameter = reasoning_options_for_provider(
        settings,
        provider.key,
        model_id,
    )
    engine = RemoteOpenAILLM(
        provider,
        model_id,
        key,
        reasoning_enabled=reasoning_enabled,
        reasoning_parameter=reasoning_parameter,
        max_output_tokens=settings.effective_max_tokens,
    )
    if not isinstance(engine, StructuredLLMEngine):
        raise RuntimeError("configured remote model does not support structured output")
    return SufficientContextAutorater(
        engine,
        model_id=model_id,
        prompt_variant=prompt_variant,
        max_tokens=max_tokens,
    )


def _write_private_results(path: Path, results: Sequence[CaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for result in results:
            output.write(json.dumps(result.as_private_dict(), ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument(
        "--prompt-variant",
        choices=tuple(item.value for item in SufficiencyPromptVariant),
        default=SufficiencyPromptVariant.PAPER_DEFINITION.value,
    )
    parser.add_argument("--reviewed-labels", type=Path)
    parser.add_argument("--autorater-max-tokens", type=int, default=96)
    parser.add_argument("--minimum-reviewed-per-class", type=int, default=20)
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
    reviewed_manifest_sha256: str | None = None
    if args.reviewed_labels is not None:
        raw_manifest = args.reviewed_labels.read_bytes()
        reviewed_manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
        payload = json.loads(raw_manifest)
        if not isinstance(payload, dict):
            raise ValueError("reviewed label manifest must be an object")
        selected = apply_reviewed_labels(selected, payload)
    assessor = _build_remote_assessor(
        args.model,
        args.provider,
        args.prompt_variant,
        args.autorater_max_tokens,
    )
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
        release_labels_reviewed=args.reviewed_labels is not None,
        minimum_class_count=args.minimum_reviewed_per_class,
    )
    inventory = Counter(item.expected_sufficient for item in all_contexts)
    overrides = release_overrides(
        false_sufficient_max=args.false_sufficient_max,
        sufficient_recall_min=args.sufficient_recall_min,
        per_class=args.per_class,
        minimum_reviewed_per_class=args.minimum_reviewed_per_class,
        reviewed_labels_present=args.reviewed_labels is not None,
    )
    artifact = make_eval_artifact_metadata(
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
                "provider": args.provider or load_settings().provider_key,
                "prompt_variant": assessor.prompt_variant.value,
                "autorater_max_tokens": assessor.max_tokens,
                "label_source": (
                    "reviewed_semantic_consensus"
                    if args.reviewed_labels is not None
                    else "uit_viquad_extractability_proxy"
                ),
                "reviewed_manifest_sha256": reviewed_manifest_sha256,
                "minimum_reviewed_per_class": args.minimum_reviewed_per_class,
            },
            ignored_untracked_paths=(args.raw_output, args.output),
        )
    report = {
        "schema_version": "soca-sufficient-context-viquad-v2",
        "artifact": artifact.to_dict(),
        "dataset": {
            "repo": DATASET_REPO,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "row_count": len(all_contexts),
            "answerable_count": inventory[True],
            "unanswerable_count": inventory[False],
        },
        "sample": {"seed": args.seed, "per_class": args.per_class},
        "release_overrides": list(overrides),
        "aggregate": aggregate,
    }
    demote_non_release_run(
        report["aggregate"],
        overrides=overrides,
        dirty=artifact.source_dirty,
    )
    report["run_class"] = report["aggregate"]["run_class"]
    write_json_artifact(args.output, report)
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    return int(not report["aggregate"]["gate"]["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
