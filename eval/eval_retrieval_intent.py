from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IntentCase:
    case_id: str
    split: str
    text: str
    expected: bool


@dataclass(frozen=True)
class IntentSignal:
    split: str
    expected: bool
    has_hits: bool
    question_marker: bool
    dense_score: float | None
    retrieval_ms: float


def load_cases(path: Path) -> tuple[IntentCase, ...]:
    cases: list[IntentCase] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no} must be an object")
            case_id, split, text, expected = (
                payload.get("id"),
                payload.get("split"),
                payload.get("text"),
                payload.get("expected"),
            )
            if (
                not isinstance(case_id, str)
                or not case_id.strip()
                or case_id in seen
                or split not in {"train", "validation"}
                or not isinstance(text, str)
                or not text.strip()
                or not isinstance(expected, bool)
            ):
                raise ValueError(f"{path}:{line_no} has invalid intent case")
            seen.add(case_id)
            cases.append(IntentCase(case_id, split, text, expected))
    if not cases or {case.split for case in cases} != {"train", "validation"}:
        raise ValueError("intent eval needs non-empty train and validation splits")
    return tuple(cases)


def predict(signal: IntentSignal, threshold: float) -> bool:
    if not signal.has_hits:
        return False
    return signal.dense_score is not None and signal.dense_score >= threshold


def classification_metrics(
    signals: Sequence[IntentSignal], threshold: float
) -> dict[str, float | int]:
    predicted = tuple(predict(signal, threshold) for signal in signals)
    tp = sum(value and signal.expected for value, signal in zip(predicted, signals, strict=True))
    fp = sum(
        value and not signal.expected for value, signal in zip(predicted, signals, strict=True)
    )
    fn = sum(
        not value and signal.expected for value, signal in zip(predicted, signals, strict=True)
    )
    tn = sum(
        not value and not signal.expected for value, signal in zip(predicted, signals, strict=True)
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "count": len(signals),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def choose_threshold(train: Sequence[IntentSignal]) -> float:
    candidates = sorted(
        {
            0.0,
            1.0,
            *(
                min(1.0, max(0.0, signal.dense_score))
                for signal in train
                if signal.dense_score is not None
            ),
        }
    )
    return max(
        candidates,
        key=lambda threshold: (
            classification_metrics(train, threshold)["f1"],
            -classification_metrics(train, threshold)["false_positive_rate"],
            threshold,
        ),
    )


def _latency_summary(signals: Sequence[IntentSignal]) -> dict[str, float]:
    ordered = sorted(signal.retrieval_ms for signal in signals)
    if not ordered:
        return {"mean_ms": 0.0, "p95_ms": 0.0}
    p95 = ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)]
    return {"mean_ms": sum(ordered) / len(ordered), "p95_ms": p95}


def evaluate_signals(signals: Sequence[IntentSignal]) -> dict[str, object]:
    train = tuple(signal for signal in signals if signal.split == "train")
    validation = tuple(signal for signal in signals if signal.split == "validation")
    threshold = choose_threshold(train)
    return {
        "threshold": threshold,
        "train": classification_metrics(train, threshold),
        "validation": classification_metrics(validation, threshold),
        "retrieval_latency": {
            "train": _latency_summary(train),
            "validation": _latency_summary(validation),
        },
    }


def collect_signals(source, cases: Sequence[IntentCase], *, limit: int) -> tuple[IntentSignal, ...]:
    signals: list[IntentSignal] = []
    for case in cases:
        started = time.perf_counter()
        batch = source.retrieve(case.text, limit=limit)
        signals.append(
            IntentSignal(
                split=case.split,
                expected=case.expected,
                has_hits=bool(batch.hits),
                question_marker=False,
                dense_score=batch.max_dense_score,
                retrieval_ms=(time.perf_counter() - started) * 1000,
            )
        )
    return tuple(signals)


def run_eval(
    *, vault: Path, cases: Sequence[IntentCase], backend: str, index_home: Path | None, limit: int
) -> dict[str, object]:
    from eval.eval_hybrid_retrieval import build_embedding_model
    from soca.knowledge.hybrid_source import HybridConfig, HybridKnowledgeSource

    source = HybridKnowledgeSource(
        vault,
        model=build_embedding_model(backend),
        index_home=index_home,
        include_globs=("wiki/**/*.md",),
        config=HybridConfig(dense_failure_policy="raise"),
    )
    report = evaluate_signals(collect_signals(source, cases, limit=limit))
    report.update({"status": "ok", "backend": backend})
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval intent gating.")
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--backend", choices=("fastembed", "model2vec"), default="fastembed")
    parser.add_argument("--index-home", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_eval(
        vault=args.vault,
        cases=load_cases(args.cases),
        backend=args.backend,
        index_home=args.index_home,
        limit=args.limit,
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
