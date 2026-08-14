"""Evaluate the shipped Smart Turn ONNX on pinned English/Vietnamese test rows."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import click
import numpy as np

from eval.provenance import run_provenance
from soca.core.smart_turn import SmartTurnDetector

DATASET_ID = "pipecat-ai/smart-turn-data-v3.2-test"
DATASET_REVISION = "0500378e8ed6d38e37b016e24d261e8e6c6a6859"
DATASET_SPLIT = "train"
MODEL_REPO_ID = "pipecat-ai/smart-turn-v3"
MODEL_REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
MODEL_FILENAME = "smart-turn-v3.2-cpu.onnx"
MODEL_SHA256 = "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f"
EXPECTED_LANGUAGE_COUNTS = {"eng": 7820, "vie": 1004}
SAMPLE_RATE = 16000


class ArtifactIdentityError(RuntimeError):
    """A pinned model artifact does not match the benchmark contract."""


class DatasetContractError(RuntimeError):
    """The pinned dataset does not expose the expected rows or schema."""


class CompleteProbabilityDetector(Protocol):
    def p_complete_batch(self, audio_windows: Sequence[np.ndarray]) -> np.ndarray: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_identity(path: Path, *, expected_sha256: str = MODEL_SHA256) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactIdentityError(f"Smart Turn model is missing: {path}")
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ArtifactIdentityError(
            f"Smart Turn model SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return {
        "repo_id": MODEL_REPO_ID,
        "revision": MODEL_REVISION,
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": actual,
    }


@dataclass
class ConfusionCounts:
    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def update(self, *, label: bool, probability: float) -> None:
        predicted = probability > 0.5
        if predicted and label:
            self.true_positive += 1
        elif predicted:
            self.false_positive += 1
        elif label:
            self.false_negative += 1
        else:
            self.true_negative += 1

    def merge(self, other: ConfusionCounts) -> None:
        self.true_positive += other.true_positive
        self.true_negative += other.true_negative
        self.false_positive += other.false_positive
        self.false_negative += other.false_negative

    def metrics(self) -> dict[str, float | int]:
        total = (
            self.true_positive
            + self.true_negative
            + self.false_positive
            + self.false_negative
        )
        precision_denominator = self.true_positive + self.false_positive
        recall_denominator = self.true_positive + self.false_negative
        precision = self.true_positive / precision_denominator if precision_denominator else 0.0
        recall = self.true_positive / recall_denominator if recall_denominator else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "sample_count": total,
            "accuracy": (
                (self.true_positive + self.true_negative) / total * 100 if total else 0.0
            ),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            # Match the upstream Smart Turn benchmark: FP/FN are percentages of
            # every sample, rather than conditional class error rates.
            "false_positive_rate": self.false_positive / total * 100 if total else 0.0,
            "false_negative_rate": self.false_negative / total * 100 if total else 0.0,
            "true_positive": self.true_positive,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
        }


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def audio_array(value: Any) -> np.ndarray:
    """Decode an HF audio mapping or the datasets/torchcodec lazy decoder."""
    if isinstance(value, Mapping):
        samples = value.get("array")
        sample_rate = value.get("sampling_rate")
    elif hasattr(value, "get_all_samples"):
        decoded = value.get_all_samples()
        samples = decoded.data
        sample_rate = decoded.sample_rate
    else:
        raise TypeError(f"unsupported audio value: {type(value)!r}")

    if int(sample_rate or 0) != SAMPLE_RATE:
        raise ValueError(f"Smart Turn requires 16 kHz audio, got {sample_rate!r}")
    array = _numpy(samples).astype(np.float32, copy=False)
    if array.ndim == 2:
        channel_axis = 0 if array.shape[0] <= 8 else 1
        array = array.mean(axis=channel_axis)
    if array.ndim != 1:
        raise ValueError(f"Smart Turn audio must be mono, got shape {array.shape}")
    return np.ascontiguousarray(array, dtype=np.float32)


def _flush_batch(
    batch: list[Mapping[str, Any]],
    *,
    detector: CompleteProbabilityDetector,
    overall: ConfusionCounts,
    per_language: defaultdict[str, ConfusionCounts],
    per_dataset: defaultdict[str, ConfusionCounts],
) -> None:
    probabilities = detector.p_complete_batch([audio_array(row["audio"]) for row in batch])
    if len(probabilities) != len(batch):
        raise RuntimeError(
            f"prediction count mismatch: expected {len(batch)}, got {len(probabilities)}"
        )
    for row, probability in zip(batch, probabilities, strict=True):
        language = str(row["language"])
        dataset = str(row.get("dataset", "unknown"))
        label = bool(row["endpoint_bool"])
        value = float(probability)
        overall.update(label=label, probability=value)
        per_language[language].update(label=label, probability=value)
        per_dataset[dataset].update(label=label, probability=value)


def evaluate_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    detector: CompleteProbabilityDetector,
    batch_size: int = 32,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    overall = ConfusionCounts()
    per_language: defaultdict[str, ConfusionCounts] = defaultdict(ConfusionCounts)
    per_dataset: defaultdict[str, ConfusionCounts] = defaultdict(ConfusionCounts)
    batch: list[Mapping[str, Any]] = []
    batch_count = 0
    started = time.perf_counter()
    for row in rows:
        required = {"audio", "endpoint_bool", "language"}
        if missing := required.difference(row):
            raise DatasetContractError(f"Smart Turn row is missing: {', '.join(sorted(missing))}")
        batch.append(row)
        if len(batch) == batch_size:
            _flush_batch(
                batch,
                detector=detector,
                overall=overall,
                per_language=per_language,
                per_dataset=per_dataset,
            )
            batch_count += 1
            batch = []
    if batch:
        _flush_batch(
            batch,
            detector=detector,
            overall=overall,
            per_language=per_language,
            per_dataset=per_dataset,
        )
        batch_count += 1
    elapsed = time.perf_counter() - started
    total = int(overall.metrics()["sample_count"])
    return {
        "overall": overall.metrics(),
        "per_language": {
            key: counts.metrics() for key, counts in sorted(per_language.items())
        },
        "per_dataset": {
            key: counts.metrics() for key, counts in sorted(per_dataset.items())
        },
        "timing": {
            "elapsed_seconds": elapsed,
            "batch_count": batch_count,
            "batch_size": batch_size,
            "samples_per_second": total / elapsed if elapsed else None,
        },
    }


def _load_rows(languages: tuple[str, ...], limit_per_language: int | None) -> Any:
    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        split=DATASET_SPLIT,
    )
    selected = dataset.filter(
        lambda values: [value in languages for value in values],
        input_columns=["language"],
        batched=True,
        desc=f"Select Smart Turn languages: {', '.join(languages)}",
    )
    counts = Counter(str(value) for value in selected["language"])
    if limit_per_language is None:
        expected = {language: EXPECTED_LANGUAGE_COUNTS[language] for language in languages}
        if counts != expected:
            raise DatasetContractError(
                f"pinned Smart Turn language counts changed: expected {expected}, got {dict(counts)}"
            )
        return selected

    remaining = dict.fromkeys(languages, limit_per_language)
    indices: list[int] = []
    for index, language in enumerate(selected["language"]):
        key = str(language)
        if remaining[key] > 0:
            indices.append(index)
            remaining[key] -= 1
    if any(remaining.values()):
        raise DatasetContractError(f"not enough rows for requested limits: {remaining}")
    return selected.select(indices)


def _rows(dataset: Any) -> Iterable[Mapping[str, Any]]:
    for index in range(len(dataset)):
        yield dataset[index]


def _runtime_identity() -> dict[str, Any]:
    import datasets
    import onnxruntime
    import transformers

    return {
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "datasets": datasets.__version__,
            "transformers": transformers.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
    }


@click.command()
@click.option(
    "--model-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("models/smart-turn-v3-onnx"),
    show_default=True,
)
@click.option("--language", "languages", multiple=True, default=("vie", "eng"))
@click.option("--batch-size", type=click.IntRange(min=1), default=32, show_default=True)
@click.option("--limit-per-language", type=click.IntRange(min=1), default=None)
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("eval/results/smart_turn_languages.json"),
    show_default=True,
)
def main(
    model_dir: Path,
    languages: tuple[str, ...],
    batch_size: int,
    limit_per_language: int | None,
    output: Path,
) -> None:
    languages = tuple(dict.fromkeys(languages))
    if not languages or any(language not in EXPECTED_LANGUAGE_COUNTS for language in languages):
        raise click.ClickException("--language must contain only vie and/or eng")
    model_path = model_dir / MODEL_FILENAME
    try:
        model_identity = verify_model_identity(model_path)
        dataset = _load_rows(languages, limit_per_language)
        detector = SmartTurnDetector(model_dir, providers=["CPUExecutionProvider"])
        detector.warmup()
        with click.progressbar(
            _rows(dataset),
            length=len(dataset),
            label="Smart Turn language inference",
        ) as rows:
            report = evaluate_rows(rows, detector=detector, batch_size=batch_size)
    except (ArtifactIdentityError, DatasetContractError, OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    language_metrics = report["per_language"]
    gap = None
    if {"vie", "eng"}.issubset(language_metrics):
        gap = language_metrics["eng"]["accuracy"] - language_metrics["vie"]["accuracy"]
    payload = {
        "metadata": run_provenance(
            provider="CPUExecutionProvider",
            characterization=limit_per_language is not None,
            **_runtime_identity(),
        ),
        "model": model_identity,
        "dataset": {
            "repo_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "languages": list(languages),
            "expected_language_counts": {
                language: EXPECTED_LANGUAGE_COUNTS[language] for language in languages
            },
            "limit_per_language": limit_per_language,
        },
        "decision": {
            "english_minus_vietnamese_accuracy_pp": gap,
            "vietnamese_model_gap": "needs_local_full_run" if limit_per_language else (
                "confirmed" if gap is not None and gap >= 5.0 else "not_confirmed"
            ),
            "diagnostic_gate": "English accuracy exceeds Vietnamese by >=5 pp",
            "fine_tune_disposition": "deferred_by_product_owner",
        },
        **report,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    click.echo(json.dumps(payload["decision"], ensure_ascii=False))
    click.echo(f"Saved {output}")


if __name__ == "__main__":
    main()
