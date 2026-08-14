"""Pair UTMOSv2 naturalness with TTS→ASR WER for Vietnamese TTS evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from eval.result_io import make_eval_artifact_metadata, write_json_artifact

REPO_ROOT = Path(__file__).resolve().parents[1]
UTMOSV2_REPOSITORY = "https://github.com/sarulab-speech/UTMOSv2"
UTMOSV2_REVISION = "cc2700db57bb83ee13dc31ebe1b868c254e15d09"
DEFAULT_OUTPUT = REPO_ROOT / "eval" / "results" / "tts_quality.json"
DEFAULT_RAW_OUTPUT = REPO_ROOT / "artifacts" / "local" / "tts_quality_rows.json"


class MOSModelError(RuntimeError):
    pass


class MOSScorer(Protocol):
    revision: str

    def score(self, path: Path) -> float: ...


@dataclass(frozen=True)
class AudioItem:
    group: str
    item_id: str
    text_sha256: str
    wav_path: Path
    wav_sha256: str


@dataclass(frozen=True)
class MOSRow:
    group: str
    item_id: str
    text_sha256: str
    wav_sha256: str
    predicted_mos: float
    model_revision: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_audio_group(group: str, manifest_path: Path) -> tuple[AudioItem, ...]:
    normalized_group = group.strip()
    if not normalized_group:
        raise ValueError("audio group name is required")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError(f"{manifest_path}: records must be a non-empty list")
    items: list[AudioItem] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{manifest_path}: record {index} is not an object")
        item_id = record.get("item_id")
        text = record.get("text_in")
        raw_wav_path = record.get("wav_path")
        if (
            not isinstance(item_id, str)
            or not item_id.strip()
            or not isinstance(text, str)
            or not text.strip()
            or not isinstance(raw_wav_path, str)
            or not raw_wav_path.strip()
        ):
            raise ValueError(f"{manifest_path}: invalid record {index}")
        normalized_id = item_id.strip()
        if normalized_id in seen:
            raise ValueError(f"{manifest_path}: duplicate item id {normalized_id}")
        seen.add(normalized_id)
        wav_path = Path(raw_wav_path).expanduser()
        if not wav_path.is_absolute():
            wav_path = manifest_path.parent / wav_path
        if not wav_path.is_file():
            raise ValueError(f"{manifest_path}: missing WAV {wav_path}")
        items.append(
            AudioItem(
                group=normalized_group,
                item_id=normalized_id,
                text_sha256=hashlib.sha256(text.strip().encode()).hexdigest(),
                wav_path=wav_path.resolve(),
                wav_sha256=_sha256(wav_path),
            )
        )
    return tuple(items)


class UTMOSV2Scorer:
    revision = UTMOSV2_REVISION

    def __init__(self) -> None:
        try:
            import utmosv2
        except ImportError as exc:
            raise MOSModelError("model_not_installed") from exc
        try:
            self._model = utmosv2.create_model(pretrained=True)
        except Exception as exc:  # noqa: BLE001 - external model boundary
            raise MOSModelError("model_load_failed") from exc

    def score(self, path: Path) -> float:
        try:
            raw = self._model.predict(input_path=str(path))
            value = float(raw)
        except Exception as exc:  # noqa: BLE001 - external model boundary
            raise MOSModelError("prediction_failed") from exc
        if not math.isfinite(value) or not 1.0 <= value <= 5.0:
            raise MOSModelError("invalid_prediction")
        return value


def evaluate_groups(
    groups: Mapping[str, Sequence[AudioItem]],
    scorer: MOSScorer,
) -> tuple[MOSRow, ...]:
    rows: list[MOSRow] = []
    for group in sorted(groups):
        for item in groups[group]:
            try:
                score = float(scorer.score(item.wav_path))
            except MOSModelError:
                raise
            except Exception as exc:  # noqa: BLE001 - scorer protocol boundary
                raise MOSModelError("prediction_failed") from exc
            if not math.isfinite(score) or not 1.0 <= score <= 5.0:
                raise MOSModelError("invalid_prediction")
            rows.append(
                MOSRow(
                    group=group,
                    item_id=item.item_id,
                    text_sha256=item.text_sha256,
                    wav_sha256=item.wav_sha256,
                    predicted_mos=score,
                    model_revision=scorer.revision,
                )
            )
    if not rows:
        raise ValueError("at least one audio item is required")
    return tuple(rows)


def summarize_groups(
    rows: Sequence[MOSRow],
    *,
    reference_group: str,
    wer: Mapping[str, float],
) -> dict[str, Any]:
    grouped: dict[str, list[MOSRow]] = {}
    for row in rows:
        grouped.setdefault(row.group, []).append(row)
    reasons: list[str] = []
    reference_rows = grouped.get(reference_group)
    if not reference_rows:
        reasons.append("reference_group_missing")
    reference_by_id = (
        {row.item_id: row for row in reference_rows} if reference_rows is not None else {}
    )
    group_summaries: dict[str, dict[str, Any]] = {}
    for group, items in sorted(grouped.items()):
        scores = [row.predicted_mos for row in items]
        paired_deltas = [
            row.predicted_mos - reference_by_id[row.item_id].predicted_mos
            for row in items
            if row.item_id in reference_by_id
            and row.text_sha256 == reference_by_id[row.item_id].text_sha256
        ]
        if group != reference_group and len(paired_deltas) != len(items):
            reasons.append(f"unpaired_items:{group}")
        group_wer = wer.get(group)
        if group != reference_group and group_wer is None:
            reasons.append(f"wer_missing:{group}")
        elif group_wer is not None and (
            not math.isfinite(group_wer) or not 0.0 <= group_wer
        ):
            reasons.append(f"wer_invalid:{group}")
        group_summaries[group] = {
            "count": len(items),
            "mean_mos": statistics.fmean(scores),
            "median_mos": statistics.median(scores),
            "mean_delta_vs_reference": (
                statistics.fmean(paired_deltas)
                if group != reference_group and paired_deltas
                else (0.0 if group == reference_group else None)
            ),
            "wer": group_wer,
        }
    return {
        "interpretation": "relative_vietnamese_indicator_not_absolute_mos",
        "reference_group": reference_group,
        "groups": group_summaries,
        "gate": {"passed": not reasons, "reasons": reasons},
    }


def load_wer_report(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "soca-tts-wer-v1":
        raise ValueError("unsupported TTS WER report")
    groups = payload.get("groups")
    if not isinstance(groups, dict):
        raise ValueError("TTS WER report has no groups")
    result: dict[str, float] = {}
    for label, value in groups.items():
        if (
            not isinstance(label, str)
            or not label
            or not isinstance(value, int | float)
            or isinstance(value, bool)
        ):
            raise ValueError("invalid TTS WER group")
        result[label] = float(value)
    return result


def _parse_group(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("group must use LABEL=MANIFEST")
    return label.strip(), Path(raw_path).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", type=_parse_group, required=True)
    parser.add_argument("--reference-group", default="human_reference")
    parser.add_argument("--wer-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifests = dict(args.group)
    if len(manifests) != len(args.group):
        raise ValueError("duplicate audio group label")
    groups = {label: load_audio_group(label, manifest) for label, manifest in manifests.items()}
    try:
        scorer = UTMOSV2Scorer()
        rows = evaluate_groups(groups, scorer)
    except MOSModelError as exc:
        report = {
            "schema_version": "soca-tts-quality-v1",
            "status": "blocked",
            "reason": str(exc),
            "artifact": make_eval_artifact_metadata(
                suite="tts_quality",
                run_type="benchmark",
                data_files=tuple(manifests.values()) + (args.wer_report,),
                config={
                    "utmosv2_repository": UTMOSV2_REPOSITORY,
                    "utmosv2_revision": UTMOSV2_REVISION,
                    "reference_group": args.reference_group,
                    "groups": sorted(groups),
                },
                ignored_untracked_paths=(args.raw_output, args.output),
            ).to_dict(),
            "summary": {
                "interpretation": "relative_vietnamese_indicator_not_absolute_mos",
                "gate": {"passed": False, "status": "blocked", "reasons": [str(exc)]},
            },
        }
        write_json_artifact(args.output, report)
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return 2
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(
        json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    wer = load_wer_report(args.wer_report)
    summary = summarize_groups(rows, reference_group=args.reference_group, wer=wer)
    audio_paths = tuple(item.wav_path for items in groups.values() for item in items)
    report = {
        "schema_version": "soca-tts-quality-v1",
        "artifact": make_eval_artifact_metadata(
            suite="tts_quality",
            run_type="benchmark",
            data_files=tuple(manifests.values()) + (args.wer_report,) + audio_paths,
            config={
                "utmosv2_repository": UTMOSV2_REPOSITORY,
                "utmosv2_revision": UTMOSV2_REVISION,
                "reference_group": args.reference_group,
                "groups": sorted(groups),
            },
            ignored_untracked_paths=(args.raw_output, args.output),
        ).to_dict(),
        "summary": summary,
    }
    write_json_artifact(args.output, report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(not summary["gate"]["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
