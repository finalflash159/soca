from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .corpora import CorpusItem
from .scoring import MatchMode


@dataclass(frozen=True)
class SynthRecord:
    item_id: str
    corpus: str
    text_in: str
    expected: str
    mode: MatchMode
    wav_path: str
    sample_rate: int
    tts_latency_ms: float
    audio_duration_ms: float


@dataclass(frozen=True)
class SynthManifest:
    engine: str
    voice: str
    records: tuple[SynthRecord, ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "engine": self.engine,
                "voice": self.voice,
                "records": [asdict(record) for record in self.records],
            },
            ensure_ascii=False,
            indent=2,
        )


def write_manifest(manifest: SynthManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_json(), encoding="utf-8")


def read_manifest(path: Path) -> SynthManifest:
    """Load a manifest written by the synth stage.

    The two stages run in different interpreters (the ASR side may live in a
    separate venv), so the manifest is the only contract between them and is
    validated rather than trusted.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read synth manifest at {path}: {exc}") from exc

    if not isinstance(raw, dict) or "records" not in raw:
        raise ValueError(f"Synth manifest at {path} has no 'records' list")

    records: list[SynthRecord] = []
    for entry in raw["records"]:
        try:
            records.append(SynthRecord(**entry))
        except TypeError as exc:
            raise ValueError(f"Malformed record in {path}: {exc}") from exc

    return SynthManifest(
        engine=str(raw.get("engine", "")),
        voice=str(raw.get("voice", "")),
        records=tuple(records),
    )


def record_from_item(
    item: CorpusItem,
    *,
    wav_path: Path,
    sample_rate: int,
    tts_latency_ms: float,
    audio_duration_ms: float,
) -> SynthRecord:
    return SynthRecord(
        item_id=item.item_id,
        corpus=item.corpus,
        text_in=item.text_in,
        expected=item.expected,
        mode=item.mode,
        wav_path=str(wav_path),
        sample_rate=sample_rate,
        tts_latency_ms=tts_latency_ms,
        audio_duration_ms=audio_duration_ms,
    )


__all__ = [
    "SynthManifest",
    "SynthRecord",
    "read_manifest",
    "record_from_item",
    "write_manifest",
]
