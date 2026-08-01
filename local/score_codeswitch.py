"""Score predictions on the code-switch set: overall WER + English-term recall.

    uv run python local/score_codeswitch.py data/asr_codeswitch/preds/*.json

Metric definitions (recorded here so the bake-off is reproducible):
  - wer:       word error rate over the whole sentence, after NFC/lowercase/
               punctuation-stripped normalization.
  - en_recall: fraction of reference English words recognized CORRECTLY,
               computed as whether that word falls in an 'equal' chunk of the
               jiwer alignment. Insertions are not counted — this is "percent
               of terms heard correctly", the least ambiguous framing.
  - cs_wer:    1 - en_recall, to compare directly against literature numbers.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from jiwer import process_words
from rich.console import Console
from rich.table import Table

from local.codeswitch_text import normalize

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "data" / "asr_codeswitch" / "manifest.jsonl"
console = Console()


def correct_reference_indices(reference: str, hypothesis: str) -> set[int]:
    """Reference token positions that were recognized correctly (in an 'equal' chunk)."""
    if not reference.strip() or not hypothesis.strip():
        return set()
    output = process_words(reference, hypothesis)
    correct: set[int] = set()
    for sentence_alignment in output.alignments:
        for chunk in sentence_alignment:
            if chunk.type == "equal":
                correct.update(range(chunk.ref_start_idx, chunk.ref_end_idx))
    return correct


def score_system(rows: list[dict], predictions: dict[str, str]) -> dict:
    total_words = 0
    total_errors = 0.0
    en_total = 0
    en_correct = 0
    misses: dict[str, int] = defaultdict(int)

    for row in rows:
        reference = normalize(row["reference"])
        hypothesis = normalize(predictions.get(row["id"], ""))
        ref_tokens = reference.split()

        output = process_words(reference, hypothesis)
        total_errors += output.substitutions + output.deletions + output.insertions
        total_words += len(ref_tokens)

        correct = correct_reference_indices(reference, hypothesis)
        for idx in row["english_indices"]:
            if idx >= len(ref_tokens):
                continue
            en_total += 1
            if idx in correct:
                en_correct += 1
            else:
                misses[ref_tokens[idx]] += 1

    en_recall = en_correct / en_total if en_total else 0.0
    return {
        "wer": total_errors / total_words if total_words else 0.0,
        "en_recall": en_recall,
        "cs_wer": 1.0 - en_recall,
        "en_total": en_total,
        "en_correct": en_correct,
        "num_utterances": len(rows),
        "top_misses": sorted(misses.items(), key=lambda kv: -kv[1])[:15],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("preds", nargs="+", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "eval" / "results" / "asr_codeswitch_bakeoff.json",
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    report: dict[str, dict] = {}
    for path in args.preds:
        payload = json.loads(path.read_text(encoding="utf-8"))
        system = payload["system"]
        report[system] = score_system(rows, payload["predictions"])
        report[system]["source"] = str(path)

    table = Table(title=f"Code-switch bake-off ({len(rows)} sentences)")
    table.add_column("system")
    table.add_column("overall WER", justify="right")
    table.add_column("CS-WER", justify="right")
    table.add_column("EN words correct", justify="right")
    for system, stats in report.items():
        table.add_row(
            system,
            f"{stats['wer']*100:.2f}%",
            f"{stats['cs_wer']*100:.2f}%",
            f"{stats['en_correct']}/{stats['en_total']}",
        )
    console.print(table)

    for system, stats in report.items():
        if stats["top_misses"]:
            console.print(
                f"\n[bold]{system}[/bold] most-missed: "
                + ", ".join(f"{w}x{n}" for w, n in stats["top_misses"])
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
