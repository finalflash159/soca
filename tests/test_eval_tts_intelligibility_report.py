from __future__ import annotations

from types import SimpleNamespace

from eval.eval_tts_intelligibility import build_wer_report
from eval.tts_intelligibility.manifest import SynthManifest
from eval.tts_intelligibility.scoring import CorpusSummary


def test_build_wer_report_exposes_group_metric_and_corpus_detail() -> None:
    report = build_wer_report(
        group_label="valtec",
        verdicts=[SimpleNamespace(wer=0.1), SimpleNamespace(wer=0.3)],
        summaries={
            "control": CorpusSummary(
                corpus="control",
                total=2,
                passed=1,
                mean_wer=0.2,
            )
        },
        manifest=SynthManifest(engine="valtec", voice="NF", records=()),
        asr_label="phowhisper_small",
    )

    assert report["schema_version"] == "soca-tts-wer-v1"
    assert report["groups"] == {"valtec": 0.2}
    assert report["details"]["valtec"]["corpora"]["control"]["pass_rate"] == 0.5
