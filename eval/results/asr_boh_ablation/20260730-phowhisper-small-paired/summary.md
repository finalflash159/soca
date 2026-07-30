# Paired production ASR / experimental BoH ablation

- Model: `huuquyet/PhoWhisper-small` (`phowhisper_small`)
- Runtime: ONNX Runtime, CoreML + CPU provider
- Data: 30 FLEURS Vietnamese test utterances; 50 ESC-50/synthetic noise rows
- Method: production `RobustASR` runs once; BoH is applied to the identical
  saved output
- Source commit: `586412ac5ae9d603d40498f011bceef39d6b007b`
- Source state: clean

| Variant | WER | CER | False reject | Hallucination | Catch rate | p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| production, no BoH | 16.39% | 7.34% | 0.0% | 10.0% | 90.0% | 31 / 9,292 ms |
| production + experimental BoH | 16.39% | 7.34% | 0.0% | 10.0% | 90.0% | 31 / 9,292 ms |

BoH matched no item and changed no prediction. The five leaked noise rows were
all speech-like; all 45 pure-noise rows were rejected. Production remains
no-BoH. This run does not claim AEC coverage because these files do not contain
the paired far-end reference required by an echo canceller.
