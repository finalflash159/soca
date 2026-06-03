# 04 — ASR Robustness (`RobustASR`)

PhoWhisper, like Whisper-style models in general, can sometimes **hallucinate
text** on silence/noise or **loop indefinitely**. `RobustASR`
(`asr/robust_asr.py`) wraps the model with a **five-stage pipeline** that turns
raw output into a trusted transcript, or rejects cleanly with a clear technical
reason.

## Five-Stage Pipeline

```mermaid
flowchart TD
    A([audio 16kHz mono]) --> S1[Stage 1 · VAD<br/>vad.detect]
    S1 -->|no speech| R1[reject: no_speech]
    S1 -->|speech| S2[Stage 2 · ASR<br/>PhoWhisper on speech-only audio]
    S2 --> E{raw_text empty?}
    E -->|yes| R2[reject: empty_asr]
    E -->|no| S2b[Stage 2b · Confidence guard]
    S2b -->|avg_logprob &lt; min| R3[reject: low_confidence]
    S2b -->|compression &gt; max| R4[reject: high_compression]
    S2b -->|ok| S3[Stage 3 · De-loop<br/>remove_consecutive_repeats]
    S3 --> S4[Stage 4 · BoH match<br/>Aho-Corasick removes known hallucination phrases]
    S4 --> S5[Stage 5 · Heuristic safety net<br/>check_heuristics]
    S5 -->|hallucination| R5[reject: heuristic:...]
    S5 -->|empty after BoH| R6[reject: empty_after_boh]
    S5 -->|ok| OK([clean transcript])
```

## Role of Each Stage

| Stage                     | What It Does                                          | Why It Exists                                                                                       |
| ------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **1 · VAD**               | Detects real speech and trims silence                 | Whisper can hallucinate on silence; trimming also saves compute                                      |
| **2 · ASR**               | Runs PhoWhisper on **speech-only** audio              | More accurate and faster than decoding silence-heavy audio                                           |
| **2b · Confidence guard** | Blocks low `avg_logprob` and high `compression_ratio` | These are classic hallucination/loop signals; checks run on raw output for faithful diagnosis        |
| **3 · De-loop**           | Removes consecutive repeated spans                    | `"tôi tôi tôi tôi..."` becomes `"tôi"`                                                               |
| **4 · BoH**               | Removes model-specific hallucination phrases          | Each PhoWhisper variant can have its own bag of hallucinations artifact                              |
| **5 · Heuristics**        | Final safety net: chars/sec ratio, abnormal length... | Catches cases that pass the previous stages                                                          |

## Result & Trace (`RobustASRResult`)

The result keeps **intermediate text from every stage** for debugging:

```mermaid
classDiagram
    class RobustASRResult {
        +str text
        +str raw_text
        +str text_after_deloop
        +str text_after_boh
        +bool has_speech
        +bool was_looping
        +tuple boh_matches
        +HeuristicCheck heuristic
        +str rejection_reason
        +float avg_logprob
        +float compression_ratio
        +str boh_status
        +str confidence_guard_status
    }
```

- `text` is the final result; it is `""` when rejected.
- `raw_text` is the raw Whisper output.
- `text_after_deloop` and `text_after_boh` expose intermediate stages.
- `boh_matches` lists BoH phrases that were removed.

`rejection_reason`, empty when accepted, is a **technical code**, for example:
`no_speech`, `empty_asr`, `low_confidence:-0.90`, `high_compression:3.10`,
`heuristic:<name>`, or `empty_after_boh`.

> This code is **not spoken directly to the user**. It is input for the
> **repair layer**, which turns it into natural Vietnamese follow-up text. See
> [06](./06-conversation-repair.md).

## Model-Specific Configuration

- **Confidence guard** and **BoH** are keyed by model. `confidence_guard_status`
  and `boh_status` explain which artifact was loaded, or why it was skipped
  because the profile was missing or mismatched. This prevents accidentally
  applying a calibration artifact to the wrong model.
- If the BoH artifact is missing, Stage 4 is skipped safely; `boh_status` records
  the reason and the pipeline does not crash.

## Why Reject Reasons Are Separate from User-Facing Speech

| Layer        | Concern                                                        |
| ------------ | -------------------------------------------------------------- |
| `RobustASR`  | _Why_ the transcript cannot be trusted, technically            |
| Repair layer | _What to say_ to the user, in a natural UX tone                 |
| UI/Inspector | Show **both**: developer-facing reason and user-facing message |

This separation means dialogue copy can change without touching ASR, and a new
technical reason only needs to be mapped to a `RepairKind`.

## Related Files

- `asr/vad.py`, `asr/deloop.py`, `asr/boh.py`,
  `asr/hallucination_heuristics.py`, `asr/whisper_onnx.py`, `asr/registry.py`.
- Threshold calibration: `soca calibrate-asr` / `soca benchmark-asr`. See
  [08](./08-registries-profiles-cli.md).
