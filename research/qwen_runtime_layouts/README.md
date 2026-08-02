# Qwen ASR runtime layout bake-off

This directory preserves the three dependency declarations and the deterministic
runtime harness used for ADR 0006. Raw command output, environment paths, full
transcripts, and vulnerability scanner output stay in the ignored/local
benchmark area; `results/macos-arm64.json` is the sanitized comparison.

Each candidate is locked and synchronized with `uv==0.11.16`. The optionalized
candidate applies `candidates/optionalized-fork/lazy_forced_aligner.patch` to
the installed Qwen ASR 0.0.6 package only for research. Production does not use
that patch.

The runtime test must use an immutable local model snapshot and offline mode:

```bash
PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  <candidate>/.venv/bin/python research/qwen_runtime_layouts/benchmark_runtime.py \
  --run-type release_benchmark --layout <name> \
  --artifact-key qwen3_asr_0_6b --artifact-revision <immutable-revision> \
  --model <absolute-snapshot> --lock <candidate-lock> \
  --source-revision <git-commit> --raw-log-reference <ignored-local-path> \
  <audio>...
```
