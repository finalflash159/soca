# Platform and audio release gates

This document defines what can be proven automatically and what remains a real
machine or device check. A green unit test is never promoted to a hardware
claim.

## Report contract

`eval/release_report.py` consumes a versioned manifest and writes a JSON report
plus a short Markdown view. Each gate has an explicit status:

- `pass`: the recorded evidence exists and is hashed;
- `fail`: the evidence demonstrates a release failure;
- `blocked`: the gate could not run because a required artifact, provider,
  private corpus, or external prerequisite is missing;
- `unsupported`: this platform or device has not been exercised.

Required non-`pass` gates block the release. The report never treats a missing
entry as a skipped success.

The executable runner additionally supports a repo-relative `result_path` or
latest-match `result_glob`, plus typed JSON checks (`eq`, `ne`, `gt`, `gte`,
`lt`, `lte`). A zero exit code is not sufficient when a command declares
metrics: the result artifact must exist and every declared check must pass.
Raw command logs and private result artifacts remain under ignored local
storage; the committed evidence report contains only sanitized aggregates and
hashes.

Example:

```json
{
  "schema_version": 1,
  "gates": [
    {
      "id": "session_process_boundary",
      "status": "pass",
      "required": true,
      "reason": "checkpoint restored in a new process",
      "evidence": ["docs/evidence/session-release.json"],
      "command": ["uv", "run", "pytest", "-q", "tests/test_session_process_boundary.py"]
    },
    {
      "id": "macos_ime_pty",
      "status": "unsupported",
      "required": true,
      "reason": "manual iTerm2 and Terminal.app matrix not run",
      "evidence": [],
      "command": ["soca", "ui"]
    }
  ]
}
```

## Platform input matrix

Run the application manually for every supported terminal/IME pair and attach
the local transcript or screen recording outside Git. Check Vietnamese Telex
and VNI input, replacement composition, delete, tab, arrows, paste and resize.
The report must name the terminal and OS version. A PTY reducer test proves the
input contract only; it cannot prove an operating-system IME boundary.

## Audio matrix

Keep these outcomes separate:

- ASR model/service health and typed failure paths;
- SmartTurn/VAD/AEC replay;
- TTS synthesis and queue metrics;
- real microphone, speaker, barge-in and underflow behavior;
- manual listening quality.

Silence, generated TTS, or a text-only voice run may be a smoke or wiring
artifact, but must not be called a microphone or listening release pass.

## Reproduction commands

```bash
uv run ruff check soca tests eval
uv run pyright soca
uv run pytest -q -m 'not real_model'
uv run pytest -q tests/test_index_watcher_integration.py tests/test_session_process_boundary.py
uv run python -m eval.eval_grounding --vault <private-vault> --dataset <private-dataset> \
  --variant hybrid --backend aiteamvn_v2 --output <local-report.json>
uv run python eval/workflow_control_gate.py --backend remote \
  --model <remote-model> --corpus <private-vault> --dataset <trajectory.jsonl> \
  --output <local-workflow-report.json>
uv run python -m eval.release_report --manifest <local-manifest.json> \
  --suite platform-audio-release --output <local-report.json>
```

Private vaults, transcripts, audio, provider logs and local report manifests
stay under ignored local storage. Commit only a reviewed, sanitized aggregate.
