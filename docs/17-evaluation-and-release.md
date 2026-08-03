# Evaluation, evidence and release gates

`BENCHMARKS.md` is the numerical record. This document explains how to read it
and prevents a smoke run, fake model, demo vault or provider characterization
from being mistaken for release evidence.

## Evidence hierarchy

| Evidence | Proves | Does not prove |
| --- | --- | --- |
| Unit test | local contract and edge case | real model/provider quality |
| Integration test | component wiring | device or release performance |
| Real-flow smoke | selected provider/model was invoked and the terminal path completed | general quality or benchmark superiority |
| Public/sanitized benchmark | comparable quality/latency on a pinned dataset | private-vault personalization |
| Private release run | production-like quality and behavior without exposing content | public reproducibility of private notes |
| Platform/device gate | the named OS, terminal, audio device or provider behavior | unsupported platforms |

Every measured run records code revision, model/provider revision, data
revision, configuration, hardware, environment, raw-log location, metrics,
failures and decision. Raw transcripts, private vaults, provider logs and audio
stay in ignored local storage. Git receives only reviewed sanitized aggregates
and hashes.

## Required retrieval/RAG cases

The trajectory matrix must include answerable paraphrases, explicit path/source
requests, follow-up correction, ASR/phonetic noise, knowledge-versus-memory
ambiguity, hard negatives, empty evidence, conflicting evidence, citation
validation and typed backend/provider failures. A result is not complete merely
because the model produced fluent prose.

## Release status vocabulary

- `pass`: required evidence exists, is hashed and satisfies its checks.
- `fail`: evidence exists and demonstrates a failed requirement.
- `blocked`: an external prerequisite or required artifact is missing.
- `unsupported`: the platform/device/provider has not been exercised.
- `deferred`: deliberately postponed with an owner and reason.

Missing evidence is never promoted to a passing skip. Automatic fallback is not
a release repair strategy: an explicit selected production component fails
closed, and changing it is an operator decision with a new gate.

## Commands

```bash
uv run ruff check soca tests eval
uv run pyright soca
uv run pytest -q -m 'not real_model'
uv run python -m eval.release_report --manifest <local-manifest.json> \
  --suite platform-audio-release --output <local-report.json>
```

The detailed measurements and historical dispositions remain in
[`BENCHMARKS.md`](../BENCHMARKS.md); the source-specific decisions live under
[`docs/adr/`](adr/).
