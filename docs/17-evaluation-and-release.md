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

## Release evidence flow

The release process is a chain of increasingly realistic claims; a later claim
does not erase a missing earlier one:

```text
unit contract
    ↓
integration wiring
    ↓
real provider/model invocation
    ↓
public or sanitized comparable benchmark
    ↓
private release-vault trajectory matrix
    ↓
named platform/device gate
    ↓
release report and operator decision
```

Each stage has a different owner and failure meaning. Unit and integration
tests belong to the repository test suite. Real-flow runs prove that the
selected provider, model, tools and terminal path were actually exercised.
Benchmark runners compare pinned candidates on a declared dataset. The
private-vault matrix checks personalization, evidence, citation and repair
behavior without publishing note content. Platform gates are only valid for
the OS, terminal, audio device and model profile named in their report.

The manifest is the input contract for the final report. It identifies the
gate, required flag, command, evidence path or glob, status, reason and typed
metric checks. The report must preserve the distinction between `fail`
(observed requirement failure), `blocked` (prerequisite missing), and
`unsupported` (the named platform or device has not been exercised). An
operator can explicitly change a selected component and run a new gate; the
report never authorizes an automatic fallback.

For a failed or blocked gate, the handoff is: keep the artifact, record the
exact owner and missing condition, correct or provision the prerequisite,
rerun with a new run ID, then update the decision. Do not overwrite the old
artifact or turn a missing measurement into a passing skip.

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

## SoCa upgrade gates

The upgrade harnesses deliberately separate “implementation exists” from
“production rollout is allowed”:

| Capability | Harness / contract | Current release state |
| --- | --- | --- |
| Sufficient context | `eval.eval_sufficient_context_viquad` | prompt/model matrix available; proxy labels cannot release; production default off |
| Controlled first clause | `eval.measure_first_clause_ttfa` | paired capture/replay measured |
| TTS naturalness + intelligibility | `eval.eval_tts_quality` + TTS WER report | `blocked` on paired human references |
| Public Vietnamese retrieval | `eval.eval_vn_mteb` | full pinned ArguAna-VN comparison |
| Backchannel classifier | `eval.eval_backchannel_classifier` | `blocked` on reviewed Vietnamese audio/model winner |
| Natural disfluency | `eval.eval_disfluency` | `blocked` on private audio and real-flow receipts |
| ARM edge daemon | `eval.eval_edge_daemon` | native tests pass; device gate blocked on Linux aarch64 SBC |

Gate decisions and owners live under `eval/gates/`. An implementation may be
merged while a gate is `fail` or `blocked` only when its production wiring is
disabled or fail-closed and the blocker is preserved in `BENCHMARKS.md`.

## Commands

```bash
uv run ruff check soca tests eval
uv run pyright soca
uv run pytest -q -m 'not real_model'
uv run python -m eval.release_report --manifest <local-manifest.json> \
  --suite platform-audio-release --output <local-report.json>

# Diagnostic only unless --reviewed-labels has two-reviewer semantic consensus.
uv run python -m eval.eval_sufficient_context_viquad \
  --provider <provider> --model <model> --prompt-variant paper_definition \
  --reviewed-labels <ignored-reviewed-labels.json>
```

UIT-ViQuAD's `is_impossible` field is an extractive-MRC proxy, not semantic
sufficient-context ground truth. Release evidence therefore requires exact
sample coverage under `sufficient_context_semantic_v1`, matching per-case labels
from at least two named reviewers, and records only the reviewed-manifest hash in the public
artifact. The detailed measurements and historical dispositions remain in
[`BENCHMARKS.md`](../BENCHMARKS.md); the source-specific decisions live under
[`docs/adr/`](adr/).
