# ADR 0006: Isolated Qwen ASR worker runtime

Status: accepted, with a time-bounded security exception

Date: 2026-08-02

## Decision

The production Qwen ASR subprocess uses an exact-locked CPython 3.11.14
environment under `runtime/qwen-asr`. It installs `qwen-asr==0.0.6` and the
dependencies required by model startup, transcription, confidence scoring,
IPC, and teardown. Upstream web/demo dependencies (`gradio`, `flask`, `sox`,
`qwen-omni-utils`, and `pytz`) are excluded by the resolver.

The worker installs a reproducibly built SoCa wheel with `--no-deps`. It never
installs the repository editable and does not inherit the main SoCa dependency
graph. A private receipt binds the worker to the dependency lock and wheel
digest. The lock currently supports macOS arm64 only; unsupported platforms
must fail rather than reuse this lock.

## Evidence and alternatives

The bake-off used Qwen3-ASR-0.6B revision
`5eb144179a02acc5e5ba31e748d22b0cf3e303b0` on an Apple M4 Pro with 48 GB RAM,
macOS 15.7.4, and four real code-switch clips (`cs_000`, `cs_007`, `cs_016`,
`cs_033`). All candidates produced byte-identical normalized transcript and
logprob records (`1c5b56be...f4444`).

| Layout | Packages | Installed bytes | Cold provision | Cold cache | Startup | Mean RTF | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| upstream-locked | 92 | 1,214,216,483 | 37.156 s | 1,233,660,887 B | 5.224 s | 0.1115 | 6.303 GB |
| explicit-minimal | 51 | 1,005,908,806 | 27.606 s | 1,033,749,524 B | 4.270 s | 0.1075 | 6.263 GB |
| optionalized fork | 46 | 936,572,841 | 23.841 s | 964,557,099 B | 3.281 s | 0.1105 | 6.029 GB |

`explicit-minimal` wins. It removes 41 packages and about 208 MB, lowers cold
provision time by 25.7%, and lowers measured startup by 18.3% without changing
Qwen source. The fork saves another 69 MB and roughly one second of startup,
but it would create a permanent source-maintenance obligation solely to lazy
load forced alignment. That trade-off is not justified for a secondary ASR
worker. The upstream import issue remains documented in
[Qwen3-ASR issue 105](https://github.com/QwenLM/Qwen3-ASR/issues/105).

The committed CycloneDX SBOM is generated from the exact lock. `pip-licenses`
found metadata for all 51 locked packages and no unknown license entries.
The existing socket contract suite covers malformed/unknown operations, typed
transcription failures, concurrent channels, request timeouts, and teardown.
Offline real-service rehearsals passed with both the 0.6B release artifact and
the 1.7B reference artifact; both exited cleanly and removed socket/ready files.
The complete repository suite passed with 1,448 tests and four expected skips.

## Security exception

`pip-audit` reports five advisories against the upstream-required
`transformers==4.57.6`, including
[GHSA-29pf-2h5f-8g72](https://github.com/advisories/GHSA-29pf-2h5f-8g72),
[GHSA-69w3-r845-3855](https://github.com/advisories/GHSA-69w3-r845-3855), and
[GHSA-fgcw-684q-jj6r](https://github.com/advisories/GHSA-fgcw-684q-jj6r).
Upgrading Transformers independently is not supported by Qwen ASR 0.0.6.

This exception is owned by `@finalflash159` and expires on 2026-11-01 or when
Qwen publishes a compatible release, whichever comes first. Runtime mitigation
is mandatory: load only a locally provisioned, full-hash-verified Qwen artifact;
set Hugging Face and Transformers offline modes; never accept an arbitrary
model repository/configuration at voice startup; and never invoke Trainer,
X-CLIP, or LightGlue paths. The model-store boundary must enforce these controls
before Qwen becomes the default voice backend. Failure to verify the artifact
must be terminal, not a network fallback.

## Consequences

- Main-runtime package versions remain unchanged.
- Worker recreation requires `uv==0.11.16` and the committed lock.
- Exact sync intentionally removes undeclared packages; the provisioner then
  installs one digest-recorded, non-editable SoCa wheel.
- A future Qwen release triggers a fresh dependency bake-off, SBOM, audit, real
  transcript/logprob parity run, and explicit replacement decision.
