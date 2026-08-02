# ADR 0007: Immutable local Qwen ASR model store

Status: accepted, mirror publication blocked

Date: 2026-08-02

## Decision

Qwen ASR workers load only immutable local generations under the XDG data root:

```text
~/.local/share/soca/models/asr/<artifact-key>/<upstream-commit>/
~/.local/share/soca/models/asr/receipts/<artifact-key>.json
```

This is a persistent runtime store, not a Hugging Face cache. A generation is
removed only by an explicit full-revision GC command. Chat, voice startup and
status inspection never download, update or repair model files.

`scripts/provision_qwen_asr.py` resolves one exact source, verifies every file
against the packaged size and SHA-256 manifest, materializes a private staging
generation, validates the Safetensors/index/config structure, and runs a real
recorded-audio transcription from the staging path with Hugging Face and
Transformers forced offline. Only then does it atomically activate a read-only
generation and fsync a private canonical receipt.

The source policy is strict:

- a pinned private mirror is the normal release source;
- upstream import requires the explicit `--source upstream` operator option;
- a missing or failed mirror is terminal and never falls back to upstream;
- tokens are read only from `HF_TOKEN` by the provisioner, are removed from the
  worker environment, and never appear in process arguments, receipts or
  committed evidence.

The current mirror fields are intentionally null. Creating the two private Hugging
Face repositories, copying the exact upstream generations, verifying their hashes
and committing their immutable mirror revisions remains blocked on
`@finalflash159`. Until that external task is complete, a default install returns
`MirrorNotPinned`; upstream import remains an explicit development/demo operation.

## Storage and lifecycle

The store uses APFS `clonefile` when available and falls back to a bounded,
progress-reporting copy. It never hardlinks model files to the global HF cache,
because later permission changes or mutation would then affect the shared inode.
The persistent generation therefore continues to work if the HF cache is removed.

Preflight derives final, staging, reusable and required free bytes from the exact
artifact manifest and measured filesystem state. It also verifies Darwin/arm64 and
the Qwen worker lock digest. There is no single hardcoded GB threshold. Atomic
activation requires enough space for a complete new generation while the current
one remains intact.

Quick verification compares receipt identity, file set, size, device, inode,
mtime and ctime. Deep verification additionally hashes every file, validates model
structure and repeats the offline audio health probe. Any drift reports
`invalid`; it is not repaired automatically. GC provides an explicit dry run and
deletes only one explicitly named 40-character inactive revision.

## Real evidence

Both packaged artifacts were provisioned on an Apple M4 Pro with 48 GiB RAM,
macOS 15.7.4, CPython 3.11.14 and runtime lock
`bbf96f7b...ec7f9f`:

| Role | Exact upstream revision | Logical bytes | Health RTF | Health avg logprob |
|---|---|---:|---:|---:|
| release 0.6B | `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` | 1,880,619,678 | 0.0988 | -0.0731 |
| reference 1.7B | `7278e1e70fe206f11671096ffdd38061171dd6e5` | 4,703,114,308 | 0.1989 | -0.1392 |

Each generation passed full SHA-256 verification and a real 66,435-sample,
16 kHz recorded voice clip. A second deep verification used an empty `HF_HOME`
with both offline flags enabled; both services loaded and transcribed from the
persistent paths. Receipts are mode `0600`, model files are `0400`, and store
directories are private (`0700` store roots and `0500` activated generations).
Sanitized reproducibility metadata is committed in
`docs/evidence/qwen-asr-model-store-20260802.json`; raw local logs are ignored.

## Consequences

- First install is explicit and potentially expensive; later starts use only the
  local generation.
- APFS clone-on-write minimizes initial physical duplication, but logical capacity
  and future changed blocks still need operational headroom.
- Artifact corruption, insufficient disk, lock contention, unsupported platform,
  runtime-lock drift and health failures are typed terminal outcomes.
- Service identity/readiness must consume this receipt and local path in the next
  lifecycle change; this ADR does not claim that voice startup has already switched
  from remote repository identifiers.
