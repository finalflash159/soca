# ADR 0005: Immutable Qwen ASR artifact identity

Date: 2026-08-02
Status: accepted for contract; blocked — owner: @finalflash159; reason: provisioning requires
private mirror commits and an exact worker lock digest before qualification

## Context

Qwen ASR previously accepted raw Hugging Face model IDs in the backend, service client and
service server. Their defaults disagreed: the backend selected 0.6B while the service selected
1.7B. A model ID also did not identify repository revision, expected files, runtime dependencies,
context policy or operational role. That was insufficient for offline startup and made an
accidental model change possible.

The upstream Qwen repository recommends a fresh isolated environment. Its Transformers loader
accepts a pretrained model name or local path. Hugging Face supports exact revision downloads,
but an omitted revision resolves mutable repository state. These properties make an immutable
local artifact contract the appropriate boundary between provisioning and runtime.

Primary references:

- <https://github.com/QwenLM/Qwen3-ASR>
- <https://huggingface.co/Qwen/Qwen3-ASR-0.6B>
- <https://huggingface.co/Qwen/Qwen3-ASR-1.7B>
- <https://huggingface.co/docs/huggingface_hub/en/guides/download>
- <https://huggingface.co/docs/hub/en/security-tokens>

## Decision

SoCa has two typed artifacts:

| Key | Role | Upstream revision |
|---|---|---|
| `qwen3_asr_0_6b` | release | `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` |
| `qwen3_asr_1_7b` | reference | `7278e1e70fe206f11671096ffdd38061171dd6e5` |

Each packaged manifest pins the complete upstream file set, byte sizes and SHA-256 values. The
hashes were computed from local snapshots at those exact commits and cross-checked against the
upstream Hub metadata for revision, size and large-file SHA-256. The manifest itself is encoded as
canonical JSON and receives its own SHA-256 identity.

The release model is the only default. The reference model must be selected explicitly and is not
a fallback. Local model paths derive from:

```text
${XDG_DATA_HOME:-~/.local/share}/soca/models/asr/<artifact-key>/<upstream-commit>/
```

Artifact manifests reject mutable revisions, unsafe relative file paths, duplicate/unsorted file
entries, invalid digests, unknown schema fields and role mismatches. Receipt inspection rejects
symlinks and group/world permissions.

The private mirror, exact worker lock digest and final context-policy digest are deliberately
`null` today. Null means unqualified, not “use latest” and not “ignore validation”. Later readiness
and qualification work must require these pins before production activation.

## Consequences

- Importing the registry reads only packaged JSON; it does not import Qwen, load weights or access
  the network.
- The model store and provisioner can validate content without embedding provider-specific logic.
- Existing research callers may still pass an explicit model ID until the service is migrated to
  local receipt-backed paths. No voice production path is changed by this decision alone.
- Private Hugging Face mirrors must preserve Apache-2.0, upstream provenance and byte identity.
  Provisioning may use a fine-grained read token; runtime must receive neither token nor network
  source.
