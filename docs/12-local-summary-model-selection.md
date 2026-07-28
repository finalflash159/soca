# Local working-summary model selection

## Current decision

`Qwen3-4B-Instruct-2507 Q4_K_M` is the selected production working-memory
summarizer. The product owner explicitly accepted its measured quality limits
and revised the release gate after the original quality-first decision.

Production now uses `background_summary`, triggers automatically at 15,000
approximate working-memory tokens, keeps a 16,384-token emergency ceiling, and
loads the local worker only for one compaction job. Missing, malformed, or
non-private weights fall back to deterministic `trim_only`; there is no remote
summary fallback or automatic runtime download.

The revised gate is explicit in `PRODUCTION_SUMMARY_RELEASE_GATE`:

| Gate | Accepted threshold | Measured |
| --- | ---: | ---: |
| schema validity | 100% | 100% |
| single fact recall | >= 80% | 80.0% |
| rolling fact recall | >= 70% | 72.5% |
| negative-state cleanliness | 100% | 100% |
| forbidden surface | 0% | 0% |
| clean exit / stopped worker | 100% / 100% | 100% / 100% |
| peak child RSS | <= 8,192 MiB | 6,030 MiB at the 15K trigger |

This acceptance does not erase the known weak areas: decision recall is 84%,
correction recall is 92%, and mixed Vietnamese/code/path placement recall is
8%. Typed ID-addressed deltas remain the required follow-up.

The decision record is
[`adr/0001-local-summary-model.md`](./adr/0001-local-summary-model.md).

## Why the first benchmark was replaced

The old `summary_session_vi_v1` had 200 rows but only eight unique expected
payloads. The old rolling set had 40 rows with one effective expected state.
Those files were deleted from the current tree and their measurements are
marked invalidated in `BENCHMARKS.md`.

The replacement suites are:

- `summary_session_vi_v2.jsonl`: 200 unique inputs, eight families, 151 unique
  expected states. Injection and noise are intentionally repeated negative
  states whose correct output is empty.
- `summary_rolling_vi_v2.jsonl`: 40 distinct sessions, four compaction
  generations each. Every generation receives the candidate's actual previous
  artifact, never the annotated expected artifact.

State-bearing cases use annotated field-and-anchor facts. Exact sentence match
is retained only as a diagnostic. Negative cases separately measure whether the
model leaves the artifact entirely empty.

## Public real-data suite

Public data is provisioned explicitly by
`scripts/provision_summary_eval_data.py`. It is stored under the ignored,
private `eval/data/summary_public/` directory. The manifest pins repository
revision, license note, deterministic sample policy, row count, and SHA-256.

| Dataset | Local rows | Role | License note |
| --- | ---: | --- | --- |
| [VSoLSCSum](https://aclanthology.org/W16-5405/) | 141 | Vietnamese social-context summaries with human annotation | CC-BY-4.0 per SEACrowd; upstream repository has no standalone license file |
| [SEAHORSE](https://aclanthology.org/2023.emnlp-main.584/) Vietnamese | 64 | Human-rated multilingual/reference sanity | Dataset card says CC; source-dataset licenses still apply |
| [WikiLingua](https://aclanthology.org/2020.findings-emnlp.360/) Vietnamese | 64 | Crowdsourced how-to summarization | CC-BY-3.0 |
| [XL-Sum](https://aclanthology.org/2021.findings-acl.413/) Vietnamese | 64 | Professionally edited news-summary sanity | CC-BY-NC-SA-4.0 |
| [DialogSum](https://aclanthology.org/2021.findings-acl.449/) English | 64 | Secondary dialogue-structure control, not Vietnamese evidence | CC-BY-NC-SA-4.0 |

Total local public suite: 397 records. The comparative probe used the first
eight deterministically sampled records per dataset and candidate
(40 records/candidate, 200 generations total). Public overlap and embedding
similarity are reports only; they do not establish state correctness or
factual attribution.

## Candidate registry

Benchmark candidates were provisioned under the ignored repository model
directory. The selected production weight was migrated without duplication to
`~/.local/share/soca/models/summary/<key>/<immutable-revision>/`, verified
against byte count and SHA-256, and kept at `0600`. Runtime never downloads it.

| Candidate | File bytes | Quant | License/release status |
| --- | ---: | --- | --- |
| Qwen3-0.6B | 639,446,688 | Q8_0 | Apache-2.0 resource floor |
| Qwen3-1.7B | 1,834,426,016 | Q8_0 | Apache-2.0 compact candidate |
| Qwen2.5-3B-Instruct | 2,104,932,768 | Q4_K_M | Qwen Research License; not release-approved |
| Qwen3-4B | 2,497,280,256 | Q4_K_M | Apache-2.0 official control |
| Qwen3-4B-Instruct-2507 | 2,497,281,120 | Q4_K_M | Apache-2.0 upstream; community Unsloth GGUF |

Qwen's SAE/`Qwen-Scope` artifacts are sparse autoencoders for interpretability,
not generative summarizers, and were not treated as candidates.

After the decision, the four rejected GGUF files were deleted, freeing about
6.7 GB. Only the 2.3-GiB Instruct-2507 production weight remains locally. Small
Hugging Face metadata directories may remain; every removed weight is
reprovisionable from the pinned registry.

## Measurement protocol

The model-selection generation runs use `llama-cpp-python 0.3.16`, Metal,
`n_gpu_layers=-1`, eight threads, `n_ctx=4096`, `temperature=0`, and
grammar-constrained JSON. The final prompt fingerprint is
`aa317641bb249d5b`; it hashes both policy text and JSON schema, so a prompt edit
cannot silently reuse an old fingerprint.

The decision order is quality first:

1. schema validity, annotated state recall, negative cleanliness, and
   injection/stale surface checks;
2. rolling retention using actual previous artifacts;
3. only after quality passes, cold load, peak RSS, latency, and disk cost.

No weighted score allows low RAM to compensate for lost decisions or
corrections.

## Public real-data probe

Metrics below are means over successfully parsed outputs. The schema
denominator is always 40. Model2Vec semantic cosine is secondary and was
computed offline on the same stored outputs.

| Candidate | Schema | Token F1 | ROUGE-L F1 | Model2Vec cosine |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 77.5% (31/40) | 0.2522 | 0.1719 | 0.6759 on 31 |
| Qwen3-1.7B | 97.5% (39/40) | 0.2667 | 0.1728 | 0.6746 on 39 |
| Qwen2.5-3B-Instruct | 100% | **0.2707** | **0.1899** | 0.6655 |
| Qwen3-4B | 100% | 0.2524 | 0.1705 | 0.6331 |
| Qwen3-4B-Instruct-2507 | 100% | 0.2690 | 0.1713 | **0.6876** |

The differences are small and domain-dependent. In particular, 0.6B's
semantic mean excludes nine parse failures and must not be compared as if its
denominator were 40.

## Final SoCa-state result

Only the strongest finalist received the final-prompt full captures. Earlier
captures are retained locally as prompt/fixture-defect evidence, but they are
not part of this decision.

| Run | Rows/generations | Schema | Fact recall | Negative clean | Forbidden surface | p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single state v2 | 200 | 100% | 80.0% | 100% | 0% | 1.508 / 2.787 s |
| Rolling v2 | 40 sessions / 160 generations | 100% | 72.5% | n/a | 0% | 11.111 / 13.863 s per session |

Single-state family recall:

| Family | Recall |
| --- | ---: |
| constraints | 100% |
| open items | 100% |
| commitments | 96% |
| corrections | 92% |
| decisions | 84% |
| mixed Vietnamese/code/path | 8% |

Rolling family recall ranged from 67.5% for correction chains to 82.5% for
open-item chains. This failed the original 95% decision/correction gate. It is
accepted under the revised product gate above, with the limitation documented
rather than hidden.

## Cold process and real lifecycle

Eight balanced cold jobs used one child process per job:

| Metric | Result |
| --- | ---: |
| schema validity | 100% |
| outer end-to-end p50 / p95 | 5.318 / 6.387 s |
| child model-load p50 | 300 ms |
| child generation p50 | 2.357 s |
| child peak RSS max | 3,822 MiB |
| clean child exit | 8/8 |
| supervisor observed worker stopped | 8/8 |

The explicit real-flow smoke also passed:

`accepted → local load/generate → publish → idle → 0600 checkpoint → reload → rendered-state round trip`

No summary worker remained after either run.

The production 15K auto-trigger was then exercised separately:

| Metric | Result |
| --- | ---: |
| trigger token count | 15,288 |
| dynamically allocated context | 20,480 tokens |
| model load | 466 ms |
| generation | 44.78 s |
| peak child RSS | 6,030 MiB |
| child exit / stopped | 0 / true |
| checkpoint + rendered round trip | pass |

Context allocation is dynamic from 4K up to a 32K maximum, so a short manual
compaction does not pay the 32K KV-cache cost. A real post-compaction answer
turn also passed with the persisted OpenRouter selection
`qwen/qwen3.7-flash`: the provider was called, received the rendered earlier
state containing the active `TTS B` decision, and returned non-empty content.

## Runtime corrections made during the bake-off

- `WorkingSummaryArtifact` now enforces its 256-token budget over prose and all
  structured fields, not prose alone.
- `WorkingMemory.render()` now injects all five structured fields into the
  answer context. Previously it discarded them and rendered only `summary`.
- Cold telemetry now includes load/generation latency, child peak RSS, exit
  code, and supervisor-observed stopped state.
- Benchmark outputs and public data are private (`0600`) and ignored by Git.
- Text, voice, and the UI engine share the same production `SessionMemory`
  worker wiring and cancel it during engine shutdown.
- Session/context character caps were raised to carry the 15K working window;
  local answer models now use their declared context window instead of an
  unconditional 4K cap.

## Architectural follow-up

Repeated rolling runs showed that asking an LLM to regenerate the complete
state can drop an unchanged field in a later generation. More prompt text did
not make this stable.

Relevant public designs point toward explicit state identity and update
semantics:

- [LangGraph running summaries](https://langchain-ai.github.io/langgraph/how-tos/memory/manage-conversation-history/)
  summarize and remove old messages, but do not provide typed stale-fact
  validity by themselves.
- [Letta memory blocks](https://docs.letta.com/guides/core-concepts/memory/memory-blocks)
  give persistent blocks identity, descriptions, limits, and explicit updates.
- [Zep temporal facts](https://help.getzep.com/v2/facts) preserve history with
  validity intervals instead of trusting one regenerated prose state.

The next experiment should therefore use typed, ID-addressed state deltas
(`add`, `replace`, `resolve`, `carry`) and a deterministic generic merge
engine. Content classification remains model-driven; the state machine must
not use keyword/regex rules. The v2 rolling suite remains the regression gate
for that design.

## Reproduction

```bash
uv run python scripts/provision_summary_eval_data.py --sample-size 64
uv run python scripts/generate_summary_benchmark_fixtures.py

uv run python eval/eval_summary_public.py \
  --dataset eval/data/summary_public/vsolscsum_vi.jsonl \
  --model-key qwen3_4b_instruct_2507_q4_k_m \
  --model-path ~/.local/share/soca/models/summary/qwen3_4b_instruct_2507_q4_k_m/<revision>/<file> \
  --limit 8 --output eval/results/summary-public.json

uv run python eval/eval_summary_bakeoff.py \
  --dataset eval/prompts/summary_session_vi_v2.jsonl \
  --model-key qwen3_4b_instruct_2507_q4_k_m \
  --model-path ~/.local/share/soca/models/summary/qwen3_4b_instruct_2507_q4_k_m/<revision>/<file> \
  --output eval/results/summary-state.json

uv run python scripts/smoke_test_working_memory_real.py \
  --answer-smoke \
  --output eval/results/summary-production-full.json
```
