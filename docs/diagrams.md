# Architecture diagrams

The repository keeps a small set of reviewed, static SVGs for GitHub and local
Markdown rendering. Each SVG was drawn from the current code map and reviewed
against an editable Lucid document. Lucid is the editable collaboration source;
the local SVG is the stable repository artifact used by README/docs.

## Diagram register

| View | Repository image | Editable Lucid source |
| --- | --- | --- |
| System boundary | [system overview](assets/diagrams/system-overview.svg) | [Lucid](https://lucid.app/lucidchart/1ab09027-f3f2-42c2-95bc-7d53e8b1977f/view) |
| Controlled turn | [controlled turn](assets/diagrams/controlled-turn.svg) | [Lucid](https://lucid.app/lucidchart/e56c8a9d-38cf-451c-ae56-6b8867f34774/view) |
| Voice pipeline | [voice pipeline](assets/diagrams/voice-pipeline.svg) | [Lucid](https://lucid.app/lucidchart/21a10a4a-6658-48de-bc3f-d2416b06e181/view) |
| Knowledge/index lifecycle | [knowledge lifecycle](assets/diagrams/knowledge-lifecycle.svg) | [Lucid](https://lucid.app/lucidchart/5f5447ff-ebee-4b82-b801-674f9815d737/view) |
| Memory lifecycle | [memory lifecycle](assets/diagrams/memory-lifecycle.svg) | [Lucid](https://lucid.app/lucidchart/3b8402d5-bba6-4dac-95ca-830aaaed0f62/view) |
| UI/engine protocol | [UI protocol](assets/diagrams/ui-engine-protocol.svg) | [Lucid](https://lucid.app/lucidchart/4e2fd950-01a6-47ff-8fe8-9e08dd8d0090/view) |

## Coverage matrix

The views are deliberately split by question, but together they cover the
production paths rather than presenting a decorative overview:

| View | Required coverage | Code anchors |
| --- | --- | --- |
| System boundary | TUI, microphone, engine, assistant runtime, workflow, guardrails, ASR, TTS, knowledge, memory, private state and remote LLM boundary | `soca/app/engine.py`, `soca/core/runtime.py`, `soca/core/workflow.py` |
| Controlled turn | Goal admission, capability choice, plan, action/tool execution, evidence, synthesis, verification, bounded repair and terminal outcome | `soca/core/workflow.py`, `soca/core/runtime.py` |
| Voice pipeline | Mic frames, VAD/AEC, selected ASR service, context/tools, text/voice-parity runtime, selected LLM, streamed TTS, speaker and barge-in | `soca/core/voice_runtime.py`, `soca/asr/`, `soca/tts/` |
| Knowledge lifecycle | Vault digest, parse/chunk, catalog, dense generation, verify/publish, retrieval fusion/rerank, evidence gate, watcher and explicit reindex | `soca/knowledge/`, `soca/tools/knowledge_tools.py` |
| Memory lifecycle | Working, core and archive boundaries, compaction, prompt assembly, proposals and approval-gated persistence | `soca/memory/`, `soca/core/runtime.py` |
| UI/engine protocol | Ink/React view, NDJSON command/event transport, engine dispatch, typed runtime events and return-to-chat behavior | `ui/src/`, `soca/app/engine.py` |

## Review checklist

- Each diagram has one question and one abstraction level.
- Nodes are arranged in a left-to-right or top-to-bottom reading direction.
- Every edge has one direction; dashed edges mean conditional/optional behavior.
- Shapes do not overlap, labels do not sit on a node boundary, and arrows do not
  cross unrelated flows.
- The diagram does not claim a fallback or data path that the code does not own.
- Text remains readable when rendered at README width; details stay in prose.

The high-level system image is included in the repository README. Detailed
views are linked from the subsystem documents and this register.

## Visual tokens

The diagrams use a technical schematic palette rather than category cards,
pastel fills or gradients. The canvas and nodes are white; semantic differences
are carried by stroke color, line style and labels:

| Token | Value | Use |
| --- | --- | --- |
| paper | `#ffffff` | canvas and node fill |
| ink | `#16202a` | headings and node labels |
| line | `#263746` | ordinary flow and borders |
| local | `#486372` | local capability/state boundary |
| control | `#a16f2b` | decisions and bounded loops |
| remote | `#8f4d37` | external data boundary only |

All nodes use square corners. Connector rules are intentional: straight or
orthogonal routes are the default; a short rounded corner is allowed only when
it prevents a collision. Large-radius arcs are not used for ordinary feedback.
The repository SVGs are rendered and inspected at README scale after every
diagram change.

## Why this visual language

This is a technical schematic, not a card-based product illustration. The
choice follows three external checks: Lucid's architecture guidance emphasizes
showing interactions and separating useful system views; GitHub's draw.io
guidance recommends explicit tiers, consistent spacing and orthogonal edges;
and WCAG requires sufficient contrast for text and meaningful non-text
boundaries. See [Lucid's architecture guide](https://lucid.co/blog/how-to-draw-architectural-diagrams),
[GitHub's draw.io diagram skill](https://github.com/github/awesome-copilot/blob/main/skills/draw-io-diagram-generator/SKILL.md),
and [WCAG 2.2 contrast requirements](https://www.w3.org/TR/WCAG22/#contrast-minimum).

The result is a white paper, white square nodes, dark readable labels and a
small set of semantic outline colors. Color is never the only signal: line
style, placement and labels carry the meaning as well. Orthogonal routing is
used whenever the geometry permits it; a short corner is used only to keep a
feedback edge outside a node or label. Full system detail stays in the
coverage matrix and subsystem prose, not in an artificially simplified hero
image.
