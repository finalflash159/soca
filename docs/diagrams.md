# Architecture diagrams

The repository keeps a small set of reviewed, static SVGs for GitHub and local
Markdown rendering. Each SVG was drawn from the current code map and reviewed
against an editable Lucid document. Lucid is the editable collaboration source;
the local SVG is the stable repository artifact used by README/docs.

## Diagram register

| View | Repository image | Editable Lucid source |
| --- | --- | --- |
| System boundary | [system overview](assets/diagrams/system-overview.svg) | [Lucid](https://lucid.app/lucidchart/89bdaa6f-62e0-4396-a158-974025fe32af/view) |
| Controlled turn | [controlled turn](assets/diagrams/controlled-turn.svg) | [Lucid](https://lucid.app/lucidchart/2a0f7744-4b8d-4b2d-89ec-af974a3f0fa0/view) |
| Voice pipeline | [voice pipeline](assets/diagrams/voice-pipeline.svg) | [Lucid](https://lucid.app/lucidchart/c962c397-6528-40ef-838a-1a4af0a562b5/view) |
| Knowledge/index lifecycle | [knowledge lifecycle](assets/diagrams/knowledge-lifecycle.svg) | [Lucid](https://lucid.app/lucidchart/7fed0060-ee54-4fe4-8846-511b464f3a1e/view) |
| Memory lifecycle | [memory lifecycle](assets/diagrams/memory-lifecycle.svg) | [Lucid](https://lucid.app/lucidchart/11a7f978-d7a2-4106-a40c-b33d82f314ed/view) |
| UI/engine protocol | [UI protocol](assets/diagrams/ui-engine-protocol.svg) | [Lucid](https://lucid.app/lucidchart/133d6a9f-ff1d-4269-b56d-275b8193b689/view) |

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
- Shapes follow a small notation: process rectangles for work, diamonds for
  decisions, parallelograms for input/output, cylinders for persistent stores,
  and folded documents for evidence or event artifacts.
- Model and provider nodes are processes/services, not document shapes;
  document geometry is reserved for evidence, proposals, and event artifacts.
- A small junction is used when independent context sources merge before the
  prompt builder; it is not a hidden processing step.

The high-level system image is included in the repository README. Detailed
views are linked from the subsystem documents and this register.

## Visual tokens

The diagrams use a technical schematic palette rather than category cards,
gradients or translucent decoration. The canvas stays white, while nodes use
solid semantic fills with dark text; stroke color, shape, line style and labels
carry the meaning as well. These are diagram roles, not a data-visualization
series palette:

| Token | Value | Use |
| --- | --- | --- |
| paper | `#ffffff` | canvas/background |
| ink | `#16202a` | headings and node labels |
| line | `#263746` | ordinary flow and borders |
| neutral fill | `#e2e8ec` | core processing and protocol nodes |
| local fill | `#c9dde3` | local capability/state nodes |
| data fill | `#d9efc7` | input/output and user-facing data |
| control fill | `#efc36f` | decisions and bounded loops |
| remote fill | `#d8987d` | external provider boundary |
| Lucid accent | `#e4ddff` | UI/control-plane emphasis |
| terminal fill | `#acd2ba` | successful terminal outcome |

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

The result is a white paper, solid semantic nodes, dark readable labels and a
small set of semantic fills. Color is never the only signal: shape, line style,
placement and labels carry the meaning as well. A recent UX design-system
discussion makes the same distinction: brand colors and data-visualization
colors should not be forced into one collection, and labeling should remain
available when color is insufficient ([discussion](https://www.reddit.com/r/UXDesign/comments/1n6dzb7/how_do_you_handle_colors_for_data_visualization/)).
Orthogonal routing is used whenever the geometry permits it; a short corner is
used only to keep a feedback edge outside a node or label. Full system detail
stays in the coverage matrix and subsystem prose, not in an artificially
simplified hero image.
