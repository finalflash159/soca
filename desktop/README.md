# SoCa desktop app

Tauri v2 shell over the same `soca engine` NDJSON boundary the Ink TUI uses.
Plan: [`zplan/soca_desktop_app_plan.vi.md`](../zplan/soca_desktop_app_plan.vi.md).
Protocol: [`docs/18-engine-protocol.md`](../docs/18-engine-protocol.md).

**No Python is reimplemented here.** This is a third surface beside the CLI and
the TUI, not a fork of the runtime.

## Status

| Phase | State |
| --- | --- |
| 0 — protocol contract | done — `docs/18-engine-protocol.md` + `tests/test_engine_protocol_contract.py` |
| 1 — app shell + sidecar | done — start/stop engine, status screen, orb |
| 2 — text conversation | done — streamed turns, blocked/terminal states, citations |
| 3 — voice | done — HUD with engine-sourced level meter, partial transcript, barge-in |
| 4 — knowledge & memory | not started |
| 5 — settings & packaging | not started |

## Run

```bash
npm install
npm run tauri dev
```

The app spawns `soca engine`, so `soca` must be resolvable. If it is not on
PATH, override the program in the UI field, or pass args (`uv` with
`args: ["run", "soca"]`).

```bash
npm test          # orb-state mapping and protocol helpers
npm run build     # tsc + vite
cd src-tauri && cargo clippy --all-targets
```

## Layout

```text
src/engine/protocol.ts   read-shapes for docs/18; no validation (§7 tolerates unknown fields)
src/engine/orb.ts        engine frames → one of nine thinking-orbs states
src/engine/conversation.ts  chat turn assembly from the answer_delta stream
src/engine/voice.ts      voice-loop state from the 20 voice event types
src/engine/useEngine.ts  Tauri event bridge; transport state only
src/components/ui/       shadcn registry components — do not hand-edit structurally
src-tauri/src/engine.rs  sidecar process manager and the docs/18 §7 shutdown sequence
```

## Decisions worth knowing

**Process management is hand-rolled, not `tauri-plugin-shell`.** The shutdown
sequence in `docs/18` §7 — send `quit`, close stdin, wait for `bye`, then
escalate — needs direct control of the child's stdio and exit. The plugin's
abstraction does not expose enough of it, and `bye` is the only evidence the
engine released the microphone and its provider clients.

**Sidecar packaging is deferred to phase 5, deliberately.** The plan flags
bundling a Python runtime as the highest risk. Phase 1 uses the documented
fallback — the app expects `soca` to be installed — so the boundary can be
proven before the packaging problem is attacked.

**Answers are assembled by appending, never replacing.** `docs/18` §6:
concatenating every `answer_delta` `payload.text` in order reproduces the final
answer exactly. `conversation.ts` also compares the reassembly against
`chat/done` and surfaces a mismatch instead of silently trusting the final text —
per-chunk text handling has regressed before.

**Not every turn animates.** A turn with a tool or retrieval is held by the
bounded controller until synthesis and verification finish, then emits every
chunk at once. Between deltas the UI is driven by `turn_progress.phase`, not by
delta arrival.

**The WebView never opens a microphone.** The level meter is driven by the
engine's `voice_level.rms`, not by Web Audio. A second browser capture would be
the two-stream arrangement that failed on clock drift, and it would compete with
AEC3 for the device — barge-in depends entirely on AEC. See
[`docs/ui-components.md`](../docs/ui-components.md).

**No endpoint countdown is shown.** The engine publishes the silence floor and
ceiling once, at `recording`, and never a remaining-time figure. Counting down
client-side would invent a decision the engine owns.

**Markdown is not rendered.** `SOCA_RUNTIME_SYSTEM_PROMPT` forbids markdown in
answers because this is spoken conversation, so the registry's Streamdown-based
`MessageResponse` is deliberately unused; answers render with
`whitespace-pre-wrap`.

## Deviations from the plan

Recorded per plan §5.7 rule 4.

| Deviation | Why |
| --- | --- |
| Removed the unused `import * as React` from `src/components/ui/scroll-area.tsx` | The registry file does not compile under this project's `noUnusedLocals`. Re-adding the component will reintroduce it. |
| ASR `transcribing` / `asr_partial` map to `working` | The plan's nine states cover assistant reasoning, not speech recognition. Adding a tenth state would break the §0.2 single-source rule. |
| The `memory` turn phase maps to `searching`, not `weaving` | `weaving` is reserved for compaction, which is what the plan means by "nén working memory". The `memory` phase is archive retrieval. |
| `connecting` is bounded by time, not by backend | A remote backend is not an activity. `connecting` shows while synthesis is open and no answer text has arrived, then becomes `composing`. |
| Deleted the vendored `prompt-input` and `inline-citation` from AI Elements; the composer is a plain textarea | Both fail to compile against `@base-ui/react@1.7.0` — they pass `openDelay`/`closeDelay` to a PreviewCard root that has neither, and `prompt-input` calls `Array.prototype.at` under an ES2020 lib target. `@base-ui/react` is already at its latest version, so this is an upstream defect, not a version we can bump past. Revisit when the registry catches up. |
| Bundle is ~750 kB | `message.tsx` pulls `streamdown` and `shiki` at module scope, so the markdown path cannot be tree-shaken even though it is unused. Acceptable for a desktop app loading from local disk; it would not be for a web page. |
