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
| 2 — text conversation | not started |
| 3 — voice | not started |
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

**`answer_delta` is not token streaming yet.** It fires once with the full
answer, so `composing` shows as one step rather than an animation. Real
streaming is pending on `feat/chat-text-streaming`; phase 2 depends on it.

## Deviations from the plan

Recorded per plan §5.7 rule 4.

| Deviation | Why |
| --- | --- |
| Removed the unused `import * as React` from `src/components/ui/scroll-area.tsx` | The registry file does not compile under this project's `noUnusedLocals`. Re-adding the component will reintroduce it. |
| ASR `transcribing` / `asr_partial` map to `working` | The plan's nine states cover assistant reasoning, not speech recognition. Adding a tenth state would break the §0.2 single-source rule. |
| The `memory` turn phase maps to `searching`, not `weaving` | `weaving` is reserved for compaction, which is what the plan means by "nén working memory". The `memory` phase is archive retrieval. |
| `connecting` is bounded by time, not by backend | A remote backend is not an activity. `connecting` shows while synthesis is open and no answer text has arrived, then becomes `composing`. |
