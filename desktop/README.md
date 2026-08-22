# SoCa desktop app

Tauri v2 shell over the same `soca engine` NDJSON boundary the Ink TUI uses.
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
| 4 — knowledge & memory | done — retrieval inspector, memory traces, proposals, vault/index |
| 5 — settings & packaging | done in code — packaged native Python sidecar, stable app-data location, migration proof, updater UI, and native package/release workflows; signed release evidence is still credential- and CI-gated in [`docs/19-desktop-packaging.md`](../docs/19-desktop-packaging.md) |

## Update — 2026-08-17

**Interface rebuilt, with light and dark modes.**

- A labelled sidebar replaces the 56 px icon rail. Session, Knowledge, Voice and
  Settings move from tabs inside one sliding sheet to **pages**. That also fixes
  a bug structurally: voice mode used to cover the whole window, so mid-call
  there was no way to reach settings or restart the engine.
- **Light mode.** The shared palette has a single ramp built for a dark
  terminal — its gold measures 1.6:1 on white, which is not readable text. Light
  keeps the same hue and moves down the ramp to `#8f6914`, at 5.0:1.
- Answers **render as markdown**: headings, lists, tables, highlighted code and
  KaTeX formulas. They were previously drawn as plain text, so `#` and `|`
  appeared literally.
- A spoken turn is now a **real turn** in the same transcript as chat. Voice used
  to reduce into live-signal state only, so everything said was lost when the
  turn ended.
- A per-turn step trail, built from `turn_progress` frames the engine already
  emits.

**Measured fixes.**

- Chat and voice now use **separate system prompts**. The runtime prompt forbids
  markdown because TTS would read `**` aloud; chat shared it, so answers could
  never be formatted. `RuntimeOptions.answer_format` picks the prompt per
  surface.
- A chat chunk is a **markdown block**, not a sentence. The TTS chunker inserts
  commas as audible pauses (`### Ví dụ` → `### Ví dụ,`, and a fence's language
  tag likewise) and strips the newlines a list depends on. Measured on a live
  turn after the fix: the stream-joined text matches `chat/done.text` byte for
  byte.
- A package carries a native frozen `soca-engine` sidecar, rather than searching
  a developer's `PATH`, Homebrew, virtualenv, or checkout. The startup recovery
  field remains an explicit opt-in override when diagnostics require it.
- The local LLM loads its weights on **first use**, never at startup.
- `st_dev` is excluded from artifact identity. It names the mount, and APFS
  reassigns it across reboots, which made every file look tampered with.

## Run

```bash
npm install
npm run tauri dev
```

Development can still launch the active checkout's `soca` command. A packaged
app launches only the frozen sidecar shipped with that package and stores its
data under the OS-managed app-data root. Type a recovery path in the startup
screen only when intentionally selecting a different engine.

```bash
npm test
npm run build     # tsc + vite
npm run package:sidecar
npm run tauri:build -- --bundles app,dmg
cd src-tauri && cargo clippy --all-targets
```

## Layout

```text
src/components/StartupView.tsx  the window before the engine runs
src/components/SessionView.tsx  the window after: rail, transcript, one control block
src/components/VoiceMode.tsx    full-screen spoken turn; orb plus an rms-driven halo
src/components/SessionPanel.tsx prompt budget and token usage
src/engine/protocol.ts   read-shapes for docs/18; no validation (§7 tolerates unknown fields)
src/engine/orb.ts        engine frames → one of nine thinking-orbs states
src/engine/conversation.ts  chat turn assembly from the answer_delta stream
src/engine/voice.ts      voice-loop state from the 20 voice event types
src/engine/knowledge.ts  retrieval traces, memory traces, proposals, index jobs
src/engine/settings.ts   providers, catalogs, key status, LLM config, profiles
src/engine/useEngine.ts  Tauri event bridge; transport state only
src/components/ui/       shadcn registry components — do not hand-edit structurally
src-tauri/src/engine.rs  sidecar process manager and the docs/18 §7 shutdown sequence
```

## Layout

Two views, switched on engine state, plus an overlay. Derived from reading
LiveKit's reference app rather than guessing — the findings and what was
rejected are recorded with the component they were evaluated for.

* **Not connected → `StartupView`.** One orb, one line, one button. The engine
  executable field hides behind "Engine không chạy được?" instead of sitting in
  the header, where it made the product read as a debug console.
* **Connected → `SessionView`.** The orb is the centre of the screen while there
  is nothing to read, and steps back to the header once the transcript fills.
  Input and its controls are one bordered block at the bottom, `max-w-2xl`.
* **The inspector is a right-hand sheet, not a tab.** Retrieval, memory,
  settings and the frame log overlay the conversation. Checking a citation must
  not mean leaving the answer it belongs to.
* **Voice mode takes the whole window.** A spoken turn has nothing to read and
  nothing to click, so the rail's voice button clears everything except the orb,
  a halo driven by `voice_level.rms`, and a control bar. The orb stays at its
  tuned 64 px — the library ships two fixed designs, not a scalable one, so the
  size that fills a screen is the halo around it, not the canvas.
* **No conversation sidebar.** Every comparable app has one; SoCa cannot. The
  protocol has no command that lists or reloads past conversations
  (`docs/18-engine-protocol.md` §2), so the sidebar would be empty chrome. Fixing
  that is engine work, not UI work.

## Decisions worth knowing

**The local LLM loads on first use, never on construction.** `LocalLlamaCppLLM`
validates its model path eagerly — a stat, so a missing file still fails when the
engine is selected — but defers `Llama(...)` until something actually generates.
Before that, every surface which merely *built* a runtime paid gigabytes of RAM
and seconds of load: `soca chat` at startup, a status query, a provider switch
that ended up remote. Releasing is symmetric: `close()` on an engine that never
loaded is a no-op, and using one after close raises rather than silently
reloading a released model.

**The neutrals are darker than the shared palette; the accent is not.** See the
comment at the top of `src/index.css` — `BG` is a contrast token in a terminal
and the whole canvas in a window.

**Process management is hand-rolled, not `tauri-plugin-shell`.** The shutdown
sequence in `docs/18` §7 — send `quit`, close stdin, wait for `bye`, then
escalate — needs direct control of the child's stdio and exit. The plugin's
abstraction does not expose enough of it, and `bye` is the only evidence the
engine released the microphone and its provider clients.

**The package includes the Python sidecar.** Tauri's `externalBin` entry embeds
a PyInstaller-built `soca-engine-$TARGET_TRIPLE` in every native package. It is
the only automatic production engine route, so a Finder/Start Menu launch has
no hidden dependency on PATH or a checkout. The build also explicitly collects
`llama_cpp` native libraries and `torchcodec` distribution metadata; without
them a freeze can build yet fail before the protocol `hello` frame. The exact
package size is a release artifact, not a stable claim in this repository.

The Settings page contains a signed-updater surface. It can show that a build
is not configured for updates rather than implying a successful check. The
keys, platform certificates, notarization credentials, and CI release evidence
remain external release gates; read
[`docs/19-desktop-packaging.md`](../docs/19-desktop-packaging.md) and
[`RELEASE_NOTES.md`](../RELEASE_NOTES.md) before publishing.

**Answers are assembled by appending, never replacing.** `docs/18` §6:
concatenating every `answer_delta` `payload.text` in order reproduces the final
answer exactly. `conversation.ts` also compares the reassembly against
`chat/done` and surfaces a mismatch instead of silently trusting the final text —
per-chunk text handling has regressed before.

**Not every turn animates.** A turn with a tool or retrieval is held by the
bounded controller until synthesis and verification finish, then emits every
chunk at once. Between deltas the UI is driven by `turn_progress.phase`, not by
delta arrival.

**API keys are write-only in this process.** The key field is sent with
`llm_set_key` and cleared immediately; the engine's keyring owns it. The only
key material that ever comes back is the `masked` form in `llm_key_status`, and
nothing here stores or logs a raw key.

**One settings surface, not six.** SoCa answers "how is this configured?"
through `/settings`, `soca status`, `soca profiles`, `soca llm-models`,
`soca asr-models` and `soca knowledge model`. The Settings tab is one place for
provider, key, model, LLM config and runtime profiles.

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
