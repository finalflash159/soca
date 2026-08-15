# UI components — sources and exceptions

The desktop app takes components from registries rather than hand-writing them,
per `zplan/soca_desktop_app_plan.vi.md` §0.1 and §5.7. Registry components are
built on Radix or Base UI and already handle focus, keyboard and ARIA; a
hand-rolled equivalent starts with none of that.

Rule 4 of §5.7 requires that anything written by hand is recorded here with its
reason. This page is that record.

## Installed from registries

| Source | Components |
| --- | --- |
| [shadcn/ui](https://ui.shadcn.com/) | `alert`, `badge`, `button`, `button-group`, `card`, `carousel`, `collapsible`, `command`, `dialog`, `dropdown-menu`, `hover-card`, `input`, `input-group`, `progress`, `scroll-area`, `select`, `separator`, `textarea`, `tooltip` |
| [AI Elements](https://registry.ai-sdk.dev/) | `conversation`, `message`, `sources`, `loader` |

All are copy-to-codebase: they live in `desktop/src/components/` as editable
source, not as a version in `node_modules`.

## Written by hand, with reasons

### Level meter — `components/VoiceHud.tsx`

**Considered:** ElevenLabs UI `Live Waveform`, LiveKit Agents UI audio
visualizer.

**Rejected for an architectural reason, not a stylistic one.** Both read the
microphone in the browser through the Web Audio API. SoCa's audio is captured by
the Python engine on a single duplex stream with AEC3, and the two-stream
arrangement — one capture for processing, another for display — is the design
that already failed on clock drift. A second `getUserMedia` in the WebView would
also compete with AEC3 for the device, and barge-in depends entirely on AEC.

The engine already publishes what a meter needs: `voice_level.metadata.rms`,
measured on the same buffer the recogniser sees. The hand-written meter renders
that array.

It is drawn as discrete bars rather than a smoothed waveform on purpose: rms is
one magnitude per frame, not a sample buffer, and a smooth curve would imply a
resolution the data does not have.

### Message composer — `components/ChatView.tsx`

**Considered:** AI Elements `prompt-input`.

**Installed, then removed:** it does not compile against `@base-ui/react@1.7.0`.
It passes `openDelay` and `closeDelay` to a PreviewCard root that accepts
neither, and calls `Array.prototype.at` under this project's ES2020 lib target.
`@base-ui/react` is already at its latest release, so there is no version to
upgrade to. `inline-citation` fails the same way and was removed with it.

Replaced by a `textarea` plus the registry `Button`. Revisit when the registry
catches up — the composer is the one place where the registry component is
clearly better than what is here now (attachments, command menu, speech button).

## Components used with a deliberately unused code path

`message.tsx` ships a Streamdown-based `MessageResponse`. It is not used:
`SOCA_RUNTIME_SYSTEM_PROMPT` forbids markdown in answers because this is spoken
conversation, so answers render with `whitespace-pre-wrap`. The import still
costs bundle weight (see `desktop/README.md`), which is acceptable for an app
loading from local disk.

## Rules that still apply

1. Search the registry before writing JSX for any interface element.
2. Install with `npx shadcn@latest add …`, never by copy-paste, so the update
   path survives.
3. Edit installed components at the token/theme level only.
4. Anything hand-written gets an entry on this page first.
5. Agent-state animation is `thinking-orbs` and nothing else — no extra
   spinners, no bespoke skeleton pulses for agent state.
