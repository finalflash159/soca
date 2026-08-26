import { describe, expect, it } from "vitest";

import { initialActivity, orbLabel, orbStateFor, reduceActivity, type OrbState } from "./orb";
import type { EngineFrame } from "./protocol";
import { helloIsCompatible, isWorkflowFrame, PROTOCOL_VERSION } from "./protocol";

function fold(frames: EngineFrame[]) {
  return frames.reduce(reduceActivity, initialActivity);
}

function stateAfter(frames: EngineFrame[]) {
  return orbStateFor(fold(frames));
}

const workflow = (event: string, extra: Record<string, unknown> = {}): EngineFrame =>
  ({
    event,
    protocol_version: PROTOCOL_VERSION,
    session_id: "s",
    run_id: "r",
    goal_id: "g",
    sequence: 0,
    surface: "chat",
    timestamp: "2026-08-15T00:00:00Z",
    node: "synthesize",
    status: "active",
    payload: {},
    ...extra,
  }) as EngineFrame;

const progress = (phase: string): EngineFrame =>
  ({ event: "turn_progress", surface: "chat", phase, status: "active" }) as EngineFrame;

describe("idle", () => {
  it("breathes before anything happens", () => {
    expect(orbStateFor(initialActivity)).toBe("breathing");
  });

  it("returns to breathing after a chat turn completes", () => {
    expect(
      stateAfter([
        { event: "chat", type: "start", text: "hi" } as EngineFrame,
        progress("retrieval"),
        { event: "chat", type: "done", text: "there" } as EngineFrame,
      ]),
    ).toBe("breathing");
  });

  it("returns to breathing when a turn ends in a terminal state", () => {
    expect(
      stateAfter([
        workflow("turn_started"),
        progress("synthesis"),
        workflow("turn_terminal", { status: "completed" }),
      ]),
    ).toBe("breathing");
  });
});

describe("visible labels", () => {
  it("uses concise English activity labels without exposing provider names", () => {
    const labels: Record<OrbState, string> = {
      listening: "Listening",
      solving: "Thinking",
      searching: "Searching",
      working: "Processing",
      connecting: "Connecting",
      composing: "Responding",
      weaving: "Compacting memory",
      shaping: "Indexing",
      breathing: "Idle",
    };

    for (const [state, label] of Object.entries(labels) as [OrbState, string][]) {
      expect(orbLabel(state)).toBe(label);
    }
  });
});

describe("turn phases", () => {
  it.each([
    ["preparing", "solving"],
    ["analyzing", "solving"],
    ["routing", "solving"],
    ["memory", "searching"],
    ["retrieval", "searching"],
    ["tool", "working"],
    ["validation", "solving"],
    ["speech", "composing"],
  ])("maps phase %s to %s", (phase, expected) => {
    expect(stateAfter([progress(phase)])).toBe(expected);
  });

  it("falls back to solving for a phase this client does not know", () => {
    // docs/18 §7: tolerate unknown values rather than crashing or idling.
    expect(stateAfter([progress("some_future_phase")])).toBe("solving");
  });
});

describe("synthesis is backend-aware", () => {
  it("shows connecting on a remote backend before any answer text", () => {
    expect(
      stateAfter([
        { event: "llm_config", backend: "remote" } as EngineFrame,
        progress("synthesis"),
      ]),
    ).toBe("connecting");
  });

  it("switches to composing once answer text arrives", () => {
    expect(
      stateAfter([
        { event: "llm_config", backend: "remote" } as EngineFrame,
        progress("synthesis"),
        workflow("answer_delta", { payload: { text: "xin chào" } }),
      ]),
    ).toBe("composing");
  });

  it("goes straight to composing on a local backend", () => {
    expect(
      stateAfter([{ event: "llm_config", backend: "local" } as EngineFrame, progress("synthesis")]),
    ).toBe("composing");
  });
});

describe("voice", () => {
  it("listens while the mic is capturing", () => {
    expect(stateAfter([{ event: "voice", type: "recording" } as EngineFrame])).toBe("listening");
  });

  it("treats barge-in as listening, not as playback", () => {
    expect(
      stateAfter([
        { event: "voice", type: "playback_started" } as EngineFrame,
        { event: "voice", type: "barge_in" } as EngineFrame,
      ]),
    ).toBe("listening");
  });

  it("composes while TTS plays", () => {
    expect(stateAfter([{ event: "voice", type: "playback_started" } as EngineFrame])).toBe(
      "composing",
    );
  });

  it("keeps speaking state when assistant playback telemetry arrives", () => {
    expect(
      stateAfter([
        { event: "voice", type: "playback_started" } as EngineFrame,
        {
          event: "voice",
          type: "voice_level",
          metadata: { source: "assistant", rms: 0.4 },
        } as EngineFrame,
      ]),
    ).toBe("composing");
  });

  it("stops listening once capture ends", () => {
    expect(
      fold([
        { event: "voice", type: "recording" } as EngineFrame,
        { event: "voice", type: "recorded" } as EngineFrame,
      ]).listening,
    ).toBe(false);
  });

  it("clears every activity when the loop stops", () => {
    const activity = fold([
      { event: "voice", type: "recording" } as EngineFrame,
      { event: "voice", type: "loop_stopped" } as EngineFrame,
    ]);
    expect(orbStateFor(activity)).toBe("breathing");
  });
});

describe("background jobs outrank turn phases", () => {
  it("weaves while working memory compacts", () => {
    expect(
      stateAfter([
        progress("retrieval"),
        { event: "memory_compaction", status: "running" } as EngineFrame,
      ]),
    ).toBe("weaving");
  });

  it("stops weaving when compaction publishes", () => {
    expect(
      stateAfter([
        { event: "memory_compaction", status: "running" } as EngineFrame,
        { event: "memory_compaction", status: "published" } as EngineFrame,
      ]),
    ).toBe("breathing");
  });

  it("shapes while a knowledge index builds", () => {
    expect(
      stateAfter([
        progress("tool"),
        { event: "knowledge_setup", action: "index", status: "running" } as EngineFrame,
      ]),
    ).toBe("shaping");
  });

  it("ignores knowledge_setup for init, which is not an index build", () => {
    expect(
      stateAfter([{ event: "knowledge_setup", action: "init", status: "running" } as EngineFrame]),
    ).toBe("breathing");
  });

  it("lets mic capture outrank a background index build", () => {
    // The user is mid-utterance; showing the index job instead would hide the
    // one thing they need feedback on.
    expect(
      stateAfter([
        { event: "knowledge_setup", action: "index", status: "running" } as EngineFrame,
        { event: "voice", type: "recording" } as EngineFrame,
      ]),
    ).toBe("listening");
  });
});

describe("protocol helpers", () => {
  it("discriminates the workflow envelope on protocol_version, not event name", () => {
    expect(isWorkflowFrame(workflow("some_future_workflow_event"))).toBe(true);
    expect(isWorkflowFrame({ event: "status" } as EngineFrame)).toBe(false);
  });

  it("accepts a hello that lists our version among supported ones", () => {
    expect(
      helloIsCompatible({
        event: "hello",
        version: 3,
        protocol_version: 3,
        supported_versions: [2, 3],
        profile: "baseline",
        no_model: false,
        stack: {},
      }),
    ).toBe(true);
  });

  it("rejects a hello that does not support our version", () => {
    expect(
      helloIsCompatible({
        event: "hello",
        version: 9,
        protocol_version: 9,
        supported_versions: [9],
        profile: "baseline",
        no_model: false,
        stack: {},
      }),
    ).toBe(false);
  });
});

describe("unknown frames", () => {
  it("leaves activity untouched", () => {
    expect(fold([{ event: "some_event_added_later", foo: 1 } as EngineFrame])).toEqual(
      initialActivity,
    );
  });
});

describe("voice startup", () => {
  // Measured 2026-08-16 by timestamping a live `voice_start`: the first
  // `recording` lands 9.2 s after the command, and none of the frames in
  // between used to reach this reducer.
  const voice = (type: string): EngineFrame =>
    ({ event: "voice", type, metadata: {} }) as EngineFrame;

  const afterVoice = (types: string[]) => stateAfter(types.map(voice));

  it("does not rest while the runtime loads", () => {
    // `breathing` here is the bug: nine seconds of loading that reads as
    // "ready and ignoring you".
    expect(afterVoice(["loading"])).not.toBe("breathing");
    expect(afterVoice(["loading", "ready"])).not.toBe("breathing");
    expect(afterVoice(["loading", "ready", "warmup"])).not.toBe("breathing");
  });

  it("settles only once the loop actually started", () => {
    expect(afterVoice(["loading", "ready", "warmup", "loop_started"])).toBe("breathing");
  });

  it("shows listening once capture opens", () => {
    expect(afterVoice(["loading", "warmup", "loop_started", "recording"])).toBe("listening");
  });

  it("clears the loading state if voice fails to start", () => {
    expect(afterVoice(["loading", "error"])).toBe("breathing");
    expect(afterVoice(["loading", "loop_stopped"])).toBe("breathing");
  });
});
