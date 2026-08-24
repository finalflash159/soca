import { describe, expect, it } from "vitest";

import type { EngineFrame } from "./protocol";
import {
  initialVoice,
  LEVEL_HISTORY,
  partialText,
  peakLevel,
  reduceVoice,
  voicePhaseLabel,
  type VoicePhase,
} from "./voice";

const voice = (type: string, extra: Record<string, unknown> = {}): EngineFrame =>
  ({ event: "voice", type, ...extra }) as EngineFrame;

function fold(frames: EngineFrame[]) {
  return frames.reduce(reduceVoice, initialVoice);
}

describe("loop lifecycle", () => {
  it("starts off", () => {
    expect(initialVoice.phase).toBe("off");
  });

  it("records the profile when the loop starts", () => {
    const state = fold([voice("loop_started", { metadata: { profile: "qwen-release" } })]);
    expect(state.phase).toBe("idle");
    expect(state.profile).toBe("qwen-release");
  });

  it("clears live signals but keeps counters when the loop stops", () => {
    const state = fold([
      voice("loop_started", { metadata: { profile: "p" } }),
      voice("recording", { metadata: {} }),
      voice("voice_level", { metadata: { rms: 0.4 } }),
      voice("turn_end", { metadata: {} }),
      voice("loop_stopped"),
    ]);
    expect(state.phase).toBe("off");
    expect(state.levels).toEqual([]);
    expect(state.turnCount).toBe(1);
    expect(state.profile).toBe("p");
  });

  it("surfaces an error and leaves the loop off", () => {
    const state = fold([voice("error", { text: "device gone" })]);
    expect(state.phase).toBe("off");
    expect(state.error).toBe("device gone");
  });
});

describe("visible labels", () => {
  it("uses consistent English names for every voice state", () => {
    const labels: Record<VoicePhase, string> = {
      off: "Off",
      starting: "Starting",
      idle: "Ready",
      listening: "Listening",
      transcribing: "Transcribing",
      thinking: "Thinking",
      speaking: "Speaking",
    };

    for (const [phase, label] of Object.entries(labels) as [VoicePhase, string][]) {
      expect(voicePhaseLabel(phase)).toBe(label);
    }
  });
});

describe("levels", () => {
  it("appends rms readings from the engine", () => {
    const state = fold([
      voice("voice_level", { metadata: { rms: 0.1 } }),
      voice("voice_level", { metadata: { rms: 0.5 } }),
    ]);
    expect(state.levels).toEqual([0.1, 0.5]);
  });

  it("clamps out-of-range readings instead of trusting them", () => {
    const state = fold([
      voice("voice_level", { metadata: { rms: 1.8 } }),
      voice("voice_level", { metadata: { rms: -0.2 } }),
    ]);
    expect(state.levels).toEqual([1, 0]);
  });

  it("ignores a reading with no numeric rms", () => {
    expect(fold([voice("voice_level", { metadata: { rms: "loud" } })]).levels).toEqual([]);
  });

  it("caps the retained window", () => {
    const frames = Array.from({ length: LEVEL_HISTORY + 20 }, (_, index) =>
      voice("voice_level", { metadata: { rms: index / 1000 } }),
    );
    const state = fold(frames);
    expect(state.levels).toHaveLength(LEVEL_HISTORY);
    // The oldest readings are the ones dropped.
    expect(state.levels[state.levels.length - 1]).toBeCloseTo((LEVEL_HISTORY + 19) / 1000);
  });

  it("resets the window once capture ends", () => {
    const state = fold([voice("voice_level", { metadata: { rms: 0.3 } }), voice("recorded")]);
    expect(state.levels).toEqual([]);
  });

  it("reports peak rather than mean", () => {
    // A mean over a window containing silence makes a working mic look dead.
    expect(peakLevel([0, 0, 0.8, 0])).toBe(0.8);
    expect(peakLevel([])).toBe(0);
  });
});

describe("endpoint configuration", () => {
  it("captures floor, ceiling and detector flags from the recording frame", () => {
    const state = fold([
      voice("recording", {
        metadata: {
          asr_model: "qwen3_asr_0_6b",
          smart_turn_enabled: true,
          adaptive_endpoint: true,
          endpoint_floor_ms: 1000,
          endpoint_ceil_ms: 2400,
        },
      }),
    ]);
    expect(state.phase).toBe("listening");
    expect(state.asrModel).toBe("qwen3_asr_0_6b");
    expect(state.endpoint).toEqual({
      adaptive: true,
      floorMs: 1000,
      ceilMs: 2400,
      smartTurn: true,
    });
  });

  it("keeps nulls when the engine omits the numbers", () => {
    const state = fold([voice("recording", { metadata: {} })]);
    expect(state.endpoint).toEqual({
      adaptive: false,
      floorMs: null,
      ceilMs: null,
      smartTurn: false,
    });
  });
});

describe("partial transcript", () => {
  it("splits committed from tentative", () => {
    const state = fold([
      voice("asr_partial", { metadata: { committed: "hôm nay", tentative: "trời" } }),
    ]);
    expect(partialText(state.partial)).toBe("hôm nay trời");
  });

  it("is cleared when a new capture starts", () => {
    const state = fold([
      voice("asr_partial", { metadata: { committed: "cũ", tentative: "" } }),
      voice("recording", { metadata: {} }),
    ]);
    expect(state.partial).toBeNull();
    expect(partialText(null)).toBe("");
  });
});

describe("barge-in", () => {
  it("arms before it fires", () => {
    const state = fold([voice("barge_in", { metadata: { phase: "armed" } })]);
    expect(state.bargeIn).toBe("armed");
  });

  it("returns the floor to the user when it fires", () => {
    const state = fold([
      voice("playback_started"),
      voice("barge_in", { metadata: { phase: "armed" } }),
      voice("barge_in", { metadata: { phase: "fired", source: "duplex_aec" } }),
    ]);
    expect(state.bargeIn).toBe("fired");
    expect(state.phase).toBe("listening");
  });

  it("disarms at the end of the turn", () => {
    const state = fold([
      voice("barge_in", { metadata: { phase: "fired" } }),
      voice("turn_end", { metadata: {} }),
    ]);
    expect(state.bargeIn).toBe("idle");
  });
});

describe("repair", () => {
  it("is a turn, not an error", () => {
    const state = fold([voice("repair", { text: "Bạn nói lại giúp mình nhé?" })]);
    expect(state.error).toBeNull();
    expect(state.repairPrompt).toBe("Bạn nói lại giúp mình nhé?");
    expect(state.phase).toBe("speaking");
  });

  it("is cleared when the next capture starts", () => {
    const state = fold([
      voice("repair", { text: "nói lại nhé" }),
      voice("recording", { metadata: {} }),
    ]);
    expect(state.repairPrompt).toBeNull();
  });
});

describe("turn outcome", () => {
  it("records rejection and terminal status", () => {
    const state = fold([
      voice("done", { metadata: { rejected: true, terminal_status: "cancelled" } }),
    ]);
    expect(state.lastTurn).toEqual({
      rejected: true,
      terminalStatus: "cancelled",
      durationS: null,
    });
  });
});

describe("phase transitions", () => {
  it("does not let a progress event pull the phase out of listening", () => {
    const state = fold([
      voice("recording", { metadata: {} }),
      voice("progress", { metadata: { stage: "llm" } }),
    ]);
    expect(state.phase).toBe("listening");
  });

  it("moves to thinking on progress once capture has ended", () => {
    const state = fold([
      voice("recorded", { metadata: {} }),
      voice("progress", { metadata: { stage: "llm" } }),
    ]);
    expect(state.phase).toBe("thinking");
  });
});

describe("non-voice frames", () => {
  it("are ignored", () => {
    expect(fold([{ event: "status" } as EngineFrame])).toEqual(initialVoice);
  });
});

describe("startup phases", () => {
  const voice = (type: string): EngineFrame =>
    ({ event: "voice", type, metadata: {} }) as EngineFrame;

  it("does not report idle before the loop exists", () => {
    // `ready` is a component signal and lands 2.2 s before `loop_started`.
    // Calling that "waiting" invites the user to talk into a closed mic.
    const state = [voice("loading"), voice("ready")].reduce(reduceVoice, initialVoice);
    expect(state.phase).toBe("starting");
  });

  it("reports idle once the loop started", () => {
    const state = [voice("loading"), voice("ready"), voice("loop_started")].reduce(
      reduceVoice,
      initialVoice,
    );
    expect(state.phase).toBe("idle");
  });
});
