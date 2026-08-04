import { describe, expect, it } from "vitest";
import type { TurnProgressEvent } from "./protocol.js";
import { initialState, reduce } from "./store.js";

function progress(
  phase: TurnProgressEvent["phase"],
  operation: string = phase,
): TurnProgressEvent {
  return {
    event: "turn_progress",
    surface: "chat",
    phase,
    operation,
    status: "active",
  };
}

describe("progress reducer", () => {
  it("queues rapid real stages instead of overwriting them with synthesis", () => {
    let state = reduce(initialState, {
      type: "engine_event",
      event: progress("analyzing"),
    });
    state = reduce(state, {
      type: "engine_event",
      event: progress("routing"),
    });
    state = reduce(state, {
      type: "engine_event",
      event: progress("memory"),
    });
    state = reduce(state, {
      type: "engine_event",
      event: progress("retrieval", "tool:knowledge.search"),
    });
    state = reduce(state, {
      type: "engine_event",
      event: progress("synthesis", "llm"),
    });

    expect(state.turnProgress?.phase).toBe("analyzing");
    expect(state.progressQueue.map((event) => event.phase)).toEqual([
      "routing",
      "memory",
      "retrieval",
      "synthesis",
    ]);

    state = reduce(state, { type: "advance_progress" });
    expect(state.turnProgress?.phase).toBe("routing");
    expect(state.progressQueue.map((event) => event.phase)).toEqual([
      "memory",
      "retrieval",
      "synthesis",
    ]);
  });

  it("deduplicates adjacent phases and clears the queue when done", () => {
    let state = reduce(initialState, {
      type: "engine_event",
      event: progress("analyzing", "normalize_input"),
    });
    state = reduce(state, {
      type: "engine_event",
      event: progress("analyzing", "input_guardrail"),
    });
    expect(state.turnProgress?.operation).toBe("input_guardrail");
    expect(state.progressQueue).toEqual([]);

    state = reduce(state, {
      type: "engine_event",
      event: { ...progress("complete"), status: "done" },
    });
    expect(state.turnProgress).toBeNull();
    expect(state.progressQueue).toEqual([]);
  });

  it("ignores stale progress from an older run and retains a failed terminal", () => {
    const event = (
      run_id: string,
      sequence: number,
      status: TurnProgressEvent["status"],
      phase: TurnProgressEvent["phase"],
    ): TurnProgressEvent => ({
      event: "turn_progress",
      surface: "chat",
      phase,
      operation: phase,
      status,
      run_id,
      goal_id: `goal-${run_id}`,
      sequence,
      terminal_status: status === "failed" ? "system_failure" : undefined,
    });

    let state = reduce(initialState, {
      type: "engine_event",
      event: event("run-old", 3, "active", "retrieval"),
    });
    state = reduce(state, {
      type: "engine_event",
      event: event("run-new", 0, "active", "analyzing"),
    });
    state = reduce(state, {
      type: "engine_event",
      event: event("run-old", 4, "active", "synthesis"),
    });
    expect(state.turnProgress?.run_id).toBe("run-new");
    expect(state.turnProgress?.phase).toBe("analyzing");

    state = reduce(state, {
      type: "engine_event",
      event: event("run-new", 1, "failed", "complete"),
    });
    expect(state.turnProgress?.status).toBe("failed");
    expect(state.progressQueue).toEqual([]);
  });

  it("keeps provisional workflow text separate until the chat terminal arrives", () => {
    let state = reduce(initialState, {
      type: "engine_event",
      event: {
        event: "answer_delta",
        protocol_version: 2,
        session_id: "session",
        run_id: "run",
        goal_id: "goal",
        sequence: 1,
        surface: "chat",
        timestamp: "2026-07-30T00:00:00Z",
        node: "synthesize",
        status: "active",
        payload: { text: "đang tổng hợp" },
      },
    });
    expect(state.pendingAnswer).toBe("đang tổng hợp");
    state = reduce(state, {
      type: "engine_event",
      event: {
        event: "turn_terminal",
        protocol_version: 2,
        session_id: "session",
        run_id: "run",
        goal_id: "goal",
        sequence: 2,
        surface: "chat",
        timestamp: "2026-07-30T00:00:01Z",
        node: "finalize",
        status: "completed",
        payload: { terminal_status: "achieved", final_text: "xong" },
      },
    });
    expect(state.pendingAnswer).toBe("");
    expect(state.workflowEvents).toHaveLength(2);
  });

  it("does not duplicate voice draft text mirrored by workflow events", () => {
    let state = reduce(initialState, {
      type: "engine_event",
      event: {
        event: "voice",
        type: "llm_token",
        text: "xin chào",
        latency_ms: null,
        metadata: {},
        usage: null,
      },
    });
    state = reduce(state, {
      type: "engine_event",
      event: {
        event: "answer_delta",
        protocol_version: 2,
        session_id: "session",
        run_id: "voice-run",
        goal_id: "voice-goal",
        sequence: 1,
        surface: "voice",
        timestamp: "2026-07-30T00:00:00Z",
        node: "synthesize",
        status: "active",
        payload: { text: "xin chào" },
      },
    });

    expect(state.pendingAnswer).toBe("xin chào");
  });

  it("tracks queued, playing, and completed speech chunks from playback events", () => {
    let state = reduce(initialState, {
      type: "engine_event",
      event: {
        event: "voice",
        type: "tts",
        text: "Câu đầu tiên.",
        latency_ms: 40,
        metadata: { chunk_index: 0, delivery: "final" },
        usage: null,
      },
    });
    state = reduce(state, {
      type: "engine_event",
      event: {
        event: "voice",
        type: "playback_started",
        text: "Câu đầu tiên.",
        latency_ms: null,
        metadata: {
          chunk_index: 0,
          delivery: "final",
          audio_duration_ms: 1200,
          sync_granularity: "audio_chunk",
        },
        usage: null,
      },
    });
    state = reduce(state, {
      type: "engine_event",
      event: {
        event: "voice",
        type: "tts",
        text: "Câu tiếp theo.",
        latency_ms: 35,
        metadata: { chunk_index: 1, delivery: "final" },
        usage: null,
      },
    });

    expect(state.voiceState).toBe("speaking");
    expect(state.speechChunks).toEqual([
      {
        index: 0,
        text: "Câu đầu tiên.",
        durationMs: 1200,
        status: "playing",
      },
      {
        index: 1,
        text: "Câu tiếp theo.",
        durationMs: null,
        status: "ready",
      },
    ]);

    state = reduce(state, {
      type: "engine_event",
      event: {
        event: "voice",
        type: "audio",
        text: "Câu đầu tiên.",
        latency_ms: 1200,
        metadata: { chunk_index: 0, delivery: "final" },
        usage: null,
      },
    });
    // The session sink reports `audio` when PCM is accepted, not when the
    // speaker has finished draining it. Keep the chunk active until the next
    // playback_started event advances the cursor.
    expect(state.speechChunks[0]?.status).toBe("playing");
    state = reduce(state, {
      type: "engine_event",
      event: {
        event: "voice",
        type: "playback_started",
        text: "Câu tiếp theo.",
        latency_ms: null,
        metadata: {
          chunk_index: 1,
          audio_duration_ms: 900,
        },
        usage: null,
      },
    });
    expect(state.speechChunks[0]?.status).toBe("complete");

    // A delayed receipt must not move a completed chunk back to the active
    // color state.
    state = reduce(state, {
      type: "engine_event",
      event: {
        event: "voice",
        type: "audio",
        text: "Câu đầu tiên.",
        latency_ms: 1200,
        metadata: { chunk_index: 0 },
        usage: null,
      },
    });
    expect(state.speechChunks[0]?.status).toBe("complete");

    state = reduce(state, {
      type: "engine_event",
      event: {
        event: "voice",
        type: "barge_in",
        text: "Barge-in detected",
        latency_ms: null,
        metadata: { phase: "fired" },
        usage: null,
      },
    });
    expect(state.speechChunks).toEqual([]);
  });

  it("keeps the speaking state when tokens stream during playback", () => {
    let state = reduce(initialState, {
      type: "engine_event",
      event: {
        event: "voice",
        type: "playback_started",
        text: "Câu đang phát.",
        latency_ms: null,
        metadata: { chunk_index: 0, audio_duration_ms: 900 },
        usage: null,
      },
    });
    expect(state.voiceState).toBe("speaking");

    // The next sentence is still being generated while chunk 0 plays.
    state = reduce(state, {
      type: "engine_event",
      event: {
        event: "voice",
        type: "llm_token",
        text: "tiếp ",
        latency_ms: null,
        metadata: {},
        usage: null,
      },
    });
    expect(state.voiceState).toBe("speaking");
    expect(state.speechChunks).toHaveLength(1);

    // A buffered `audio` event does not mean the speaker is done.
    state = reduce(state, {
      type: "engine_event",
      event: {
        event: "voice",
        type: "audio",
        text: "Câu đang phát.",
        latency_ms: 900,
        metadata: { chunk_index: 0 },
        usage: null,
      },
    });
    state = reduce(state, {
      type: "engine_event",
      event: {
        event: "voice",
        type: "llm_token",
        text: "nữa ",
        latency_ms: null,
        metadata: {},
        usage: null,
      },
    });
    expect(state.voiceState).toBe("speaking");
  });

  it("drops the speech caption when playback is interrupted", () => {
    let state = reduce(initialState, {
      type: "engine_event",
      event: {
        event: "voice",
        type: "tts",
        text: "Câu bị cắt ngang.",
        latency_ms: 40,
        metadata: { chunk_index: 0, delivery: "final" },
        usage: null,
      },
    });
    expect(state.speechChunks).toHaveLength(1);

    state = reduce(state, {
      type: "engine_event",
      event: {
        event: "voice",
        type: "interrupted",
        text: "",
        latency_ms: null,
        metadata: {},
        usage: null,
      },
    });
    expect(state.speechChunks).toEqual([]);
  });

  it("keeps citations as protocol data and closes temporary info on a new turn", () => {
    let state = reduce(
      { ...initialState, activeInfo: "status" },
      { type: "user_message", text: "attention" },
    );
    expect(state.activeInfo).toBeNull();

    state = reduce(state, {
      type: "engine_event",
      event: {
        event: "chat",
        type: "done",
        text: "Attention dùng query, key và value.",
        citations: [
          {
            label: "K1",
            path: "wiki/learning/attention.md",
            title: "Attention",
            line_start: 12,
            line_end: 18,
            source: "knowledge",
          },
        ],
      },
    });

    expect(state.timeline.at(-1)).toMatchObject({
      kind: "soca",
      text: "Attention dùng query, key và value.",
      citations: [
        {
          label: "K1",
          path: "wiki/learning/attention.md",
          source: "knowledge",
        },
      ],
    });
  });
});

describe("voice ASR status", () => {
  it("shows the concrete backend loading message", () => {
    const state = reduce(initialState, {
      type: "engine_event",
      event: {
        event: "voice",
        type: "loading",
        text: "Loading Qwen ASR…",
        latency_ms: null,
        metadata: {},
        usage: null,
      },
    });

    expect(state.voiceState).toBe("loading");
    expect(state.voiceNote).toBe("Loading Qwen ASR…");
  });

  it("shows transcription separately from recording", () => {
    const state = reduce(initialState, {
      type: "engine_event",
      event: {
        event: "voice",
        type: "transcribing",
        text: "Transcribing",
        latency_ms: null,
        metadata: {},
        usage: null,
      },
    });

    expect(state.voiceState).toBe("processing");
    expect(state.voiceNote).toBe("Transcribing…");
  });

  it("explains an automatic stop after passive-silence callouts", () => {
    const state = reduce(initialState, {
      type: "engine_event",
      event: {
        event: "voice",
        type: "loop_stopped",
        text: "Voice loop stopped",
        latency_ms: null,
        metadata: {
          turns: 3,
          requested: true,
          stop_reason: "passive_silence_callout_limit",
        },
        usage: null,
      },
    });

    expect(state.voiceRunning).toBe(false);
    expect(state.voiceNote).toBe("tự dừng sau 3 lần nhắc vì im lặng");
  });
});
