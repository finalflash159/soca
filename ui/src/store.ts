import type { EngineEvent } from "./protocol.js";

export type Mode = "chat" | "voice" | "status";
export type VoiceState =
  "loading" | "idle" | "listening" | "processing" | "speaking" | "error";

export interface TimelineEntry {
  kind: "user" | "soca" | "system" | "error";
  text: string;
}

export interface StatusProfile {
  key: string;
  status: string;
  asr: string;
  llm: string;
  tts: string;
  voice: string | null;
}

export interface AppState {
  mode: Mode;
  connected: boolean;
  profile: string;
  noModel: boolean;
  stack: Record<string, string | null>;
  timeline: TimelineEntry[];
  voiceState: VoiceState;
  voiceNote: string;
  voiceRunning: boolean;
  turnIndex: number | null;
  lastLatencyMs: number | null;
  lastRoute: string;
  chatBusy: boolean;
  profiles: StatusProfile[];
  notice: string;
}

export const initialState: AppState = {
  mode: "chat",
  connected: false,
  profile: "",
  noModel: false,
  stack: {},
  timeline: [],
  voiceState: "idle",
  voiceNote: "",
  voiceRunning: false,
  turnIndex: null,
  lastLatencyMs: null,
  lastRoute: "",
  chatBusy: false,
  profiles: [],
  notice: "",
};

export type Action =
  | { type: "engine_event"; event: EngineEvent }
  | { type: "set_mode"; mode: Mode }
  | { type: "user_message"; text: string }
  | { type: "system_message"; text: string }
  | { type: "voice_started" }
  | { type: "clear_timeline" };

function push(
  timeline: TimelineEntry[],
  entry: TimelineEntry,
): TimelineEntry[] {
  return [...timeline.slice(-199), entry];
}

// Voice event type -> UI state, mirroring the Textual _VOICE_RENDERERS table.
function reduceVoice(
  state: AppState,
  event: EngineEvent & { event: "voice" },
): AppState {
  const meta = event.metadata ?? {};
  switch (event.type) {
    case "loading":
      return { ...state, voiceState: "loading", voiceNote: "tải ASR/LLM/TTS…" };
    case "ready":
    case "warmup":
      return { ...state, voiceState: "loading", voiceNote: "khởi động…" };
    case "loop_started":
      return {
        ...state,
        voiceRunning: true,
        voiceState: "listening",
        voiceNote: "",
      };
    case "turn_start": {
      const turn =
        typeof meta["turn_index"] === "number"
          ? (meta["turn_index"] as number)
          : null;
      return {
        ...state,
        voiceState: "listening",
        voiceNote: "",
        turnIndex: turn,
      };
    }
    case "recording":
      return { ...state, voiceState: "listening", voiceNote: "đang nghe…" };
    case "recorded":
      return { ...state, voiceState: "processing", voiceNote: "" };
    case "asr":
      return {
        ...state,
        voiceState: "processing",
        timeline: push(state.timeline, {
          kind: "user",
          text: event.text || "<trống>",
        }),
      };
    case "repair":
      return {
        ...state,
        voiceState: "processing",
        voiceNote: "follow-up",
        timeline: push(state.timeline, {
          kind: "system",
          text: `Follow-up: ${event.text}`,
        }),
      };
    case "llm_token":
      return { ...state, voiceState: "processing" };
    case "tts":
      return { ...state, voiceState: "speaking" };
    case "done": {
      const rejected = Boolean(meta["rejected"]);
      const route = String(meta["runtime_route"] ?? state.lastRoute ?? "");
      const next: AppState = {
        ...state,
        voiceState: "idle",
        voiceNote: "",
        lastRoute: route,
        lastLatencyMs: event.latency_ms,
      };
      if (event.text && !rejected) {
        next.timeline = push(state.timeline, {
          kind: "soca",
          text: event.text,
        });
      }
      return next;
    }
    case "turn_end":
      return { ...state, voiceState: "idle" };
    case "loop_stopped": {
      const turns = meta["turns"];
      return {
        ...state,
        voiceRunning: false,
        voiceState: "idle",
        voiceNote: `đã dừng (${typeof turns === "number" ? turns : 0} lượt)`,
      };
    }
    case "error":
      return {
        ...state,
        voiceState: "error",
        voiceNote: "lỗi",
        voiceRunning: false,
        timeline: push(state.timeline, { kind: "error", text: event.text }),
      };
    default:
      return state;
  }
}

function reduceEngineEvent(state: AppState, event: EngineEvent): AppState {
  switch (event.event) {
    case "hello":
      return {
        ...state,
        connected: true,
        profile: event.profile,
        noModel: event.no_model,
        stack: event.stack,
      };
    case "voice":
      return reduceVoice(state, event);
    case "chat":
      switch (event.type) {
        case "loading":
          return { ...state, notice: "đang build text runtime…" };
        case "ready":
          return { ...state, notice: "" };
        case "done":
          return {
            ...state,
            chatBusy: false,
            lastRoute: event.route ?? "",
            timeline: push(state.timeline, {
              kind: "soca",
              text: event.text ?? "",
            }),
          };
        case "error":
          return {
            ...state,
            chatBusy: false,
            timeline: push(state.timeline, {
              kind: "error",
              text: event.text ?? "lỗi chat",
            }),
          };
        default:
          return state;
      }
    case "status":
      return { ...state, profiles: event.profiles };
    case "memory": {
      const text = event.enabled
        ? event.text
          ? `session memory:\n${event.text}`
          : "session memory: (trống)"
        : "session memory: đang tắt (--no-memory)";
      return {
        ...state,
        timeline: push(state.timeline, { kind: "system", text }),
      };
    }
    case "usage": {
      const text =
        `phiên: ${event.turns} lượt (${event.llm_turns} LLM) ` +
        `· prompt ${event.prompt_tokens} tok · completion ${event.completion_tokens} tok ` +
        `· TTFT ~${Math.round(event.mean_ttft_ms)}ms · ${event.mean_tokens_per_second.toFixed(1)} tok/s`;
      return {
        ...state,
        timeline: push(state.timeline, { kind: "system", text }),
      };
    }
    case "engine_error":
      return {
        ...state,
        timeline: push(state.timeline, { kind: "error", text: event.message }),
      };
    case "bye":
      return { ...state, connected: false };
    default:
      return state;
  }
}

export function reduce(state: AppState, action: Action): AppState {
  switch (action.type) {
    case "engine_event":
      return reduceEngineEvent(state, action.event);
    case "set_mode":
      return { ...state, mode: action.mode };
    case "user_message":
      return {
        ...state,
        chatBusy: true,
        timeline: push(state.timeline, { kind: "user", text: action.text }),
      };
    case "system_message":
      return {
        ...state,
        timeline: push(state.timeline, { kind: "system", text: action.text }),
      };
    case "voice_started":
      return {
        ...state,
        voiceState: "loading",
        voiceNote: "khởi động voice loop…",
      };
    case "clear_timeline":
      return { ...state, timeline: [] };
    default:
      return state;
  }
}
