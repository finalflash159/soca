import type {
  CitationRecord,
  ContextEvent,
  EngineEvent,
  LlmConfigEvent,
  MemoryCompactionEvent,
  MemoryEvent,
  MemoryTraceEvent,
  RemoteModelEvent,
  TurnProgressEvent,
  UsageEvent,
  RuntimeComponentStatus,
  WorkflowEvent,
} from "./protocol.js";

export type Mode = "chat" | "voice" | "status" | "settings";
export type InteractiveMode = "chat" | "voice" | "settings";
export type InfoView =
  | "status"
  | "context"
  | "memory"
  | "compaction"
  | "compacted_summary";
export type VoiceState =
  "loading" | "idle" | "listening" | "processing" | "speaking" | "error";

export interface TimelineEntry {
  kind: "user" | "soca" | "system" | "error";
  text: string;
  citations?: CitationRecord[];
  latencyMs?: number;
}

export interface MemoryProposal {
  id: string;
  kind: "preference" | "stable_fact" | "project" | "correction";
  statement: string;
  confidence: number;
  createdAt: string;
}

export interface RetrievalTrace {
  query: string;
  tier: "deterministic" | "semantic" | "llm" | "none";
  latencyMs: number;
  columns: Array<{
    source: string;
    hits: Array<{
      path: string;
      score: number;
      sparse_score?: number;
      dense_score?: number;
      fusion_score?: number;
    }>;
  }>;
  fused: Array<{
    path: string;
    picked: boolean;
    backend?: string;
    score?: number;
  }>;
  rejectedCount: number;
  evidence: Record<string, unknown> | null;
}

export interface StatusProfile {
  key: string;
  status: string;
  asr: string;
  llm: string;
  tts: string;
  voice: string | null;
}

export interface LlmProviderStatus {
  key: string;
  label: string;
  has_key: boolean;
  has_pricing_api: boolean;
}

export interface KnowledgeIndexStatus {
  vault_path?: string;
  sparse_state: string;
  dense_state: string;
  revision: number;
  documents: number;
  chunks: number;
}

// Live partial transcript while the user is still speaking: committed words are
// stable (LocalAgreement), tentative words may still change on the next decode.
export interface Caption {
  committed: string;
  tentative: string;
}

export interface SpeechChunk {
  index: number;
  text: string;
  durationMs: number | null;
  status: "ready" | "playing" | "complete";
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
  caption: Caption | null;
  speechChunks: SpeechChunk[];
  voiceLevel: number;
  bargeIn: "off" | "armed" | "fired";
  routerTier: "deterministic" | "semantic" | "llm" | "none";
  routerLatencyMs: number;
  memoryMode: "blob" | "retrieved" | "degraded";
  memoryHits: number;
  proposals: MemoryProposal[];
  proposalsOpen: boolean;
  memoryActionError: string;
  retrievalTrace: RetrievalTrace | null;
  llmProviders: LlmProviderStatus[];
  llmCatalog: RemoteModelEvent[];
  llmCatalogProvider: string;
  llmConfig: LlmConfigEvent | null;
  llmKeyPendingProvider: string | null;
  settingsNotice: string;
  knowledgeIndex: KnowledgeIndexStatus | null;
  runtimeComponents: RuntimeComponentStatus[];
  activeInfo: InfoView | null;
  context: ContextEvent | null;
  memorySnapshot: MemoryEvent | null;
  usageSnapshot: UsageEvent | null;
  memoryCompaction: MemoryCompactionEvent | null;
  turnProgress: TurnProgressEvent | null;
  progressQueue: TurnProgressEvent[];
  progressRunId: string | null;
  pendingAnswer: string;
  workflowEvents: WorkflowEvent[];
  workflowTerminalStatus: string | null;
  memoryTelemetry: MemoryTraceEvent | null;
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
  caption: null,
  speechChunks: [],
  voiceLevel: 0,
  bargeIn: "off",
  routerTier: "none",
  routerLatencyMs: 0,
  memoryMode: "blob",
  memoryHits: 0,
  proposals: [],
  proposalsOpen: false,
  memoryActionError: "",
  retrievalTrace: null,
  llmProviders: [],
  llmCatalog: [],
  llmCatalogProvider: "",
  llmConfig: null,
  llmKeyPendingProvider: null,
  settingsNotice: "",
  knowledgeIndex: null,
  runtimeComponents: [],
  activeInfo: null,
  context: null,
  memorySnapshot: null,
  usageSnapshot: null,
  memoryCompaction: null,
  turnProgress: null,
  progressQueue: [],
  progressRunId: null,
  pendingAnswer: "",
  workflowEvents: [],
  workflowTerminalStatus: null,
  memoryTelemetry: null,
};

export type Action =
  | { type: "engine_event"; event: EngineEvent }
  | { type: "set_mode"; mode: Mode }
  | { type: "user_message"; text: string }
  | { type: "system_message"; text: string }
  | { type: "voice_started" }
  | { type: "show_info"; view: InfoView }
  | { type: "clear_info" }
  | { type: "clear_timeline" }
  | { type: "clear_proposals" }
  | { type: "advance_progress" };

function push(
  timeline: TimelineEntry[],
  entry: TimelineEntry,
): TimelineEntry[] {
  return [...timeline.slice(-199), entry];
}

function citationRecords(value: unknown): CitationRecord[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is CitationRecord => {
    if (typeof item !== "object" || item === null) return false;
    const record = item as Record<string, unknown>;
    return (
      typeof record["label"] === "string" &&
      typeof record["path"] === "string" &&
      typeof record["title"] === "string" &&
      (record["source"] === "knowledge" || record["source"] === "memory")
    );
  });
}

function speechChunkIndex(meta: Record<string, unknown>): number | null {
  const value = meta["chunk_index"];
  return typeof value === "number" && Number.isInteger(value) && value >= 0
    ? value
    : null;
}

function upsertSpeechChunk(
  chunks: SpeechChunk[],
  event: EngineEvent & { event: "voice" },
  status: SpeechChunk["status"],
): SpeechChunk[] {
  const meta = event.metadata ?? {};
  const index = speechChunkIndex(meta);
  if (index === null || !event.text) return chunks;
  const durationValue = meta["audio_duration_ms"];
  const durationMs =
    typeof durationValue === "number" && Number.isFinite(durationValue)
      ? Math.max(0, durationValue)
      : null;
  const previous = chunks.find((chunk) => chunk.index === index);
  const next: SpeechChunk = {
    index,
    text: event.text,
    durationMs: durationMs ?? previous?.durationMs ?? null,
    status:
      status === "ready" && previous && previous.status !== "ready"
        ? previous.status
        : status,
  };
  return [...chunks.filter((chunk) => chunk.index !== index), next].sort(
    (left, right) => left.index - right.index,
  );
}

// Voice event type -> UI state, mirroring the Textual _VOICE_RENDERERS table.
function reduceVoiceCore(
  state: AppState,
  event: EngineEvent & { event: "voice" },
): AppState {
  const meta = event.metadata ?? {};
  switch (event.type) {
    case "asr_partial":
      return {
        ...state,
        voiceState: "listening",
        caption: {
          committed: String(meta["committed"] ?? ""),
          tentative: String(meta["tentative"] ?? ""),
        },
      };
    case "voice_level":
      return {
        ...state,
        voiceLevel: Math.max(0, Math.min(1, Number(meta["rms"] ?? 0))),
      };
    case "barge_in":
      return {
        ...state,
        bargeIn: meta["phase"] === "fired" ? "fired" : "armed",
        voiceState: meta["phase"] === "fired" ? "listening" : state.voiceState,
        speechChunks: meta["phase"] === "fired" ? [] : state.speechChunks,
      };
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
        bargeIn: "armed",
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
        speechChunks: [],
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
      return {
        ...state,
        // Tokens for the next sentence can arrive while a chunk is still
        // playing; downgrading to "processing" there would hide the speaking
        // row and its caption mid-playback.
        voiceState: state.speechChunks.some(
          (chunk) => chunk.status === "playing",
        )
          ? state.voiceState
          : "processing",
      };
    case "tts":
      return {
        ...state,
        voiceState: "speaking",
        speechChunks: upsertSpeechChunk(state.speechChunks, event, "ready"),
      };
    case "playback_started":
      return {
        ...state,
        voiceState: "speaking",
        speechChunks: upsertSpeechChunk(state.speechChunks, event, "playing"),
      };
    case "audio":
      return {
        ...state,
        speechChunks: upsertSpeechChunk(state.speechChunks, event, "complete"),
      };
    case "interrupted":
      return { ...state, speechChunks: [] };
    case "done": {
      const rejected = Boolean(meta["rejected"]);
      const route = String(meta["runtime_route"] ?? state.lastRoute ?? "");
      const next: AppState = {
        ...state,
        voiceState: "idle",
        voiceNote: "",
        pendingAnswer: "",
        lastRoute: route,
        lastLatencyMs: event.latency_ms,
        bargeIn: "armed",
        speechChunks: [],
      };
      if (event.text && !rejected) {
        next.timeline = push(state.timeline, {
          kind: "soca",
          text: event.text,
          citations: citationRecords(meta["citations"]),
          latencyMs: event.latency_ms ?? undefined,
        });
      }
      return next;
    }
    case "turn_end":
      return { ...state, voiceState: "idle", speechChunks: [] };
    case "loop_stopped": {
      const turns = meta["turns"];
      return {
        ...state,
        voiceRunning: false,
        voiceState: "idle",
        turnProgress: null,
        progressQueue: [],
        voiceNote: `đã dừng (${typeof turns === "number" ? turns : 0} lượt)`,
        bargeIn: "off",
        speechChunks: [],
      };
    }
    case "error":
      return {
        ...state,
        voiceState: "error",
        voiceNote: "lỗi",
        voiceRunning: false,
        turnProgress: null,
        progressQueue: [],
        speechChunks: [],
        timeline: push(state.timeline, { kind: "error", text: event.text }),
      };
    default:
      return state;
  }
}

// Any voice event other than "asr_partial" ends the live caption, so the
// partial transcript never lingers past the moment the user stops speaking.
function reduceVoice(
  state: AppState,
  event: EngineEvent & { event: "voice" },
): AppState {
  const next = reduceVoiceCore(state, event);
  if (event.type === "asr_partial" || next.caption === null) return next;
  return { ...next, caption: null };
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
    case "turn_progress": {
      const current = state.turnProgress;
      if (
        state.progressRunId &&
        event.run_id &&
        state.progressRunId !== event.run_id
      ) {
        if (event.sequence !== 0) return state;
        return {
          ...state,
          progressRunId: event.run_id,
          turnProgress: event,
          progressQueue: [],
        };
      }
      if (
        current?.run_id &&
        event.run_id &&
        current.run_id === event.run_id &&
        current.sequence !== undefined &&
        event.sequence !== undefined &&
        event.sequence <= current.sequence
      ) {
        return state;
      }
      if (current?.run_id && event.run_id && current.run_id !== event.run_id) {
        return {
          ...state,
          progressRunId: event.run_id ?? state.progressRunId,
          turnProgress: event,
          progressQueue: [],
        };
      }
      if (event.status === "failed" || event.status === "cancelled") {
        return {
          ...state,
          progressRunId: event.run_id ?? state.progressRunId,
          turnProgress: event,
          progressQueue: [],
        };
      }
      if (event.status === "done") {
        return {
          ...state,
          turnProgress: null,
          progressQueue: [],
        };
      }
      if (state.turnProgress === null) {
        return {
          ...state,
          progressRunId: event.run_id ?? state.progressRunId,
          turnProgress: event,
          progressQueue: [],
        };
      }
      if (state.turnProgress.phase === event.phase) {
        return { ...state, turnProgress: event };
      }
      const lastQueued = state.progressQueue.at(-1);
      const progressQueue =
        lastQueued?.phase === event.phase
          ? [...state.progressQueue.slice(0, -1), event]
          : [...state.progressQueue, event];
      return {
        ...state,
        progressQueue,
      };
    }
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
            pendingAnswer: "",
            turnProgress: null,
            progressQueue: [],
            lastRoute: event.route ?? "",
            timeline: push(state.timeline, {
              kind: "soca",
              text: event.text ?? "",
              citations: event.citations ?? [],
              latencyMs:
                event.usage &&
                typeof event.usage["total_latency_ms"] === "number"
                  ? Number(event.usage["total_latency_ms"])
                  : undefined,
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
      return {
        ...state,
        profiles: event.profiles,
        knowledgeIndex: event.knowledge_index ?? null,
        runtimeComponents: event.runtime_components ?? [],
      };
    case "context":
      return { ...state, context: event };
    case "memory":
      return { ...state, memorySnapshot: event };
    case "memory_compaction":
      return {
        ...state,
        memoryCompaction: event,
      };
    case "usage":
      return { ...state, usageSnapshot: event };
    case "router_trace":
      return {
        ...state,
        routerTier: event.tier,
        routerLatencyMs: event.latency_ms,
      };
    case "memory_trace":
      return {
        ...state,
        memoryMode: event.mode,
        memoryHits: event.hit_count ?? event.hits.length,
        memoryTelemetry: event,
      };
    case "memory_proposals":
      return {
        ...state,
        proposals: event.proposals,
        proposalsOpen: true,
        memoryActionError: "",
      };
    case "memory_action":
      return event.ok
        ? {
            ...state,
            proposals: state.proposals.filter(
              (proposal) => proposal.id !== event.proposal_id,
            ),
            memoryActionError: "",
          }
        : {
            ...state,
            memoryActionError: event.error_code ?? "memory action failed",
          };
    case "retrieval_trace":
      return {
        ...state,
        retrievalTrace: {
          query: event.query,
          tier: event.tier,
          latencyMs: event.latency_ms,
          columns: event.columns,
          fused: event.fused,
          rejectedCount: event.rejected_count ?? 0,
          evidence: event.evidence ?? null,
        },
      };
    case "turn_started":
    case "goal_resolved":
    case "step_started":
    case "step_progress":
    case "step_completed":
    case "public_update":
    case "answer_delta":
    case "verification_started":
    case "verification_completed":
    case "turn_terminal": {
      const last = state.workflowEvents.at(-1);
      if (
        last &&
        last.run_id === event.run_id &&
        event.sequence <= last.sequence
      ) {
        return state;
      }
      const workflowEvents =
        last && last.run_id !== event.run_id
          ? [event]
          : [...state.workflowEvents, event];
      const payload = event.payload;
      const delta = typeof payload["text"] === "string" ? payload["text"] : "";
      return {
        ...state,
        workflowEvents,
        pendingAnswer:
          event.event === "answer_delta"
            ? state.pendingAnswer + delta
            : event.event === "turn_terminal"
              ? ""
              : state.pendingAnswer,
        workflowTerminalStatus:
          event.event === "turn_terminal" && typeof payload["terminal_status"] === "string"
            ? payload["terminal_status"]
            : state.workflowTerminalStatus,
      };
    }
    case "llm_providers":
      return { ...state, llmProviders: event.providers, settingsNotice: "" };
    case "llm_catalog":
      return {
        ...state,
        llmCatalog: event.models,
        llmCatalogProvider: event.provider,
        settingsNotice: "",
      };
    case "llm_key_status": {
      if (event.pending) {
        return {
          ...state,
          llmKeyPendingProvider: event.provider,
          settingsNotice: event.message ?? "Đang xác thực API key…",
        };
      }
      const llmProviders = state.llmProviders.map((provider) =>
        provider.key === event.provider && event.ok
          ? { ...provider, has_key: true }
          : provider,
      );
      return {
        ...state,
        llmProviders,
        llmKeyPendingProvider:
          state.llmKeyPendingProvider === event.provider
            ? null
            : state.llmKeyPendingProvider,
        settingsNotice: event.ok
          ? `API key đã được xác thực${event.masked ? ` (${event.masked})` : ""}.`
          : (event.message ?? "Không thể xác thực API key."),
      };
    }
    case "llm_config":
      return {
        ...state,
        llmConfig: event,
        settingsNotice: "",
        stack: {
          ...state.stack,
          llm:
            event.backend === "remote"
              ? `${event.provider}:${event.model}`
              : event.model,
        },
      };
    case "engine_error":
      return {
        ...state,
        chatBusy: false,
        settingsNotice: event.message,
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
        activeInfo: null,
        chatBusy: true,
        turnProgress: null,
        progressQueue: [],
        progressRunId: null,
        pendingAnswer: "",
        workflowEvents: [],
        workflowTerminalStatus: null,
        retrievalTrace: null,
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
        speechChunks: [],
      };
    case "show_info":
      return { ...state, activeInfo: action.view };
    case "clear_info":
      return { ...state, activeInfo: null };
    case "clear_timeline":
      return { ...state, timeline: [] };
    case "clear_proposals":
      return {
        ...state,
        proposals: [],
        proposalsOpen: false,
        memoryActionError: "",
      };
    case "advance_progress": {
      const [next, ...rest] = state.progressQueue;
      return next
        ? { ...state, turnProgress: next, progressQueue: rest }
        : state;
    }
    default:
      return state;
  }
}
