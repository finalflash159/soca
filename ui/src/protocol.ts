// Wire types for the `soca engine` NDJSON protocol (see soca/app/engine.py).

export interface EngineCommand {
  cmd:
    | "status"
    | "chat"
    | "voice_start"
    | "voice_stop"
    | "memory"
    | "usage"
    | "llm_providers"
    | "llm_models"
    | "llm_set_key"
    | "llm_select"
    | "llm_config"
    | "memory_proposals"
    | "memory_approve"
    | "memory_reject"
    | "inspect"
    | "quit";
  text?: string;
  max_turns?: number | null;
  provider?: string;
  query?: string;
  key?: string;
  backend?: "local" | "remote";
  model?: string;
  proposal_id?: string;
}

export interface HelloEvent {
  event: "hello";
  version: number;
  profile: string;
  no_model: boolean;
  stack: Record<string, string | null>;
}

export interface VoiceEvent {
  event: "voice";
  type: string;
  text: string;
  latency_ms: number | null;
  metadata: Record<string, unknown>;
  usage: Record<string, unknown> | null;
}

export interface RouterTraceEvent {
  event: "router_trace";
  tier: "deterministic" | "semantic" | "llm" | "none";
  tool: string | null;
  latency_ms: number;
}

export interface MemoryTraceEvent {
  event: "memory_trace";
  mode: "blob" | "retrieved" | "degraded";
  degraded_reason?: string;
  hits: Array<{
    id: string;
    corpus: "profile" | "episode";
    relevance: number;
    recency: number;
    importance: number;
    total: number;
  }>;
  compacted_turn_count: number;
  recent_turn_count: number;
  background_status: "idle" | "queued" | "running" | "failed";
  episodic_enabled: boolean;
  pending_proposal_count: number;
}

export interface MemoryProposalEvent {
  event: "memory_proposals";
  proposals: Array<{
    id: string;
    kind: "preference" | "stable_fact" | "project" | "correction";
    statement: string;
    confidence: number;
    createdAt: string;
  }>;
}

export interface MemoryActionEvent {
  event: "memory_action";
  proposal_id: string;
  action: "approved" | "rejected";
  ok: boolean;
  error_code?: string;
}

export interface RetrievalTraceEvent {
  event: "retrieval_trace";
  query: string;
  tier: "deterministic" | "semantic" | "llm" | "none";
  latency_ms: number;
  columns: Array<{
    source: "bm25" | "dense";
    hits: Array<{ path: string; score: number }>;
  }>;
  fused: Array<{ path: string; picked: boolean }>;
}

export interface ChatEvent {
  event: "chat";
  type: "start" | "loading" | "ready" | "done" | "error";
  text?: string;
  route?: string;
  blocked?: boolean;
  usage?: Record<string, unknown> | null;
  llm_status?: string;
  knowledge_status?: string;
  memory_status?: string;
}

export interface StatusEvent {
  event: "status";
  profiles: Array<{
    key: string;
    status: string;
    asr: string;
    llm: string;
    tts: string;
    voice: string | null;
  }>;
  knowledge_index?: {
    sparse_state: string;
    dense_state: string;
    revision: number;
    documents: number;
    chunks: number;
  } | null;
}

export interface MemoryEvent {
  event: "memory";
  enabled: boolean;
  text: string;
}

export interface UsageEvent {
  event: "usage";
  turns: number;
  llm_turns: number;
  prompt_tokens: number;
  completion_tokens: number;
  mean_ttft_ms: number;
  mean_tokens_per_second: number;
}

export interface LlmProviderEvent {
  event: "llm_providers";
  providers: Array<{
    key: string;
    label: string;
    has_key: boolean;
    has_pricing_api: boolean;
  }>;
}

export interface RemoteModelEvent {
  id: string;
  label: string;
  context_length: number | null;
  price_prompt_per_1m: number | null;
  price_completion_per_1m: number | null;
  pricing_source: "live" | "table" | "unknown";
}

export interface LlmCatalogEvent {
  event: "llm_catalog";
  provider: string;
  models: RemoteModelEvent[];
  pricing_as_of: string;
}

export interface LlmKeyStatusEvent {
  event: "llm_key_status";
  provider: string;
  ok: boolean;
  masked?: string;
  message?: string;
}

export interface LlmConfigEvent {
  event: "llm_config";
  backend: "local" | "remote";
  provider: string;
  model: string;
  max_tokens: number;
  temperature: number;
  top_p: number;
  pricing_as_of: string;
  pricing: RemoteModelEvent | null;
}

export interface EngineErrorEvent {
  event: "engine_error";
  message: string;
}

export interface ByeEvent {
  event: "bye";
}

// Catalogs from providers such as OpenRouter contain hundreds of models. The
// engine sends them as one NDJSON event, so the protocol limit must allow a
// valid catalog while still rejecting unexpectedly huge child-process output.
const MAX_PROTOCOL_LINE_LENGTH = 256_000;

export type EngineEvent =
  | HelloEvent
  | VoiceEvent
  | RouterTraceEvent
  | MemoryTraceEvent
  | MemoryProposalEvent
  | MemoryActionEvent
  | RetrievalTraceEvent
  | ChatEvent
  | StatusEvent
  | MemoryEvent
  | UsageEvent
  | LlmProviderEvent
  | LlmCatalogEvent
  | LlmKeyStatusEvent
  | LlmConfigEvent
  | EngineErrorEvent
  | ByeEvent;

export function parseEngineEvent(line: string): EngineEvent | null {
  const trimmed = line.trim();
  if (!trimmed || trimmed.length > MAX_PROTOCOL_LINE_LENGTH) return null;
  try {
    const parsed: unknown = JSON.parse(trimmed);
    const eventName =
      typeof parsed === "object" && parsed !== null
        ? (parsed as { event?: unknown }).event
        : null;
    const known = new Set([
      "hello",
      "voice",
      "router_trace",
      "memory_trace",
      "memory_proposals",
      "memory_action",
      "retrieval_trace",
      "chat",
      "status",
      "memory",
      "usage",
      "llm_providers",
      "llm_catalog",
      "llm_key_status",
      "llm_config",
      "engine_error",
      "bye",
    ]);
    if (
      typeof parsed !== "object" ||
      parsed === null ||
      typeof eventName !== "string" ||
      !known.has(eventName)
    )
      return null;
    return parsed as EngineEvent;
  } catch {
    return null;
  }
}
