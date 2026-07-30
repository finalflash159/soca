// Wire types for the `soca engine` NDJSON protocol (see soca/app/engine.py).

export const PROTOCOL_VERSION = 2;
export const SUPPORTED_PROTOCOL_VERSIONS = [1, 2] as const;

export interface EngineCommand {
  cmd:
    | "status"
    | "chat"
    | "voice_start"
    | "voice_stop"
    | "memory"
    | "memory_compact"
    | "context"
    | "usage"
    | "llm_providers"
    | "llm_models"
    | "llm_set_key"
    | "llm_select"
    | "llm_config"
    | "memory_proposals"
    | "memory_approve"
    | "memory_reject"
    | "quit";
  text?: string;
  max_turns?: number | null;
  provider?: string;
  query?: string;
  key?: string;
  backend?: "local" | "remote";
  model?: string;
  max_tokens?: number;
  reasoning_enabled?: boolean;
  proposal_id?: string;
  action?: "request" | "status" | "cancel";
}

export interface HelloEvent {
  event: "hello";
  version: number;
  protocol_version?: number;
  supported_versions?: number[];
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
  evidence_status: string;
  answer_policy: string;
  answer_policy_reason: string;
  grounding_policy_version: string;
  citation_count: number;
  memory_access_plan: {
    include_core: boolean;
    include_working: boolean;
    archive_mode: "none" | "semantic" | "episodic" | "both";
    archive_query: string | null;
    reason: string;
  } | null;
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
  hit_count?: number;
  compacted_turn_count: number | null;
  recent_turn_count: number | null;
  background_status: "idle" | "queued" | "running" | "failed";
  summary_worker_state: string;
  summary_generation: number | null;
  pending_compaction: boolean;
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
  rejected_count?: number;
  evidence?: Record<string, unknown> | null;
}

export type TurnProgressPhase =
  | "preparing"
  | "analyzing"
  | "routing"
  | "memory"
  | "retrieval"
  | "tool"
  | "synthesis"
  | "validation"
  | "speech"
  | "complete";

export interface TurnProgressEvent {
  event: "turn_progress";
  surface: "chat" | "voice";
  phase: TurnProgressPhase;
  operation: string;
  status: "active" | "done" | "failed" | "cancelled";
  run_id?: string;
  goal_id?: string;
  sequence?: number;
  terminal_status?: string;
  detail?: string;
}

export type WorkflowEventName =
  | "turn_started"
  | "goal_resolved"
  | "step_started"
  | "step_progress"
  | "step_completed"
  | "public_update"
  | "answer_delta"
  | "verification_started"
  | "verification_completed"
  | "turn_terminal";

const WORKFLOW_EVENT_NAMES = new Set<WorkflowEventName>([
  "turn_started",
  "goal_resolved",
  "step_started",
  "step_progress",
  "step_completed",
  "public_update",
  "answer_delta",
  "verification_started",
  "verification_completed",
  "turn_terminal",
]);

export type WorkflowNode =
  | "admit"
  | "resolve_goal"
  | "choose_capability"
  | "make_plan"
  | "authorize_action"
  | "execute_action"
  | "assess_observation"
  | "revise_query"
  | "synthesize"
  | "verify_answer"
  | "repair_answer"
  | "ask_clarification"
  | "finalize";

const WORKFLOW_NODES = new Set<WorkflowNode>([
  "admit",
  "resolve_goal",
  "choose_capability",
  "make_plan",
  "authorize_action",
  "execute_action",
  "assess_observation",
  "revise_query",
  "synthesize",
  "verify_answer",
  "repair_answer",
  "ask_clarification",
  "finalize",
]);

type WorkflowEventStatus =
  | "started"
  | "active"
  | "completed"
  | "failed"
  | "cancelled";

const WORKFLOW_STATUSES = new Set<WorkflowEventStatus>([
  "started",
  "active",
  "completed",
  "failed",
  "cancelled",
]);

export interface WorkflowEvent {
  event: WorkflowEventName;
  protocol_version: 2;
  session_id: string;
  run_id: string;
  goal_id: string;
  sequence: number;
  surface: "ask" | "cli" | "chat" | "voice";
  timestamp: string;
  node: WorkflowNode;
  status: WorkflowEventStatus;
  payload: Record<string, unknown>;
}

function isWorkflowEvent(value: Record<string, unknown>): boolean {
  return (
    typeof value.event === "string" &&
    WORKFLOW_EVENT_NAMES.has(value.event as WorkflowEventName) &&
    value.protocol_version === PROTOCOL_VERSION &&
    typeof value.session_id === "string" &&
    value.session_id.length > 0 &&
    typeof value.run_id === "string" &&
    value.run_id.length > 0 &&
    typeof value.goal_id === "string" &&
    value.goal_id.length > 0 &&
    typeof value.sequence === "number" &&
    Number.isInteger(value.sequence) &&
    value.sequence >= 0 &&
    ["ask", "cli", "chat", "voice"].includes(String(value.surface)) &&
    typeof value.timestamp === "string" &&
    !Number.isNaN(Date.parse(value.timestamp)) &&
    typeof value.node === "string" &&
    WORKFLOW_NODES.has(value.node as WorkflowNode) &&
    typeof value.status === "string" &&
    WORKFLOW_STATUSES.has(value.status as WorkflowEventStatus) &&
    typeof value.payload === "object" &&
    value.payload !== null &&
    !Array.isArray(value.payload)
  );
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
  run_id?: string;
  goal_id?: string;
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
  runtime_components?: RuntimeComponentStatus[];
}

export interface RuntimeComponentStatus {
  id: string;
  label: string;
  status: "loaded" | "ready" | "configured" | "disabled" | "missing" | "degraded";
  detail: string;
}

export interface MemoryEvent {
  event: "memory";
  enabled: boolean;
  text: string;
  summary: string;
  recent: string;
  stats: SessionMemoryStats | null;
}

export interface SessionMemoryStats {
  current_tokens: number;
  rendered_tokens: number;
  hard_limit_tokens: number;
  high_watermark_tokens: number;
  target_tokens: number;
  summary_tokens: number;
  recent_tokens: number;
  turn_count: number;
  complete_turn_count: number;
  summary_generation: number | null;
  pending_compaction: boolean;
  worker_state: string;
}

export interface ContextComponent {
  id:
    | "system"
    | "core_memory"
    | "working_summary"
    | "recent_conversation"
    | "prompt_scaffolding"
    | "archive_memory"
    | "knowledge"
    | "current_input"
    | "answer_prefix"
    | "memory"
    | "joint_grounding_policy"
    | "answer_policy";
  label: string;
  tokens: number | null;
  policy: "always" | "always_when_present" | "on_demand" | "per_turn";
  included?: boolean;
  required?: boolean;
  priority?: number;
}

export interface ContextEvent {
  event: "context";
  estimated: boolean;
  token_counter: string;
  session: SessionMemoryStats | null;
  resident_prompt_tokens: number;
  output_reserve_tokens: number;
  model_context_tokens: number | null;
  available_dynamic_tokens: number | null;
  input_budget_tokens?: number | null;
  prompt_hash?: string | null;
  prompt_manifest?: Record<string, unknown>;
  observed_prompt_tokens?: number | null;
  observed_prompt_token_source?: string | null;
  provider_prompt_tokens?: number | null;
  prompt_token_delta?: number | null;
  components: ContextComponent[];
}

export interface MemoryCompactionEvent {
  event: "memory_compaction";
  status: string;
  generation?: number | null;
  detail?: string;
  before_tokens?: number | null;
  after_tokens?: number | null;
  compacted_turns?: number;
  complete_turns?: number;
  minimum_complete_turns?: number | null;
  elapsed_ms?: number | null;
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
  max_output_tokens?: number | null;
  reasoning_supported?: boolean | null;
  reasoning_mandatory?: boolean;
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
  pending?: boolean;
  masked?: string;
  message?: string;
}

export interface LlmConfigEvent {
  event: "llm_config";
  backend: "local" | "remote";
  provider: string;
  model: string;
  max_tokens: number;
  effective_max_tokens?: number;
  reasoning_enabled?: boolean;
  effective_reasoning_enabled?: boolean | null;
  reasoning_supported?: boolean | null;
  reasoning_mandatory?: boolean;
  temperature: number;
  top_p: number;
  pricing_as_of: string;
  pricing: RemoteModelEvent | null;
  context_length: number | null;
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
  | TurnProgressEvent
  | WorkflowEvent
  | ChatEvent
  | StatusEvent
  | MemoryEvent
  | ContextEvent
  | MemoryCompactionEvent
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
      "turn_progress",
      "turn_started",
      "goal_resolved",
      "step_started",
      "step_progress",
      "step_completed",
      "public_update",
      "answer_delta",
      "verification_started",
      "verification_completed",
      "turn_terminal",
      "chat",
      "status",
      "memory",
      "context",
      "memory_compaction",
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
    if (
      WORKFLOW_EVENT_NAMES.has(eventName as WorkflowEventName) &&
      !isWorkflowEvent(parsed as Record<string, unknown>)
    )
      return null;
    return adaptLegacyEvent(parsed as Record<string, unknown>) as unknown as EngineEvent;
  } catch {
    return null;
  }
}

export function adaptLegacyEvent(event: Record<string, unknown>): Record<string, unknown> {
  if (event.event !== "hello" || typeof event.version !== "number") return event;
  if (event.protocol_version !== undefined) return event;
  return {
    ...event,
    protocol_version: event.version,
    supported_versions: [event.version],
  };
}
