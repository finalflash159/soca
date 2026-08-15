/**
 * Types for the `soca engine` NDJSON protocol.
 *
 * Source of truth: `docs/18-engine-protocol.md`, pinned by
 * `tests/test_engine_protocol_contract.py`. Keep this file in step with that
 * document — the Python test guards the engine side, nothing guards this side
 * except review.
 *
 * Everything is deliberately permissive about extra keys: §7 obliges a client
 * to tolerate unknown events and unknown fields, so these are read-shapes, not
 * validators.
 */

export const PROTOCOL_VERSION = 2;

/** Commands the engine accepts (docs/18 §2). */
export type EngineCommand =
  | { cmd: "status" }
  | { cmd: "context" }
  | { cmd: "memory" }
  | { cmd: "memory_compact"; action?: "request" | "status" | "cancel" }
  | { cmd: "memory_proposals" }
  | { cmd: "memory_approve"; proposal_id: string }
  | { cmd: "memory_reject"; proposal_id: string }
  | { cmd: "usage" }
  | { cmd: "llm_providers" }
  | { cmd: "llm_models"; provider: string; query?: string }
  | { cmd: "llm_set_key"; provider: string; key: string }
  | { cmd: "llm_select"; [key: string]: unknown }
  | { cmd: "llm_config" }
  | { cmd: "chat"; text: string }
  | { cmd: "voice_start"; max_turns?: number }
  | { cmd: "voice_stop" }
  | { cmd: "voice_profile_select"; [key: string]: unknown }
  | { cmd: "knowledge_init" }
  | { cmd: "knowledge_index" }
  | { cmd: "quit" };

/** §6 — workflow envelope vocabulary. */
export type WorkflowEventName =
  | "turn_started"
  | "step_started"
  | "step_progress"
  | "step_completed"
  | "verification_started"
  | "verification_completed"
  | "answer_delta"
  | "public_update"
  | "goal_resolved"
  | "turn_terminal";

export type WorkflowStatus = "started" | "active" | "completed" | "failed" | "cancelled";

export type TerminalStatus =
  | "achieved"
  | "needs_clarification"
  | "insufficient_evidence"
  | "safe_failure"
  | "budget_exhausted"
  | "cancelled"
  | "system_failure";

export type TurnNode =
  | "admit"
  | "resolve_goal"
  | "ask_clarification"
  | "choose_capability"
  | "make_plan"
  | "authorize_action"
  | "execute_action"
  | "assess_observation"
  | "revise_query"
  | "synthesize"
  | "verify_answer"
  | "repair_answer"
  | "finalize";

export interface WorkflowFrame {
  event: WorkflowEventName;
  protocol_version: number;
  session_id: string;
  run_id: string;
  goal_id: string;
  sequence: number;
  surface: string;
  timestamp: string;
  node: TurnNode;
  status: WorkflowStatus;
  payload: Record<string, unknown>;
}

/** §5 — the 20 voice event types. */
export type VoiceEventType =
  | "loading"
  | "warmup"
  | "ready"
  | "loop_started"
  | "loop_stopped"
  | "recording"
  | "voice_level"
  | "audio"
  | "recorded"
  | "transcribing"
  | "asr_partial"
  | "repair"
  | "turn_start"
  | "progress"
  | "turn_end"
  | "done"
  | "tts"
  | "playback_started"
  | "barge_in"
  | "error";

export interface HelloFrame {
  event: "hello";
  version: number;
  protocol_version: number;
  supported_versions: number[];
  profile: string;
  no_model: boolean;
  stack: Record<string, string>;
}

export interface ChatFrame {
  event: "chat";
  type: "loading" | "ready" | "start" | "done" | "error";
  text?: string;
  run_id?: string;
  goal_id?: string;
  route?: string;
  /** §4: a terminal outcome, never an error state. */
  blocked?: boolean;
  citations?: unknown[];
  usage?: Record<string, unknown>;
  llm_status?: string;
  knowledge_status?: string;
  memory_status?: string;
}

export interface VoiceFrame {
  event: "voice";
  type: VoiceEventType;
  text?: string;
  latency_ms?: number;
  metadata?: Record<string, unknown>;
  usage?: Record<string, unknown>;
}

export interface StatusFrame {
  event: "status";
  profiles?: unknown[];
  knowledge_vault?: unknown;
  knowledge_index?: unknown | null;
  runtime_components?: Array<{ name?: string; status?: string; [key: string]: unknown }>;
}

export interface EngineErrorFrame {
  event: "engine_error";
  message: string;
  code?: string;
  detail?: string;
}

export interface KnowledgeSetupFrame {
  event: "knowledge_setup";
  action: string;
  status: string;
  vault: string;
  detail: string;
  error_code?: string;
}

export interface LlmConfigFrame {
  event: "llm_config";
  backend: string;
  provider: string;
  model: string;
  /** §7: display the effective values, not the requested ones. */
  effective_max_tokens: number;
  effective_reasoning_enabled: boolean;
  reasoning_mandatory: boolean;
  runtime_ready: boolean;
  settings_error: string | null;
  [key: string]: unknown;
}

export interface GenericFrame {
  event: string;
  [key: string]: unknown;
}

export type EngineFrame =
  | HelloFrame
  | ChatFrame
  | VoiceFrame
  | StatusFrame
  | EngineErrorFrame
  | KnowledgeSetupFrame
  | LlmConfigFrame
  | WorkflowFrame
  | GenericFrame;

/**
 * Discriminate the two envelope shapes (§3).
 *
 * Keyed on `protocol_version`, not on the event name, exactly as the document
 * requires — new workflow event names must not fall through to the flat branch.
 */
export function isWorkflowFrame(frame: EngineFrame): frame is WorkflowFrame {
  return typeof (frame as WorkflowFrame).protocol_version === "number" && "node" in frame;
}

export function isChatFrame(frame: EngineFrame): frame is ChatFrame {
  return frame.event === "chat";
}

export function isVoiceFrame(frame: EngineFrame): frame is VoiceFrame {
  return frame.event === "voice";
}

/** §7 obligation 1: refuse an engine speaking a version we do not implement. */
export function helloIsCompatible(hello: HelloFrame): boolean {
  return (
    hello.protocol_version === PROTOCOL_VERSION ||
    (Array.isArray(hello.supported_versions) &&
      hello.supported_versions.includes(PROTOCOL_VERSION))
  );
}
