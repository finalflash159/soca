// Wire types for the `soca engine` NDJSON protocol (see soca/app/engine.py).

export interface EngineCommand {
  cmd: "status" | "chat" | "voice_start" | "voice_stop" | "memory" | "usage" | "quit";
  text?: string;
  max_turns?: number | null;
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

export interface EngineErrorEvent {
  event: "engine_error";
  message: string;
}

export interface ByeEvent {
  event: "bye";
}

export type EngineEvent =
  | HelloEvent
  | VoiceEvent
  | ChatEvent
  | StatusEvent
  | MemoryEvent
  | UsageEvent
  | EngineErrorEvent
  | ByeEvent;

export function parseEngineEvent(line: string): EngineEvent | null {
  const trimmed = line.trim();
  if (!trimmed) return null;
  try {
    const parsed = JSON.parse(trimmed) as { event?: string };
    if (typeof parsed !== "object" || parsed === null || !parsed.event)
      return null;
    return parsed as EngineEvent;
  } catch {
    return null;
  }
}
