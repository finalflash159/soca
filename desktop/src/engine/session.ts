/**
 * Prompt budget and token usage for the current session.
 *
 * These two events had no consumer at all, which made `/context` and `/usage`
 * dead commands: the palette sent them, the engine answered, and nothing on
 * screen changed. Offering a command that does nothing is worse than not
 * offering it.
 *
 * Shapes come from `docs/18-engine-protocol.md` §4.
 */

import type { EngineFrame } from "./protocol";

export interface ContextComponent {
  name?: string;
  tokens?: number;
  [key: string]: unknown;
}

export interface ContextState {
  /** False when the engine could not build a manifest (`PromptBudgetError`). */
  ready: boolean;
  /** True for a projection from resident state, false for a turn that ran. */
  estimated: boolean;
  error: string | null;
  errorDetail: string | null;
  residentPromptTokens: number | null;
  outputReserveTokens: number | null;
  modelContextTokens: number | null;
  inputBudgetTokens: number | null;
  availableDynamicTokens: number | null;
  /** Null until a provider reports real counts (§4). */
  observedPromptTokens: number | null;
  providerPromptTokens: number | null;
  components: ContextComponent[];
}

export interface UsageState {
  turns: number;
  llmTurns: number;
  promptTokens: number;
  completionTokens: number;
  meanTtftMs: number | null;
  meanTokensPerSecond: number | null;
}

export interface SessionState {
  context: ContextState | null;
  usage: UsageState | null;
}

export const initialSession: SessionState = { context: null, usage: null };

function num(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function reduceSession(state: SessionState, frame: EngineFrame): SessionState {
  if (frame.event === "context") {
    return {
      ...state,
      context: {
        ready: frame.ready === true,
        estimated: frame.estimated === true,
        error: typeof frame.context_error === "string" ? frame.context_error : null,
        errorDetail:
          typeof frame.context_error_detail === "string" ? frame.context_error_detail : null,
        residentPromptTokens: num(frame.resident_prompt_tokens),
        outputReserveTokens: num(frame.output_reserve_tokens),
        modelContextTokens: num(frame.model_context_tokens),
        inputBudgetTokens: num(frame.input_budget_tokens),
        availableDynamicTokens: num(frame.available_dynamic_tokens),
        observedPromptTokens: num(frame.observed_prompt_tokens),
        providerPromptTokens: num(frame.provider_prompt_tokens),
        components: Array.isArray(frame.components) ? (frame.components as ContextComponent[]) : [],
      },
    };
  }

  if (frame.event === "usage") {
    return {
      ...state,
      usage: {
        turns: num(frame.turns) ?? 0,
        llmTurns: num(frame.llm_turns) ?? 0,
        promptTokens: num(frame.prompt_tokens) ?? 0,
        completionTokens: num(frame.completion_tokens) ?? 0,
        meanTtftMs: num(frame.mean_ttft_ms),
        meanTokensPerSecond: num(frame.mean_tokens_per_second),
      },
    };
  }

  return state;
}

/** Share of the input budget already spent by resident context. */
export function budgetUsedFraction(context: ContextState | null): number | null {
  if (context === null || !context.ready) {
    return null;
  }
  const { residentPromptTokens: used, inputBudgetTokens: budget } = context;
  if (used === null || budget === null || budget <= 0) {
    return null;
  }
  return Math.min(1, used / budget);
}

/**
 * How the figures should be read.
 *
 * §4 is explicit that an estimate must not be presented as observed usage, so
 * the label carries the distinction rather than leaving it to a tooltip.
 */
export function contextBasis(context: ContextState | null): string {
  if (context === null) {
    return "chưa có dữ liệu";
  }
  if (!context.ready) {
    return `không dựng được manifest${context.error !== null ? ` · ${context.error}` : ""}`;
  }
  return context.estimated ? "ước lượng từ trạng thái hiện tại" : "đo trên lượt vừa chạy";
}
