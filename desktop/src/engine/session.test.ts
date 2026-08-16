import { describe, expect, it } from "vitest";

import type { EngineFrame } from "./protocol";
import { budgetUsedFraction, contextBasis, initialSession, reduceSession } from "./session";

function fold(frames: EngineFrame[]) {
  return frames.reduce(reduceSession, initialSession);
}

const readyContext = {
  event: "context",
  ready: true,
  estimated: true,
  resident_prompt_tokens: 4000,
  input_budget_tokens: 16000,
  model_context_tokens: 200000,
  output_reserve_tokens: 8000,
  available_dynamic_tokens: 12000,
  observed_prompt_tokens: null,
  provider_prompt_tokens: null,
  components: [{ name: "system", tokens: 300 }],
} as EngineFrame;

describe("context", () => {
  it("reads a manifest", () => {
    const state = fold([readyContext]);
    expect(state.context?.ready).toBe(true);
    expect(state.context?.residentPromptTokens).toBe(4000);
    expect(state.context?.components).toHaveLength(1);
  });

  it("keeps provider counts null until a provider reports them", () => {
    // docs/18 §4 forbids presenting an estimate as observed usage.
    const state = fold([readyContext]);
    expect(state.context?.providerPromptTokens).toBeNull();
    expect(state.context?.observedPromptTokens).toBeNull();
  });

  it("carries a PromptBudgetError instead of pretending to have a manifest", () => {
    const state = fold([
      {
        event: "context",
        ready: false,
        estimated: true,
        context_error: "prompt_over_budget",
        context_error_detail: "resident 20000 > 16000",
        components: [],
      } as EngineFrame,
    ]);
    expect(state.context?.ready).toBe(false);
    expect(contextBasis(state.context)).toContain("prompt_over_budget");
  });

  it("names the basis so an estimate is never read as measured", () => {
    expect(contextBasis(fold([readyContext]).context)).toContain("ước lượng");
    expect(
      contextBasis(fold([{ ...readyContext, estimated: false } as EngineFrame]).context),
    ).toContain("lượt vừa chạy");
    expect(contextBasis(null)).toContain("chưa có");
  });
});

describe("budget fraction", () => {
  it("is the resident share of the input budget", () => {
    expect(budgetUsedFraction(fold([readyContext]).context)).toBeCloseTo(0.25);
  });

  it("is null when the manifest failed", () => {
    const state = fold([{ event: "context", ready: false, components: [] } as EngineFrame]);
    expect(budgetUsedFraction(state.context)).toBeNull();
  });

  it("never exceeds one", () => {
    const state = fold([{ ...readyContext, resident_prompt_tokens: 99999 } as EngineFrame]);
    expect(budgetUsedFraction(state.context)).toBe(1);
  });
});

describe("usage", () => {
  it("reads session totals", () => {
    const state = fold([
      {
        event: "usage",
        turns: 3,
        llm_turns: 2,
        prompt_tokens: 1840,
        completion_tokens: 260,
        mean_ttft_ms: 1287,
        mean_tokens_per_second: 42.5,
      } as EngineFrame,
    ]);
    expect(state.usage?.turns).toBe(3);
    expect(state.usage?.meanTtftMs).toBe(1287);
  });

  it("defaults counters to zero but leaves rates null when absent", () => {
    const state = fold([{ event: "usage" } as EngineFrame]);
    expect(state.usage?.turns).toBe(0);
    expect(state.usage?.meanTtftMs).toBeNull();
  });
});

describe("unknown frames", () => {
  it("are ignored", () => {
    expect(fold([{ event: "status" } as EngineFrame])).toEqual(initialSession);
  });
});
