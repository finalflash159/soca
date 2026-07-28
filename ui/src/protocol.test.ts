import { describe, expect, it } from "vitest";
import { parseEngineEvent } from "./protocol.js";

describe("parseEngineEvent", () => {
  it("parses a remote LLM catalog event", () => {
    const event = parseEngineEvent(
      JSON.stringify({
        event: "llm_catalog",
        provider: "openrouter",
        pricing_as_of: "2026-07",
        models: [
          {
            id: "openai/gpt-4o-mini",
            label: "GPT-4o mini",
            context_length: 128000,
            price_prompt_per_1m: 0.15,
            price_completion_per_1m: 0.6,
            pricing_source: "live",
          },
        ],
      }),
    );

    expect(event?.event).toBe("llm_catalog");
  });

  it("parses the structured context token breakdown", () => {
    const event = parseEngineEvent(
      JSON.stringify({
        event: "context",
        estimated: true,
        token_counter: "utf8_bytes_div_4",
        session: {
          current_tokens: 1200,
          rendered_tokens: 1200,
          hard_limit_tokens: 16384,
          high_watermark_tokens: 15000,
          target_tokens: 12000,
          summary_tokens: 200,
          recent_tokens: 1000,
          turn_count: 4,
          complete_turn_count: 4,
          summary_generation: 1,
          pending_compaction: false,
          worker_state: "idle",
        },
        resident_prompt_tokens: 1400,
        output_reserve_tokens: 4096,
        model_context_tokens: 32768,
        available_dynamic_tokens: 27272,
        components: [],
      }),
    );

    expect(event?.event).toBe("context");
    if (event?.event === "context") {
      expect(event.session?.hard_limit_tokens).toBe(16384);
    }
  });

  it("rejects malformed protocol lines", () => {
    expect(parseEngineEvent("not-json")).toBeNull();
    expect(parseEngineEvent('{"event": 1}')).toBeNull();
  });

  it("accepts a large remote catalog event", () => {
    const models = Array.from({ length: 500 }, (_, index) => ({
      id: `provider/model-${index}-${"x".repeat(100)}`,
      label: `Model ${index}`,
      context_length: 128000,
      price_prompt_per_1m: 0.15,
      price_completion_per_1m: 0.6,
      pricing_source: "live",
    }));
    const line = JSON.stringify({
      event: "llm_catalog",
      provider: "openrouter",
      pricing_as_of: "2026-07",
      models,
    });

    expect(line.length).toBeGreaterThan(64_000);
    expect(parseEngineEvent(line)?.event).toBe("llm_catalog");
  });
});
