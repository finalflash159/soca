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

  it("rejects malformed protocol lines", () => {
    expect(parseEngineEvent("not-json")).toBeNull();
    expect(parseEngineEvent('{"event": 1}')).toBeNull();
  });
});
