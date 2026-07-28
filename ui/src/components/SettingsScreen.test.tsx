import React from "react";
import { render } from "ink-testing-library";
import { describe, expect, it, vi } from "vitest";
import { SettingsScreen, filterCatalog } from "./SettingsScreen.js";

const providers = [
  { key: "openai", label: "OpenAI", has_key: false, has_pricing_api: false },
  {
    key: "openrouter",
    label: "OpenRouter",
    has_key: true,
    has_pricing_api: true,
  },
  { key: "groq", label: "Groq", has_key: false, has_pricing_api: false },
  { key: "gemini", label: "Gemini", has_key: false, has_pricing_api: false },
];

describe("SettingsScreen", () => {
  it("renders provider state, privacy warning, and transparent model prices", () => {
    const view = render(
      <SettingsScreen
        config={{
          event: "llm_config",
          backend: "remote",
          provider: "openrouter",
          model: "openai/gpt-4o-mini",
          max_tokens: 160,
          temperature: 0.2,
          top_p: 0.95,
          pricing_as_of: "2026-07",
          pricing: null,
          context_length: 128000,
        }}
        providers={providers}
        catalog={[
          {
            id: "openai/gpt-4o-mini",
            label: "GPT-4o mini",
            context_length: 128000,
            price_prompt_per_1m: 0.15,
            price_completion_per_1m: 0.6,
            pricing_source: "live",
          },
        ]}
        catalogProvider="openrouter"
        keyPendingProvider={null}
        notice=""
        onRequestModels={vi.fn()}
        onSetKey={vi.fn()}
        onSelect={vi.fn()}
        onExit={vi.fn()}
      />,
    );

    const frame = view.lastFrame();
    expect(frame).toContain("Cài đặt LLM");
    expect(frame).toContain("OpenRouter");
    expect(frame).toContain("Remote gửi transcript");
    expect(frame).toContain("128k");
    expect(frame).toContain("$0.15 / $0.60 / 1M");
    expect(frame).toContain("live");
    view.unmount();
  });

  const twoModelCatalog = [
    {
      id: "openai/gpt-4o-mini",
      label: "GPT-4o mini",
      context_length: 128000,
      price_prompt_per_1m: 0.15,
      price_completion_per_1m: 0.6,
      pricing_source: "live" as const,
    },
    {
      id: "meta-llama/llama-3.3-70b-instruct",
      label: "Llama 3.3 70B",
      context_length: 131072,
      price_prompt_per_1m: 0.59,
      price_completion_per_1m: 0.79,
      pricing_source: "live" as const,
    },
  ];

  it("renders the full catalog with a count when the query is empty", () => {
    const view = render(
      <SettingsScreen
        config={{
          event: "llm_config",
          backend: "remote",
          provider: "openrouter",
          model: "openai/gpt-4o-mini",
          max_tokens: 160,
          temperature: 0.2,
          top_p: 0.95,
          pricing_as_of: "2026-07",
          pricing: null,
          context_length: 128000,
        }}
        providers={providers}
        catalog={twoModelCatalog}
        catalogProvider="openrouter"
        keyPendingProvider={null}
        notice=""
        onRequestModels={vi.fn()}
        onSetKey={vi.fn()}
        onSelect={vi.fn()}
        onExit={vi.fn()}
      />,
    );

    const frame = view.lastFrame() ?? "";
    expect(frame).toContain("2/2 model");
    expect(frame).toContain("gpt-4o-mini");
    expect(frame).toContain("llama-3.3-70b");
    view.unmount();
  });

  describe("filterCatalog (realtime filter)", () => {
    it("returns the whole catalog for an empty or whitespace query", () => {
      expect(filterCatalog(twoModelCatalog, "")).toHaveLength(2);
      expect(filterCatalog(twoModelCatalog, "   ")).toHaveLength(2);
    });

    it("matches a case-insensitive substring across id and label", () => {
      const result = filterCatalog(twoModelCatalog, "LLAMA");
      expect(result.map((m) => m.id)).toEqual([
        "meta-llama/llama-3.3-70b-instruct",
      ]);
    });

    it("matches on the human label, not just the id", () => {
      const result = filterCatalog(twoModelCatalog, "mini");
      expect(result.map((m) => m.id)).toEqual(["openai/gpt-4o-mini"]);
    });

    it("ANDs whitespace-separated tokens, order-independent", () => {
      const result = filterCatalog(twoModelCatalog, "70b llama");
      expect(result.map((m) => m.id)).toEqual([
        "meta-llama/llama-3.3-70b-instruct",
      ]);
    });

    it("returns nothing when no model matches", () => {
      expect(filterCatalog(twoModelCatalog, "nonexistent")).toEqual([]);
    });
  });

  it("starts replacement with an empty key buffer", async () => {
    const onSetKey = vi.fn();
    const view = render(
      <SettingsScreen
        config={{
          event: "llm_config",
          backend: "remote",
          provider: "openrouter",
          model: "openai/gpt-4o-mini",
          max_tokens: 4096,
          temperature: 0.2,
          top_p: 0.95,
          pricing_as_of: "2026-07",
          pricing: null,
          context_length: 128000,
        }}
        providers={providers}
        catalog={[]}
        catalogProvider=""
        keyPendingProvider={null}
        notice=""
        onRequestModels={vi.fn()}
        onSetKey={onSetKey}
        onSelect={vi.fn()}
        onExit={vi.fn()}
      />,
    );

    await new Promise((resolve) => setImmediate(resolve));
    view.stdin.write("r");
    await new Promise((resolve) => setImmediate(resolve));
    view.stdin.write("old-secret");
    await new Promise((resolve) => setImmediate(resolve));
    view.stdin.write("\u001b");
    await new Promise((resolve) => setImmediate(resolve));
    view.stdin.write("r");
    await new Promise((resolve) => setImmediate(resolve));
    view.stdin.write("new-secret");
    await new Promise((resolve) => setImmediate(resolve));
    view.stdin.write("\r");
    await new Promise((resolve) => setImmediate(resolve));

    expect(onSetKey).toHaveBeenCalledWith("openrouter", "new-secret");
    view.unmount();
  });

  it("clears the whole temporary key buffer on Delete", async () => {
    const onSetKey = vi.fn();
    const view = render(
      <SettingsScreen
        config={{
          event: "llm_config",
          backend: "remote",
          provider: "openrouter",
          model: "openai/gpt-4o-mini",
          max_tokens: 4096,
          temperature: 0.2,
          top_p: 0.95,
          pricing_as_of: "2026-07",
          pricing: null,
          context_length: 128000,
        }}
        providers={providers}
        catalog={[]}
        catalogProvider=""
        keyPendingProvider={null}
        notice=""
        onRequestModels={vi.fn()}
        onSetKey={onSetKey}
        onSelect={vi.fn()}
        onExit={vi.fn()}
      />,
    );

    await new Promise((resolve) => setImmediate(resolve));
    view.stdin.write("r");
    await new Promise((resolve) => setImmediate(resolve));
    view.stdin.write("old-secret");
    await new Promise((resolve) => setImmediate(resolve));
    view.stdin.write("\u001b[3~");
    await new Promise((resolve) => setImmediate(resolve));
    view.stdin.write("\r");
    await new Promise((resolve) => setImmediate(resolve));

    expect(onSetKey).not.toHaveBeenCalled();
    view.unmount();
  });
});
