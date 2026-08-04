import React from "react";
import { render } from "ink-testing-library";
import { describe, expect, it, vi } from "vitest";
import {
  MAX_MAX_TOKENS,
  MIN_MAX_TOKENS,
  SettingsScreen,
  filterCatalog,
  validateMaxTokens,
} from "./SettingsScreen.js";

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
  it("validates the configurable output range without accepting text", () => {
    expect(validateMaxTokens("")).toContain("Nhập");
    expect(validateMaxTokens("abc")).toContain("không hợp lệ");
    expect(validateMaxTokens(String(MIN_MAX_TOKENS - 1))).toContain("Tối thiểu");
    expect(validateMaxTokens(String(MAX_MAX_TOKENS + 1))).toContain("Tối đa");
    expect(validateMaxTokens("4096")).toBe("");
  });

  it("offers the saved configuration as the first explicit choice", async () => {
    const onExit = vi.fn();
    const view = render(
      <SettingsScreen
        config={{
          event: "llm_config",
          backend: "remote",
          provider: "openrouter",
          model: "z-ai/glm-5",
          max_tokens: 500000,
          effective_max_tokens: 131072,
          reasoning_enabled: true,
          effective_reasoning_enabled: true,
          reasoning_supported: true,
          reasoning_mandatory: false,
          temperature: 0.2,
          top_p: 0.95,
          pricing_as_of: "2026-07",
          pricing: null,
          context_length: 200000,
        }}
        providers={providers}
        catalog={[]}
        catalogProvider=""
        keyPendingProvider={null}
        notice=""
        onRequestModels={vi.fn()}
        onSetKey={vi.fn()}
        onSelect={vi.fn()}
        onExit={onExit}
      />,
    );

    await new Promise((resolve) => setImmediate(resolve));
    expect(view.lastFrame()).toContain("Cấu hình gần nhất");
    expect(view.lastFrame()).toContain("500.000 → 131.072 output tok");
    expect(view.lastFrame()).toContain("reasoning bật");
    view.stdin.write("\r");
    await new Promise((resolve) => setImmediate(resolve));
    expect(onExit).toHaveBeenCalledOnce();
    view.unmount();
  });

  it("shows the selected ASR in the recent configuration for voice", () => {
    const view = render(
      <SettingsScreen
        config={{
          event: "llm_config",
          backend: "local",
          provider: "openrouter",
          model: "arcee_vylinh_3b_q4_k_m",
          max_tokens: 4096,
          temperature: 0.2,
          top_p: 0.95,
          pricing_as_of: "2026-08",
          pricing: null,
          context_length: 32768,
        }}
        returnMode="voice"
        providers={[]}
        profiles={[
          {
            key: "qwen-release",
            status: "ok",
            asr: "qwen3_asr_0_6b",
            llm: "arcee_vylinh_3b_q4_k_m",
            tts: "valtec_multispeaker",
            voice: "NF",
          },
        ]}
        activeProfile="qwen-release"
        catalog={[]}
        catalogProvider=""
        keyPendingProvider={null}
        notice=""
        onRequestModels={vi.fn()}
        onSetKey={vi.fn()}
        onSelect={vi.fn()}
        onExit={vi.fn()}
      />,
    );

    expect(view.lastFrame()).toContain("ASR: qwen-release · qwen3_asr_0_6b");
    view.unmount();
  });

  it("focuses the saved configuration after voice setup from the main UI", async () => {
    const view = render(
      <SettingsScreen
        config={{
          event: "llm_config",
          backend: "remote",
          provider: "openrouter",
          model: "openai/gpt-4o-mini",
          max_tokens: 4096,
          effective_max_tokens: 4096,
          reasoning_enabled: false,
          effective_reasoning_enabled: false,
          reasoning_supported: true,
          reasoning_mandatory: false,
          temperature: 0.2,
          top_p: 0.95,
          pricing_as_of: "2026-08",
          pricing: null,
          context_length: 128000,
        }}
        returnMode="voice"
        providers={providers}
        profiles={[
          {
            key: "qwen-release",
            status: "ok",
            asr: "qwen3_asr_0_6b",
            llm: "openai/gpt-4o-mini",
            tts: "valtec_multispeaker",
            voice: "NF",
          },
        ]}
        activeProfile="qwen-release"
        knowledgeVault={{
          path: "/tmp/Knowledge",
          initialized: true,
          index_home: "/tmp/Knowledge/.soca/knowledge_index",
        }}
        catalog={[]}
        catalogProvider=""
        keyPendingProvider={null}
        notice=""
        onRequestModels={vi.fn()}
        onSetKey={vi.fn()}
        onSelect={vi.fn()}
        onExit={vi.fn()}
      />,
    );

    await new Promise((resolve) => setImmediate(resolve));
    const frame = view.lastFrame() ?? "";
    expect(frame.indexOf("Cấu hình gần nhất")).toBeLessThan(
      frame.indexOf("đang chọn"),
    );
    expect(frame.indexOf("đang chọn")).toBeLessThan(frame.indexOf("Voice ASR"));

    view.stdin.write("\u001b[B");
    await new Promise((resolve) => setImmediate(resolve));
    const asrFrame = view.lastFrame() ?? "";
    expect(asrFrame.indexOf("Voice ASR")).toBeLessThan(
      asrFrame.indexOf("đang chọn"),
    );
    view.unmount();
  });

  it("keeps ASR selection in setup and confirms the engine-applied profile", async () => {
    const onProfileSelect = vi.fn();
    const onExit = vi.fn();
    const profiles = [
      {
        key: "baseline",
        status: "ok",
        asr: "phowhisper_small",
        llm: "arcee_vylinh_3b_q4_k_m",
        tts: "valtec_multispeaker",
        voice: "NF",
      },
      {
        key: "qwen-release",
        status: "ok",
        asr: "qwen3_asr_0_6b",
        llm: "arcee_vylinh_3b_q4_k_m",
        tts: "valtec_multispeaker",
        voice: "NF",
      },
    ];
    const baseProps = {
      config: {
        event: "llm_config" as const,
        backend: "local" as const,
        provider: "openrouter",
        model: "arcee_vylinh_3b_q4_k_m",
        max_tokens: 4096,
        temperature: 0.2,
        top_p: 0.95,
        pricing_as_of: "2026-08",
        pricing: null,
        context_length: 32768,
      },
      providers: [],
      profiles,
      activeProfile: "baseline",
      catalog: [],
      catalogProvider: "",
      keyPendingProvider: null,
      notice: "",
      onRequestModels: vi.fn(),
      onSetKey: vi.fn(),
      onSelect: vi.fn(),
      onProfileSelect,
      onExit,
    };
    const view = render(<SettingsScreen {...baseProps} />);
    const tick = () => new Promise((resolve) => setImmediate(resolve));

    await tick();
    view.stdin.write("a");
    await tick();
    view.stdin.write("\u001b[B");
    await tick();
    view.stdin.write("\r");
    await tick();

    expect(onProfileSelect).toHaveBeenCalledWith("qwen-release");
    expect(onExit).not.toHaveBeenCalled();
    expect(view.lastFrame()).toContain("Đang áp dụng qwen-release");
    expect(view.lastFrame()).toContain("←/→ chọn provider");

    view.rerender(
      <SettingsScreen
        {...baseProps}
        activeProfile="qwen-release"
      />,
    );
    await tick();
    expect(view.lastFrame()).toContain("qwen-release đã áp dụng.");
    expect(view.lastFrame()).not.toContain("Đang áp dụng qwen-release");
    view.unmount();
  });

  it("offers explicit vault initialization before voice setup", async () => {
    const onKnowledgeInit = vi.fn();
    const view = render(
      <SettingsScreen
        config={null}
        providers={providers}
        knowledgeVault={{
          path: "/workspace/Knowledge",
          initialized: false,
          index_home: "/workspace/Knowledge/.soca/knowledge_index",
        }}
        knowledgeIndex={null}
        knowledgeSetup={null}
        catalog={[]}
        catalogProvider=""
        keyPendingProvider={null}
        notice=""
        onRequestModels={vi.fn()}
        onSetKey={vi.fn()}
        onSelect={vi.fn()}
        onKnowledgeInit={onKnowledgeInit}
        onKnowledgeIndex={vi.fn()}
        onExit={vi.fn()}
      />,
    );

    await new Promise((resolve) => setImmediate(resolve));
    expect(view.lastFrame()).toContain("chưa init");
    view.stdin.write("\r");
    await new Promise((resolve) => setImmediate(resolve));
    expect(onKnowledgeInit).toHaveBeenCalledOnce();
    view.unmount();
  });

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

    const frame = view.lastFrame() ?? "";
    expect(frame).toContain("Cài đặt LLM");
    expect(frame).toContain("OpenRouter");
    expect(frame).toContain("Remote gửi transcript");
    expect(frame).toContain("128k");
    expect(frame).toContain("$0.15 / $0.60 / 1M");
    expect(frame).toContain("live");
    expect(frame.indexOf("Voice ASR")).toBeLessThan(frame.indexOf("Local"));
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
    view.stdin.write("e");
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
    view.stdin.write("e");
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

  it("configures model output and reasoning before applying", async () => {
    const onSelect = vi.fn();
    const view = render(
      <SettingsScreen
        config={{
          event: "llm_config",
          backend: "remote",
          provider: "openrouter",
          model: "openai/gpt-4o-mini",
          max_tokens: 4096,
          effective_max_tokens: 4096,
          reasoning_enabled: false,
          effective_reasoning_enabled: false,
          reasoning_supported: true,
          reasoning_mandatory: false,
          temperature: 0.2,
          top_p: 0.95,
          pricing_as_of: "2026-07",
          pricing: null,
          context_length: 128000,
        }}
        providers={providers}
        catalog={[
          {
            ...twoModelCatalog[0]!,
            max_output_tokens: 16384,
            reasoning_supported: true,
            reasoning_mandatory: false,
          },
        ]}
        catalogProvider="openrouter"
        keyPendingProvider={null}
        notice=""
        onRequestModels={vi.fn()}
        onSetKey={vi.fn()}
        onSelect={onSelect}
        onExit={vi.fn()}
      />,
    );

    const tick = () => new Promise((resolve) => setImmediate(resolve));
    await tick();
    view.stdin.write("e");
    await tick();
    view.stdin.write("\r");
    await tick();
    view.stdin.write("\u001b[B");
    await tick();
    view.stdin.write("\r");
    await tick();
    expect(view.lastFrame()).toContain("Generation");
    view.stdin.write("\u001b[3~");
    await tick();
    view.stdin.write("8192abc");
    await tick();
    expect(view.lastFrame()).toContain("8192");
    expect(view.lastFrame()).not.toContain("8192abc");
    view.stdin.write("\r");
    await tick();
    view.stdin.write(" ");
    await tick();
    view.stdin.write("\r");
    await tick();

    expect(onSelect).toHaveBeenCalledWith({
      backend: "remote",
      provider: "openrouter",
      model: "openai/gpt-4o-mini",
      max_tokens: 8192,
      reasoning_enabled: true,
    });
    view.unmount();
  });
});
