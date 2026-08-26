import { describe, expect, it } from "vitest";

import type { EngineFrame } from "./protocol";
import { initialSettings, modelPrice, reasoningSummary, reduceSettings } from "./settings";

function fold(frames: EngineFrame[]) {
  return frames.reduce(reduceSettings, initialSettings);
}

const model = {
  id: "openai/gpt-5.6-luna",
  label: "GPT-5.6 Luna",
  context_length: 200000,
  price_prompt_per_1m: 1.25,
  price_completion_per_1m: 10,
  pricing_source: "api",
  max_output_tokens: 8192,
  reasoning_supported: true,
  reasoning_mandatory: false,
};

describe("providers", () => {
  it("reads the list without ever holding key material", () => {
    const state = fold([
      {
        event: "llm_providers",
        providers: [{ key: "openai", label: "OpenAI", has_key: true, has_pricing_api: false }],
      } as EngineFrame,
    ]);
    expect(state.providers[0]).toEqual({
      key: "openai",
      label: "OpenAI",
      hasKey: true,
      hasPricingApi: false,
    });
    expect(JSON.stringify(state)).not.toContain("sk-");
  });
});

describe("catalog", () => {
  it("treats the first empty frame as loading, not as no models", () => {
    const state = fold([
      {
        event: "llm_catalog",
        provider: "openai",
        models: [],
        loading: true,
        pricing_as_of: "2026-07",
      } as EngineFrame,
    ]);
    expect(state.catalogLoading.openai).toBe(true);
    expect(state.catalog.openai).toEqual([]);
    expect(state.pricingAsOf).toBe("2026-07");
  });

  it("clears the loading flag when the real catalog arrives", () => {
    const state = fold([
      { event: "llm_catalog", provider: "openai", models: [] } as EngineFrame,
      { event: "llm_catalog", provider: "openai", models: [model], loading: false } as EngineFrame,
    ]);
    expect(state.catalogLoading.openai).toBe(false);
    expect(state.catalog.openai).toHaveLength(1);
  });

  it("keeps catalogs separate per provider", () => {
    const state = fold([
      { event: "llm_catalog", provider: "openai", models: [model] } as EngineFrame,
      { event: "llm_catalog", provider: "groq", models: [], loading: true } as EngineFrame,
    ]);
    expect(state.catalog.openai).toHaveLength(1);
    expect(state.catalogLoading.groq).toBe(true);
  });

  it("formats a price only when the provider publishes one", () => {
    expect(modelPrice(model)).toBe("$1.25 / $10.00 per 1M");
    expect(modelPrice({ ...model, price_prompt_per_1m: null })).toBeNull();
  });
});

describe("key status", () => {
  it("carries pending through without claiming success", () => {
    const state = fold([
      { event: "llm_key_status", provider: "openai", ok: false, pending: true } as EngineFrame,
    ]);
    expect(state.keyStatus.openai.pending).toBe(true);
    expect(state.keyStatus.openai.ok).toBe(false);
  });

  it("marks the provider as keyed once validation succeeds", () => {
    const state = fold([
      {
        event: "llm_providers",
        providers: [{ key: "openai", label: "OpenAI", has_key: false, has_pricing_api: false }],
      } as EngineFrame,
      { event: "llm_key_status", provider: "openai", ok: true, masked: "sk-…4f2a" } as EngineFrame,
    ]);
    expect(state.providers[0].hasKey).toBe(true);
    expect(state.keyStatus.openai.masked).toBe("sk-…4f2a");
  });

  it("does not mark the provider keyed on a failed validation", () => {
    const state = fold([
      {
        event: "llm_providers",
        providers: [{ key: "groq", label: "Groq", has_key: false, has_pricing_api: false }],
      } as EngineFrame,
      {
        event: "llm_key_status",
        provider: "groq",
        ok: false,
        message: "unauthorized",
      } as EngineFrame,
    ]);
    expect(state.providers[0].hasKey).toBe(false);
    expect(state.keyStatus.groq.message).toBe("unauthorized");
  });
});

describe("reasoning", () => {
  const base = {
    event: "llm_config",
    backend: "remote",
    provider: "openai",
    model: "m",
    runtime_ready: true,
    settings_error: null,
  };

  it("reports the effective value when it differs from the request", () => {
    const state = fold([
      {
        ...base,
        reasoning_supported: true,
        reasoning_mandatory: false,
        reasoning_enabled: false,
        effective_reasoning_enabled: true,
      } as EngineFrame,
    ]);
    expect(reasoningSummary(state.config)).toBe("requested off, effective on");
  });

  it("says a mandatory model cannot be turned off", () => {
    const state = fold([
      {
        ...base,
        reasoning_supported: true,
        reasoning_mandatory: true,
        reasoning_enabled: false,
        effective_reasoning_enabled: true,
      } as EngineFrame,
    ]);
    expect(reasoningSummary(state.config)).toContain("requires it");
  });

  it("says unsupported rather than off", () => {
    const state = fold([
      {
        ...base,
        reasoning_supported: false,
        reasoning_mandatory: false,
        reasoning_enabled: false,
        effective_reasoning_enabled: false,
      } as EngineFrame,
    ]);
    expect(reasoningSummary(state.config)).toContain("not supported");
  });

  it("is unknown before any config arrives", () => {
    expect(reasoningSummary(null)).toBe("unknown");
  });
});

describe("config", () => {
  it("surfaces a settings error and a not-ready runtime", () => {
    const state = fold([
      {
        event: "llm_config",
        backend: "remote",
        provider: "openai",
        model: "m",
        runtime_ready: false,
        settings_error: "llm.json is invalid",
      } as EngineFrame,
    ]);
    expect(state.config?.runtimeReady).toBe(false);
    expect(state.config?.settingsError).toBe("llm.json is invalid");
  });
});

describe("voice profiles", () => {
  it("marks the profile confirmed by the engine as the effective profile", () => {
    const state = fold([
      {
        event: "status",
        active_profile: "quiet",
        profiles: [
          {
            key: "quiet",
            status: "ok",
            asr: "a",
            tts: "t",
            voice: "v",
            note: "selected profile is ready",
          },
        ],
      } as EngineFrame,
    ]);

    expect(state.activeProfile).toBe("quiet");
    expect(state.profiles[0].key).toBe("quiet");
    expect(state.profiles[0].note).toBe("selected profile is ready");
  });
});

describe("runtime profiles", () => {
  it("reads them from the status frame", () => {
    const state = fold([
      {
        event: "status",
        profiles: [
          {
            key: "qwen-release",
            status: "ok",
            asr: "qwen3_asr_0_6b",
            llm: "x",
            tts: "valtec",
            voice: "v",
          },
          { key: "baseline", status: "blocked", asr: "phowhisper_small" },
        ],
      } as EngineFrame,
    ]);
    expect(state.profiles).toHaveLength(2);
    expect(state.profiles[0].asr).toBe("qwen3_asr_0_6b");
    expect(state.profiles[1].tts).toBeNull();
  });
});

describe("runtime components", () => {
  it("preserves typed readiness details for the settings and voice entry gates", () => {
    const state = fold([
      {
        event: "status",
        runtime_components: [
          { id: "chat_llm", label: "Chat LLM", status: "ready", detail: "remote · openrouter:gpt" },
          { id: "voice_asr", label: "Voice ASR", status: "missing", detail: "qwen-asr" },
          { id: "tts", label: "TTS", status: "ready", detail: "valtec" },
        ],
      } as EngineFrame,
    ]);

    expect(state.runtimeComponents).toEqual([
      { id: "chat_llm", label: "Chat LLM", status: "ready", detail: "remote · openrouter:gpt" },
      { id: "voice_asr", label: "Voice ASR", status: "missing", detail: "qwen-asr" },
      { id: "tts", label: "TTS", status: "ready", detail: "valtec" },
    ]);
  });
});

describe("unknown frames", () => {
  it("are ignored", () => {
    expect(fold([{ event: "memory" } as EngineFrame])).toEqual(initialSettings);
  });
});
