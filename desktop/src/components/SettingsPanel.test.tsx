// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ComponentProps } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { SettingsPanel } from "./SettingsPanel";
import { initialSettings, type SettingsState } from "@/engine/settings";
import type { SessionHistoryState } from "@/engine/session-history";

const sessionHistory: SessionHistoryState = {
  sessions: [],
  nextCursor: null,
  listState: "ready",
  listError: null,
  snapshotError: null,
  activeSessionId: "session-1",
  active: null,
  persistence: "ram_only",
  autoOpenLast: false,
  busy: false,
  operation: null,
};

const localConfig = {
  backend: "local",
  provider: "openrouter",
  model: "arcee_vylinh_3b_q4_k_m",
  maxTokens: 4096,
  effectiveMaxTokens: 4096,
  reasoningEnabled: false,
  effectiveReasoningEnabled: false,
  reasoningSupported: false,
  reasoningMandatory: false,
  contextLength: 8192,
  runtimeReady: true,
  runtimeReason: null,
  localModelPath: "/models/model.gguf",
  settingsError: null,
};

const remoteModel = {
  id: "openai/gpt-4o-mini",
  label: "GPT-4o mini",
  context_length: 128000,
  price_prompt_per_1m: 0.15,
  price_completion_per_1m: 0.6,
  pricing_source: "table",
  max_output_tokens: 16384,
  reasoning_supported: false,
  reasoning_mandatory: false,
};

function renderPanel(
  overrides: Partial<ComponentProps<typeof SettingsPanel>> = {},
) {
  const settings: SettingsState = {
    ...initialSettings,
    providers: [
      {
        key: "openrouter",
        label: "OpenRouter",
        hasKey: true,
        hasPricingApi: true,
      },
    ],
    config: localConfig,
    catalog: { openrouter: [remoteModel] },
    catalogLoading: { openrouter: false },
    ...overrides.settings,
  };
  const { settings: _overriddenSettings, ...restOverrides } = overrides;
  const props: ComponentProps<typeof SettingsPanel> = {
    connected: true,
    themeChoice: "dark",
    onSetTheme: vi.fn(),
    onLoadProviders: vi.fn(),
    onSetKey: vi.fn(),
    onLoadModels: vi.fn(),
    onSelectModel: vi.fn(async () => true),
    onSelectProfile: vi.fn(async () => true),
    onApplyGeneration: vi.fn(async () => true),
    modelRoot: { path: "/models", source: "managed" },
    onSetModelRoot: vi.fn(async () => null),
    engineError: null,
    sessionHistory,
    persistenceChangePending: false,
    onRequestSessionPersistence: vi.fn(),
    onSetAutoOpenLast: vi.fn(),
    ...restOverrides,
    settings,
  };
  return { ...render(<SettingsPanel {...props} />), props };
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("SettingsPanel remote configuration", () => {
  it("opens remote setup without applying a local GGUF id as a remote model", async () => {
    const user = userEvent.setup();
    const { props } = renderPanel();

    await user.click(screen.getByRole("button", { name: "Remote" }));

    expect(screen.getByText(/chưa áp dụng — hãy chọn model remote/i)).toBeTruthy();
    expect(screen.getByText(/chỉ chuyển sang remote sau khi/i)).toBeTruthy();
    expect(props.onApplyGeneration).not.toHaveBeenCalled();
  });

  it("commits remote only from a provider catalog selection", async () => {
    const user = userEvent.setup();
    const { props } = renderPanel();

    await user.click(screen.getByRole("button", { name: "Remote" }));
    await user.click(screen.getByRole("button", { name: "Chọn" }));

    expect(props.onSelectModel).toHaveBeenCalledWith("openrouter", remoteModel.id);
  });

  it("shows remote readiness failure as configuration state, not as local model failure", () => {
    renderPanel({
      settings: {
        ...initialSettings,
        providers: [
          { key: "openrouter", label: "OpenRouter", hasKey: true, hasPricingApi: true },
        ],
        config: {
          ...localConfig,
          backend: "remote",
          model: remoteModel.id,
          localModelPath: null,
          runtimeReady: false,
          runtimeReason: "Đang tải danh mục model của OpenRouter…",
        },
        catalog: { openrouter: [] },
        catalogLoading: { openrouter: true },
      },
    });

    expect(screen.getByText(/đang tải danh mục model/i)).toBeTruthy();
    expect(screen.queryByText(/chưa tìm thấy model local/i)).toBeNull();
  });

  it("shows each voice dependency instead of presenting an unavailable microphone as ready", () => {
    renderPanel({
      settings: {
        ...initialSettings,
        config: localConfig,
        runtimeComponents: [
          { id: "chat_llm", label: "Chat LLM", status: "ready", detail: "local · chat.gguf" },
          { id: "voice_asr", label: "Voice ASR", status: "missing", detail: "qwen-asr" },
          { id: "voice_llm", label: "Voice LLM", status: "ready", detail: "remote · openrouter:gpt" },
          { id: "tts", label: "TTS", status: "missing", detail: "valtec" },
        ],
      },
    });

    expect(screen.getByText("Trạng thái thoại")).toBeTruthy();
    expect(
      screen.getByText("Speech recognition is not installed for the selected voice profile."),
    ).toBeTruthy();
  });
});
