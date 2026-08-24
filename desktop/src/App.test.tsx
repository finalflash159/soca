// @vitest-environment jsdom

import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { EngineFrame } from "@/engine/protocol";

const tauri = vi.hoisted(() => ({
  listeners: new Map<string, (event: { payload: unknown }) => void>(),
  invoke: vi.fn(),
  listen: vi.fn(),
}));

vi.mock("@tauri-apps/api/core", () => ({ invoke: tauri.invoke }));
vi.mock("@tauri-apps/api/event", () => ({ listen: tauri.listen }));

import App from "./App";

const EVENT_CHANNEL = "soca://engine-event";
const STATUS_CHANNEL = "soca://engine-status";
const PERSISTENCE_STORAGE_KEY = "soca.session-persistence.v1";
const SIDEBAR_PREFERENCE_STORAGE_KEY = "soca.sidebar-open.v1";
const oldSessionId = "11111111-1111-4111-8111-111111111111";
const newSessionId = "22222222-2222-4222-8222-222222222222";
const otherSessionId = "33333333-3333-4333-8333-333333333333";

function emit(channel: string, payload: unknown): void {
  const listener = tauri.listeners.get(channel);
  if (listener === undefined) throw new Error(`listener not attached for ${channel}`);
  act(() => listener({ payload }));
}

function engineSendCommands(): Array<Record<string, unknown>> {
  return tauri.invoke.mock.calls
    .filter(([command]) => command === "engine_send")
    .map(([, arguments_]) => (arguments_ as { command: Record<string, unknown> }).command);
}

async function renderReadyApp(): Promise<void> {
  render(<App />);
  await waitFor(() => {
    expect(tauri.invoke).toHaveBeenCalledWith(
      "engine_start",
      expect.objectContaining({ options: expect.any(Object) }),
    );
  });
  emit(STATUS_CHANNEL, { state: "running" });
  emit(EVENT_CHANNEL, {
    event: "hello",
    version: 1,
    protocol_version: 3,
    supported_versions: [3],
    profile: "baseline",
    no_model: true,
    stack: {},
  });
  emit(EVENT_CHANNEL, {
    event: "llm_config",
    backend: "remote",
    provider: "openrouter",
    model: "openai/gpt-5",
    runtime_ready: true,
    runtime_reason: null,
    settings_error: null,
  });
  await waitFor(() => {
    expect((screen.getByRole("textbox", { name: "Message" }) as HTMLTextAreaElement).disabled).toBe(false);
  }, { timeout: 5_000 });
}

function oldSnapshot(nextTurnCursor: number | null = null): EngineFrame {
  return {
    event: "session_snapshot",
    session: {
      session_id: oldSessionId,
      title: "Phiên đang mở",
      preview: "Câu hỏi trước đó",
      updated_at: "2026-08-22T10:00:00Z",
      revision: 3,
      turn_count: 1,
      checkpoint_only: false,
    },
    turns: [
      {
        turn_id: "turn-old",
        sequence: 1,
        surface: "chat",
        user_text: "Câu hỏi trước đó",
        assistant_text: "Câu trả lời trước đó",
        status: "completed",
        terminal_status: "achieved",
      },
    ],
    next_turn_cursor: nextTurnCursor,
  } as EngineFrame;
}

function newSnapshot(): EngineFrame {
  return {
    event: "session_snapshot",
    session: {
      session_id: newSessionId,
      title: "Cuộc trò chuyện mới",
      preview: "",
      updated_at: "2026-08-22T10:05:00Z",
      revision: 1,
      turn_count: 0,
      checkpoint_only: false,
    },
    turns: [],
    next_turn_cursor: null,
  } as EngineFrame;
}

beforeEach(() => {
  tauri.listeners.clear();
  tauri.invoke.mockReset();
  tauri.listen.mockReset();
  tauri.invoke.mockResolvedValue(undefined);
  tauri.listen.mockImplementation(async (channel: string, callback: (event: { payload: unknown }) => void) => {
    tauri.listeners.set(channel, callback);
    return () => tauri.listeners.delete(channel);
  });
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    value: () => ({
      matches: false,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    }),
  });
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
  // Node's experimental Web Storage can replace JSDOM's object with a partial
  // implementation when its backing-file flag is unset. The app needs the
  // complete browser contract, so make the test boundary explicit.
  const items = new Map<string, string>();
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: {
      get length() {
        return items.size;
      },
      clear: () => items.clear(),
      getItem: (key: string) => items.get(key) ?? null,
      key: (index: number) => Array.from(items.keys())[index] ?? null,
      removeItem: (key: string) => items.delete(key),
      setItem: (key: string, value: string) => items.set(key, value),
    } satisfies Storage,
  });
});

afterEach(cleanup);

describe("desktop session lifecycle", () => {
  it("remembers an explicit desktop sidebar collapse without touching session data", async () => {
    await renderReadyApp();

    await userEvent.setup().click(screen.getByRole("button", { name: "Thu gọn thanh bên" }));

    expect(window.localStorage.getItem(SIDEBAR_PREFERENCE_STORAGE_KEY)).toBe("collapsed");
    expect(window.localStorage.getItem(PERSISTENCE_STORAGE_KEY)).toBeNull();
  });

  it("blocks the shell before it can send normal commands to an incompatible engine", async () => {
    render(<App />);
    await waitFor(() => {
      expect(tauri.invoke).toHaveBeenCalledWith(
        "engine_start",
        expect.objectContaining({ options: expect.any(Object) }),
      );
    });
    emit(STATUS_CHANNEL, { state: "running" });
    emit(EVENT_CHANNEL, {
      event: "hello",
      version: 1,
      protocol_version: 2,
      supported_versions: [2],
      profile: "baseline",
      no_model: true,
      stack: {},
    });

    await waitFor(() => {
      expect(screen.getByText(/engine speaks protocol 2/i)).not.toBeNull();
    });
    expect(engineSendCommands()).toEqual([]);
  });

  it("stops a possibly live sidecar before retrying from startup recovery", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitFor(() => {
      expect(tauri.invoke).toHaveBeenCalledWith(
        "engine_start",
        expect.objectContaining({ options: expect.any(Object) }),
      );
    });
    emit(STATUS_CHANNEL, { state: "failed", message: "engine already running" });

    await user.click(screen.getByRole("button", { name: "Khởi động" }));

    await waitFor(() => {
      expect(tauri.invoke).toHaveBeenCalledWith("engine_stop");
      expect(
        tauri.invoke.mock.calls.filter(([command]) => command === "engine_start"),
      ).toHaveLength(2);
    });
  });

  it("opens Voice even when setup is incomplete and offers a direct setup action", async () => {
    const user = userEvent.setup();
    await renderReadyApp();
    emit(EVENT_CHANNEL, {
      event: "status",
      runtime_components: [
        { id: "voice_asr", label: "Voice ASR", status: "missing", detail: "qwen-asr" },
        { id: "voice_llm", label: "Voice LLM", status: "ready", detail: "remote · openrouter:gpt" },
        { id: "tts", label: "TTS", status: "missing", detail: "valtec" },
      ],
    });

    await user.click(screen.getByRole("button", { name: "Thoại" }));

    await waitFor(() => {
      expect(screen.getByText("Thiết lập Voice trước khi bật microphone")).not.toBeNull();
      expect(
        screen.getByText("Qwen ASR chưa sẵn sàng."),
      ).not.toBeNull();
    });
    expect(screen.getByTestId("voice-orb").getAttribute("data-voice-orb-mode")).toBe("setup");
    expect(engineSendCommands().some((command) => command.cmd === "voice_start")).toBe(false);

    await user.click(screen.getByRole("button", { name: "Mở thiết lập Voice" }));
    await waitFor(() => {
      expect(screen.getByText("Trạng thái thoại")).not.toBeNull();
    });
  });

  it("waits for remote verification without sending a provisioned Qwen install to Settings", async () => {
    const user = userEvent.setup();
    await renderReadyApp();
    emit(EVENT_CHANNEL, {
      event: "llm_config",
      backend: "remote",
      provider: "openrouter",
      model: "openai/gpt-5",
      runtime_ready: false,
      runtime_state: "checking",
      runtime_reason: "Đang tải danh mục model của OpenRouter…",
      settings_error: null,
    });
    emit(EVENT_CHANNEL, {
      event: "status",
      runtime_components: [
        { id: "voice_asr", label: "Voice ASR", status: "ok", detail: "Qwen verified" },
        { id: "voice_llm", label: "Voice LLM", status: "missing", detail: "catalog loading" },
        { id: "tts", label: "TTS", status: "ready", detail: "valtec" },
      ],
    });

    await user.click(screen.getByRole("button", { name: "Thoại" }));

    await waitFor(() => {
      expect(screen.getByText("Đang chuẩn bị Voice…")).not.toBeNull();
    });
    expect(screen.queryByRole("button", { name: "Mở thiết lập Voice" })).toBeNull();
    expect((screen.getByRole("button", { name: "Bật mic" }) as HTMLButtonElement).disabled).toBe(true);

    emit(EVENT_CHANNEL, {
      event: "llm_config",
      backend: "remote",
      provider: "openrouter",
      model: "openai/gpt-5",
      runtime_ready: true,
      runtime_state: "ready",
      runtime_reason: null,
      settings_error: null,
    });

    await waitFor(() => {
      expect((screen.getByRole("button", { name: "Bật mic" }) as HTMLButtonElement).disabled).toBe(false);
    });
    expect(screen.queryByRole("button", { name: "Mở thiết lập Voice" })).toBeNull();
  });

  it("uses the immersive orb only while the microphone is capturing", async () => {
    const user = userEvent.setup();
    await renderReadyApp();
    emit(EVENT_CHANNEL, {
      event: "status",
      runtime_components: [
        { id: "voice_asr", label: "Voice ASR", status: "ready", detail: "qwen" },
        { id: "voice_llm", label: "Voice LLM", status: "ready", detail: "remote" },
        { id: "tts", label: "TTS", status: "ready", detail: "valtec" },
      ],
    });

    await user.click(screen.getByRole("button", { name: "Thoại" }));
    emit(EVENT_CHANNEL, { event: "voice", type: "recording", metadata: {} });
    await waitFor(() => {
      expect(screen.getByTestId("voice-orb").getAttribute("data-voice-orb-presentation")).toBe("immersive");
    });

    emit(EVENT_CHANNEL, {
      event: "voice",
      type: "asr_partial",
      metadata: { committed: "xin chào", tentative: " SoCa" },
    });
    expect(screen.queryByText("xin chào")).toBeNull();

    emit(EVENT_CHANNEL, { event: "voice", type: "recorded", metadata: {} });
    await waitFor(() => {
      expect(screen.getByTestId("voice-orb").getAttribute("data-voice-orb-presentation")).toBe("compact");
      expect(screen.getByText("xin chào")).not.toBeNull();
    });
  });

  it("loads older transcript pages with the engine's exclusive sequence boundary", async () => {
    const user = userEvent.setup();
    await renderReadyApp();
    emit(EVENT_CHANNEL, oldSnapshot(0));

    await user.click(screen.getByRole("button", { name: "Tải lượt cũ hơn" }));
    await waitFor(() => {
      expect(
        engineSendCommands().some(
          (command) => command.cmd === "session_turns" && command.before_sequence === 0,
        ),
      ).toBe(true);
    });
    expect(screen.getByText("Câu hỏi trước đó")).not.toBeNull();

    emit(EVENT_CHANNEL, {
      ...oldSnapshot(null),
      event: "session_turns_page",
      turns: [
        {
          turn_id: "turn-earlier",
          sequence: 0,
          surface: "voice",
          user_text: "Câu hỏi cũ hơn",
          assistant_text: "Câu trả lời cũ hơn",
          status: "completed",
          terminal_status: "achieved",
        },
        ...((oldSnapshot() as { turns: Array<Record<string, unknown>> }).turns),
      ],
    });

    await waitFor(() => {
      expect(screen.getByText("Câu hỏi cũ hơn")).not.toBeNull();
      expect(screen.getByText("Câu hỏi trước đó")).not.toBeNull();
    });
  });

  it("keeps the active transcript until session_create completes and its replacement snapshot arrives", async () => {
    const user = userEvent.setup();
    await renderReadyApp();
    emit(EVENT_CHANNEL, oldSnapshot());
    expect(screen.getByText("Câu hỏi trước đó")).not.toBeNull();

    await user.click(screen.getByRole("button", { name: "Cuộc trò chuyện mới" }));
    await waitFor(() => {
      expect(engineSendCommands().some((command) => command.cmd === "session_create")).toBe(true);
    });
    const create = engineSendCommands().find((command) => command.cmd === "session_create");
    const requestId = create?.request_id;
    expect(typeof requestId).toBe("string");

    emit(EVENT_CHANNEL, {
      event: "session_operation",
      request_id: requestId,
      action: "create",
      status: "started",
      session_id: null,
      revision: null,
      error_code: null,
    });
    expect(screen.getByText("Câu hỏi trước đó")).not.toBeNull();

    emit(EVENT_CHANNEL, {
      event: "session_operation",
      request_id: requestId,
      action: "create",
      status: "completed",
      session_id: newSessionId,
      revision: 1,
      error_code: null,
    });
    expect(screen.getByText("Câu hỏi trước đó")).not.toBeNull();

    emit(EVENT_CHANNEL, newSnapshot());
    await waitFor(() => {
      expect(screen.queryByText("Câu hỏi trước đó")).toBeNull();
      expect(document.activeElement).toBe(screen.getByRole("textbox", { name: "Message" }));
    });
  });

  it("retains the active transcript and announces a typed failure when creation is rejected", async () => {
    const user = userEvent.setup();
    await renderReadyApp();
    emit(EVENT_CHANNEL, oldSnapshot());

    await user.click(screen.getByRole("button", { name: "Cuộc trò chuyện mới" }));
    await waitFor(() => {
      expect(engineSendCommands().some((command) => command.cmd === "session_create")).toBe(true);
    });
    const create = engineSendCommands().find((command) => command.cmd === "session_create");

    emit(EVENT_CHANNEL, {
      event: "session_operation",
      request_id: create?.request_id,
      action: "create",
      status: "failed",
      session_id: null,
      revision: null,
      error_code: "DiskFullError",
    });

    await waitFor(() => {
      expect(screen.getByText("Câu hỏi trước đó")).not.toBeNull();
      expect(screen.getByRole("alert").textContent).toContain("DiskFullError");
    });
  });

  it("re-enables the current session after deleting a non-active saved session", async () => {
    const user = userEvent.setup();
    await renderReadyApp();
    emit(EVENT_CHANNEL, oldSnapshot());
    emit(EVENT_CHANNEL, {
      event: "sessions_page",
      persistence: "local_resumable",
      next_cursor: null,
      sessions: [
        {
          session_id: oldSessionId,
          title: "Phiên đang mở",
          preview: "",
          updated_at: "2026-08-22T10:00:00Z",
          revision: 3,
          turn_count: 1,
          checkpoint_only: false,
        },
        {
          session_id: otherSessionId,
          title: "Phiên cần xóa",
          preview: "",
          updated_at: "2026-08-21T10:00:00Z",
          revision: 2,
          turn_count: 1,
          checkpoint_only: false,
        },
      ],
    });

    await user.click(screen.getByRole("button", { name: /thao tác cho phiên phiên cần xóa/i }));
    await user.click(await screen.findByRole("menuitem", { name: /xóa vĩnh viễn/i }));
    await user.click(screen.getByRole("button", { name: "Xóa vĩnh viễn" }));
    await waitFor(() => {
      expect(engineSendCommands().some((command) => command.cmd === "session_delete")).toBe(true);
    });
    const deletion = engineSendCommands().find((command) => command.cmd === "session_delete");

    emit(EVENT_CHANNEL, {
      event: "session_operation",
      request_id: deletion?.request_id,
      action: "delete",
      status: "started",
      session_id: null,
      revision: null,
      error_code: null,
    });
    expect((screen.getByRole("textbox", { name: "Message" }) as HTMLTextAreaElement).disabled).toBe(true);

    emit(EVENT_CHANNEL, {
      event: "session_operation",
      request_id: deletion?.request_id,
      action: "delete",
      status: "completed",
      session_id: otherSessionId,
      revision: 2,
      error_code: null,
    });
    await waitFor(() => {
      expect((screen.getByRole("textbox", { name: "Message" }) as HTMLTextAreaElement).disabled).toBe(false);
      expect(screen.getByText("Câu hỏi trước đó")).not.toBeNull();
    });
  });

  it("requires consent, saves only the launcher choice, and restarts for local session storage", async () => {
    const user = userEvent.setup();
    await renderReadyApp();
    emit(EVENT_CHANNEL, {
      event: "session_preferences",
      persistence: "ram_only",
      auto_open_last: false,
      last_active_session_id: null,
    });

    await user.click(screen.getByRole("button", { name: "Cài đặt" }));
    await user.click(await screen.findByRole("button", { name: "Bật lưu phiên" }));
    expect(screen.getByRole("dialog").textContent).toContain("Bật lưu phiên trên máy?");
    await user.click(screen.getByRole("button", { name: "Đồng ý và khởi động lại" }));

    await waitFor(() => {
      expect(tauri.invoke).toHaveBeenCalledWith("engine_stop");
      expect(tauri.invoke).toHaveBeenCalledWith(
        "engine_start",
        expect.objectContaining({
          options: expect.objectContaining({ args: ["--session-persistence", "local_resumable"] }),
        }),
      );
    });
    expect(window.localStorage.getItem(PERSISTENCE_STORAGE_KEY)).toBe("local_resumable");
  });
});
