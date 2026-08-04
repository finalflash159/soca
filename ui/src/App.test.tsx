import React from "react";
import { render } from "ink-testing-library";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { App, Splash, defaultSettingsReturnMode } from "./App.js";

const harness = vi.hoisted(() => ({
  sent: [] as Array<Record<string, unknown>>,
  compactionPolls: 0,
}));

vi.mock("./engine.js", () => {
  class EngineClient {
    private listeners: Record<string, Array<(value: unknown) => void>> = {};

    on(name: string, listener: (value: unknown) => void): this {
      this.listeners[name] = [...(this.listeners[name] ?? []), listener];
      return this;
    }

    private emit(name: string, value: unknown): void {
      for (const listener of this.listeners[name] ?? []) listener(value);
    }

    start(): void {
      this.emit("event", {
        event: "hello",
        version: 1,
        profile: "baseline",
        no_model: true,
        stack: { llm: "arcee_vylinh_3b_q4_k_m" },
      });
      this.emit("event", {
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
        components: [
          {
            id: "system",
            label: "System instructions",
            tokens: 120,
            policy: "always",
          },
          {
            id: "knowledge",
            label: "Knowledge retrieval",
            tokens: null,
            policy: "on_demand",
          },
        ],
      });
    }

    send(command: Record<string, unknown>): void {
      harness.sent.push(command);
      if (command["cmd"] === "memory_proposals") {
        this.emit("event", {
          event: "memory_proposals",
          proposals: [],
        });
      } else if (
        command["cmd"] === "memory_compact" &&
        command["action"] === "request"
      ) {
        this.emit("event", {
          event: "memory_compaction",
          status: "accepted",
          generation: 2,
          detail: "",
          before_tokens: 1200,
          after_tokens: null,
          compacted_turns: 4,
          complete_turns: 6,
          minimum_complete_turns: 5,
          elapsed_ms: 2,
        });
      } else if (
        command["cmd"] === "memory_compact" &&
        command["action"] === "status"
      ) {
        harness.compactionPolls += 1;
        this.emit("event", {
          event: "memory_compaction",
          status: "published",
          generation: 2,
          detail: "",
          before_tokens: 1200,
          after_tokens: 760,
          compacted_turns: 4,
          complete_turns: 6,
          minimum_complete_turns: 5,
          elapsed_ms: 420,
        });
      } else if (command["cmd"] === "memory") {
        this.emit("event", {
          event: "memory",
          enabled: true,
          text: "Earlier conversation state:\nSummary:\nĐã giữ quyết định TTS local.",
          summary:
            "Earlier conversation state:\nSummary:\nĐã giữ quyết định TTS local.",
          recent: "Recent conversation:\nUser: câu mới",
          stats: {
            current_tokens: 760,
            rendered_tokens: 760,
            hard_limit_tokens: 16384,
            high_watermark_tokens: 15000,
            target_tokens: 12000,
            summary_tokens: 80,
            recent_tokens: 680,
            turn_count: 2,
            complete_turn_count: 2,
            summary_generation: 2,
            pending_compaction: false,
            worker_state: "idle",
          },
        });
      } else if (command["cmd"] === "voice_start") {
        // Mid-turn: chunk 0 is being played, chunk 1 is already synthesized.
        this.emit("event", {
          event: "voice",
          type: "turn_start",
          text: "",
          latency_ms: null,
          metadata: { turn: 1 },
          usage: null,
        });
        this.emit("event", {
          event: "voice",
          type: "tts",
          text: "Bayes là quy tắc cập nhật niềm tin.",
          latency_ms: 210,
          metadata: { chunk_index: 0, delivery: "final" },
          usage: null,
        });
        this.emit("event", {
          event: "voice",
          type: "playback_started",
          text: "Bayes là quy tắc cập nhật niềm tin.",
          latency_ms: null,
          metadata: {
            chunk_index: 0,
            delivery: "final",
            audio_duration_ms: 1500,
            sync_granularity: "audio_chunk",
          },
          usage: null,
        });
        this.emit("event", {
          event: "voice",
          type: "tts",
          text: "Bạn muốn mình ví dụ không?",
          latency_ms: 180,
          metadata: { chunk_index: 1, delivery: "final" },
          usage: null,
        });
      } else if (command["cmd"] === "voice_stop") {
        // Playback drains, then the turn closes.
        this.emit("event", {
          event: "voice",
          type: "audio",
          text: "Bayes là quy tắc cập nhật niềm tin.",
          latency_ms: 1500,
          metadata: { chunk_index: 0, delivery: "final" },
          usage: null,
        });
        this.emit("event", {
          event: "voice",
          type: "playback_started",
          text: "Bạn muốn mình ví dụ không?",
          latency_ms: null,
          metadata: {
            chunk_index: 1,
            delivery: "final",
            audio_duration_ms: 900,
            sync_granularity: "audio_chunk",
          },
          usage: null,
        });
        this.emit("event", {
          event: "voice",
          type: "audio",
          text: "Bạn muốn mình ví dụ không?",
          latency_ms: 900,
          metadata: { chunk_index: 1, delivery: "final" },
          usage: null,
        });
        this.emit("event", {
          event: "voice",
          type: "done",
          text: "Bayes là quy tắc cập nhật niềm tin. Bạn muốn mình ví dụ không?",
          latency_ms: 3200,
          metadata: { runtime_route: "llm", citations: [] },
          usage: null,
        });
      } else if (command["cmd"] === "usage") {
        this.emit("event", {
          event: "usage",
          turns: 1,
          llm_turns: 1,
          prompt_tokens: 682,
          completion_tokens: 128,
          mean_ttft_ms: 6354,
          mean_tokens_per_second: 20.1,
        });
      } else if (command["cmd"] === "chat") {
        this.emit("event", {
          event: "turn_started",
          protocol_version: 2,
          session_id: "session",
          run_id: "run-catalog",
          goal_id: "goal-catalog",
          sequence: 0,
          surface: "chat",
          timestamp: "2026-07-30T00:00:00Z",
          node: "admit",
          status: "started",
          payload: {},
        });
        this.emit("event", {
          event: "retrieval_trace",
          query: String(command["text"] ?? ""),
          tier: "semantic",
          latency_ms: 4.2,
          columns: [
            {
              source: "hybrid",
              hits: [{ path: "wiki/learning/attention.md", score: 0.91 }],
            },
          ],
          fused: [
            {
              path: "wiki/learning/attention.md",
              picked: true,
              backend: "hybrid",
              score: 0.91,
            },
          ],
          rejected_count: 0,
          evidence: { status: "supported", reason: "accepted" },
        });
        this.emit("event", {
          event: "turn_progress",
          surface: "chat",
          phase: "retrieval",
          operation: "tool:knowledge.search",
          status: "active",
        });
        this.emit("event", {
          event: "turn_terminal",
          protocol_version: 2,
          session_id: "session",
          run_id: "run-catalog",
          goal_id: "goal-catalog",
          sequence: 1,
          surface: "chat",
          timestamp: "2026-07-30T00:00:01Z",
          node: "finalize",
          status: "completed",
          payload: { terminal_status: "achieved", final_text: "xong" },
        });
      }
    }

    stop(): void {}
  }

  return { EngineClient };
});

async function tick(): Promise<void> {
  await new Promise((resolve) => setImmediate(resolve));
}

function stripAnsi(value: string): string {
  return value.replace(/\u001b\[[0-?]*[ -/]*[@-~]/g, "");
}

describe("App slash command interaction", () => {
  beforeEach(() => {
    harness.sent.length = 0;
    harness.compactionPolls = 0;
  });

  it("returns bare setup to voice while keeping chat explicit", () => {
    expect(defaultSettingsReturnMode("settings")).toBe("voice");
    expect(defaultSettingsReturnMode("voice")).toBe("voice");
    expect(defaultSettingsReturnMode("chat")).toBe("chat");
    expect(defaultSettingsReturnMode("status")).toBe("chat");
  });

  it("keeps the cold launch on the existing main splash", async () => {
    const onDone = vi.fn();
    const view = render(<Splash onDone={onDone} />);
    await tick();

    expect(view.lastFrame()).toContain("asr · llm · tts · barge-in");
    expect(view.lastFrame()).toContain("↵ chat");
    expect(view.lastFrame()).toContain("v voice");

    view.stdin.write("\r");
    expect(onDone).toHaveBeenCalledWith("chat");
    view.unmount();
  });

  it("does not strand the splash when raw terminal input is unavailable", async () => {
    const onDone = vi.fn();
    const view = render(<Splash onDone={onDone} rawModeSupported={false} />);
    await tick();

    expect(onDone).toHaveBeenCalledWith("chat");
    view.unmount();
  });

  it("opens the bare UI on the main chat surface, not settings", async () => {
    const view = render(<App target="chat" setupFirst={false} noModel />);
    await tick();

    expect(view.lastFrame()).toContain("╭─chat");
    expect(view.lastFrame()).not.toContain("Cài đặt LLM");
    view.unmount();
  });

  it("filters and selects commands, then dismisses info on new input", async () => {
    const view = render(<App target="status" noModel />);
    await tick();

    expect(view.lastFrame()).toContain("runtime status");
    expect(view.lastFrame()).toContain("~1.20k / 16.4k tok");

    view.stdin.write("/");
    await tick();
    expect(view.lastFrame()).toContain("slash commands");
    expect(view.lastFrame()).toContain("/context");
    expect(view.lastFrame()).toContain("/memory-proposals");
    expect(view.lastFrame()).not.toContain("/inspect");

    for (let index = 0; index < 4; index += 1) {
      view.stdin.write("\u001b[B");
      await tick();
    }
    view.stdin.write("\r");
    await tick();

    expect(harness.sent).toContainEqual({ cmd: "context" });
    expect(harness.sent).toContainEqual({ cmd: "usage" });
    expect(view.lastFrame()).toContain("context & usage");
    expect(view.lastFrame()).toContain("Knowledge retrieval");
    expect(view.lastFrame()).toContain("TTFT 6.35 s");

    view.stdin.write("x");
    await tick();
    expect(view.lastFrame()).not.toContain("context & usage");
    expect(view.lastFrame()).toContain("x");
    expect(view.lastFrame()).not.toContain("No messages yet.");
    const promptBorder = stripAnsi(view.lastFrame() ?? "")
      .split("\n")
      .find((line) => line.includes("chat") && line.includes("─"));
    expect(promptBorder).toBeDefined();
    expect(Array.from(promptBorder ?? "").length).toBeLessThan(100);
    view.unmount();
  });

  it("shows an explicit empty state for memory proposals", async () => {
    const view = render(<App target="status" noModel />);
    await tick();

    view.stdin.write("/memory-proposals");
    await tick();
    view.stdin.write("\r");
    await tick();

    expect(harness.sent).toContainEqual({ cmd: "memory_proposals" });
    expect(view.lastFrame()).toContain("No pending proposals");
    expect(view.lastFrame()).toContain("Esc to return");
    view.unmount();
  });

  it("renders the engine-reported turn phase instead of generic progress", async () => {
    const view = render(<App target="status" noModel />);
    await tick();

    view.stdin.write("ghi chú nói gì về Bayes?");
    await tick();
    view.stdin.write("\r");
    await tick();

    expect(harness.sent).toContainEqual({
      cmd: "chat",
      text: "ghi chú nói gì về Bayes?",
    });
    expect(view.lastFrame()).toContain("Running knowledge.search…");
    expect(view.lastFrame()).not.toContain("SoCa đang xử lý");
    expect(view.lastFrame()).not.toContain("✓");
    expect(view.lastFrame()).not.toContain("SoCa đang soạn");
    view.unmount();
  });

  it("keeps workflow and retrieval diagnostics hidden until their commands", async () => {
    const view = render(<App target="status" noModel />);
    await tick();

    view.stdin.write("attention");
    await tick();
    view.stdin.write("\r");
    await tick();
    expect(view.lastFrame()).not.toContain("terminal trace is retained");
    expect(view.lastFrame()).not.toContain("backend hybrid");

    view.stdin.write("/workflow");
    await tick();
    view.stdin.write("\r");
    await tick();
    expect(view.lastFrame()).toContain("turn started");

    view.stdin.write("transformer");
    await tick();
    expect(view.lastFrame()).not.toContain("turn started");
    view.stdin.write("\r");
    await tick();
    expect(view.lastFrame()).not.toContain("turn started");

    view.stdin.write("/retrieval");
    await tick();
    view.stdin.write("\r");
    await tick();
    expect(view.lastFrame()).toContain("backend hybrid");
    view.unmount();
  });

  it("captions the answer while speaking, then hands it to the timeline", async () => {
    const view = render(<App target="status" />);
    await tick();

    view.stdin.write("/voice");
    await tick();
    view.stdin.write("\r");
    await tick();

    expect(harness.sent).toContainEqual({ cmd: "voice_start" });
    const speaking = stripAnsi(view.lastFrame() ?? "");
    // The existing status row survives; the caption is additive.
    expect(speaking).toContain("speaking");
    expect(speaking).toContain("Bayes là quy tắc cập nhật niềm tin.");
    expect(speaking).toContain("Bạn muốn mình ví dụ không?");

    view.stdin.write("/stop");
    await tick();
    view.stdin.write("\r");
    await tick();

    const finished = stripAnsi(view.lastFrame() ?? "");
    expect(finished).not.toContain("speaking");
    expect(finished).toContain(
      "Bayes là quy tắc cập nhật niềm tin. Bạn muốn mình ví dụ không?",
    );
    view.unmount();
  });

  it("shows compaction progress, result metrics, and expandable summary", async () => {
    const view = render(<App target="status" noModel />);
    await tick();

    view.stdin.write("/compact");
    await tick();
    view.stdin.write("\r");
    await tick();

    expect(view.lastFrame()).toContain("đang compact");
    expect(view.lastFrame()).toContain("nguồn ~1.20k token");

    await new Promise((resolve) => setTimeout(resolve, 320));
    expect(harness.compactionPolls).toBeGreaterThan(0);
    expect(view.lastFrame()).toContain("Compact hoàn tất");
    expect(view.lastFrame()).toContain("~1.20k → ~760 token");
    expect(view.lastFrame()).toContain("/compact-show");

    view.stdin.write("/compact-show");
    await tick();
    view.stdin.write("\r");
    await tick();

    expect(view.lastFrame()).toContain("compacted summary");
    expect(view.lastFrame()).toContain("Đã giữ quyết định TTS local.");

    view.stdin.write("x");
    await tick();
    expect(view.lastFrame()).not.toContain("compacted summary");
    view.unmount();
  });
});
