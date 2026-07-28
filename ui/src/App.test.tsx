import React from "react";
import { render } from "ink-testing-library";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App.js";

const harness = vi.hoisted(() => ({
  sent: [] as Array<Record<string, unknown>>,
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
      }
    }

    stop(): void {}
  }

  return { EngineClient };
});

async function tick(): Promise<void> {
  await new Promise((resolve) => setImmediate(resolve));
}

describe("App slash command interaction", () => {
  beforeEach(() => {
    harness.sent.length = 0;
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
    expect(view.lastFrame()).toContain("/memory proposals");
    expect(view.lastFrame()).not.toContain("/inspect");

    for (let index = 0; index < 4; index += 1) {
      view.stdin.write("\u001b[B");
      await tick();
    }
    view.stdin.write("\r");
    await tick();

    expect(harness.sent).toContainEqual({ cmd: "context" });
    expect(view.lastFrame()).toContain("context breakdown");
    expect(view.lastFrame()).toContain("Knowledge retrieval");

    view.stdin.write("x");
    await tick();
    expect(view.lastFrame()).not.toContain("context breakdown");
    expect(view.lastFrame()).toContain("x");
    view.unmount();
  });

  it("shows an explicit empty state for memory proposals", async () => {
    const view = render(<App target="status" noModel />);
    await tick();

    view.stdin.write("/memory proposals");
    await tick();
    view.stdin.write("\r");
    await tick();

    expect(harness.sent).toContainEqual({ cmd: "memory_proposals" });
    expect(view.lastFrame()).toContain("No pending proposals");
    expect(view.lastFrame()).toContain("Esc to return");
    view.unmount();
  });
});
