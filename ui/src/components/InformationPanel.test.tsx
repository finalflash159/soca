import React from "react";
import { render } from "ink-testing-library";
import { describe, expect, it } from "vitest";
import type { InfoView } from "../store.js";
import { InformationPanel } from "./InformationPanel.js";

const TITLES: Array<[InfoView, string]> = [
  ["status", "runtime status"],
  ["context", "context & usage"],
  ["memory", "working memory"],
  ["compaction", "memory compaction"],
  ["compacted_summary", "compacted summary"],
];

describe("InformationPanel", () => {
  it.each(TITLES)("renders a visible border for %s", (view, title) => {
    const panel = render(
      <InformationPanel
        view={view}
        width={78}
        context={null}
        memory={null}
        usage={null}
        stack={{}}
        knowledge={null}
        memoryCompaction={null}
      />,
    );

    expect(panel.lastFrame()).toContain("╭─");
    expect(panel.lastFrame()).toContain(title);
    expect(panel.lastFrame()).toContain("╰");
    panel.unmount();
  });

  it("shows active chat and voice runtimes without a static profile label", () => {
    const panel = render(
      <InformationPanel
        view="status"
        width={100}
        context={null}
        memory={null}
        usage={null}
        stack={{
          asr: "phowhisper_small",
          llm: "arcee_vylinh_3b_q4_k_m",
          tts: "valtec_multispeaker",
          voice: "NF",
        }}
        knowledge={{
          sparse_state: "ready",
          dense_state: "ready",
          revision: 1,
          documents: 6,
          chunks: 23,
        }}
        llmConfig={{
          event: "llm_config",
          backend: "remote",
          provider: "openrouter",
          model: "some/model",
          max_tokens: 4096,
          effective_max_tokens: 4096,
          reasoning_enabled: false,
          temperature: 0.2,
          top_p: 0.95,
          pricing_as_of: "test",
          pricing: null,
          context_length: 32768,
        }}
        memoryCompaction={null}
      />,
    );

    const frame = panel.lastFrame() ?? "";
    expect(frame).toContain("Chat runtime");
    expect(frame).toContain("remote · openrouter · some/model");
    expect(frame).toContain("Voice runtime");
    expect(frame).toContain("ASR phowhisper_small");
    expect(frame).not.toContain("baseline");
    panel.unmount();
  });

  it("shows the explicit five-turn manual compaction gate", () => {
    const panel = render(
      <InformationPanel
        view="compaction"
        width={78}
        context={null}
        memory={null}
        usage={null}
        stack={{}}
        knowledge={null}
        memoryCompaction={{
          event: "memory_compaction",
          status: "noop",
          detail: "not_enough_complete_turns",
          complete_turns: 3,
          minimum_complete_turns: 5,
        }}
      />,
    );

    expect(panel.lastFrame()).toContain("ít nhất 5 lượt");
    expect(panel.lastFrame()).toContain("3/5");
    panel.unmount();
  });

  it("explains when an empty rolling summary would drop previous state", () => {
    const panel = render(
      <InformationPanel
        view="compaction"
        width={78}
        context={null}
        memory={null}
        usage={null}
        stack={{}}
        knowledge={null}
        memoryCompaction={{
          event: "memory_compaction",
          status: "failed",
          detail: "empty_continuity_summary",
          complete_turns: 5,
          minimum_complete_turns: 5,
        }}
      />,
    );

    expect(panel.lastFrame()).toContain("summary rỗng");
    expect(panel.lastFrame()).toContain("giữ nguyên");
    panel.unmount();
  });

  it("reports an empty first summary as failure without deleting history", () => {
    const panel = render(
      <InformationPanel
        view="compaction"
        width={88}
        context={null}
        memory={null}
        usage={null}
        stack={{}}
        knowledge={null}
        memoryCompaction={{
          event: "memory_compaction",
          status: "failed",
          detail: "empty_continuity_summary",
          before_tokens: 1200,
          after_tokens: 900,
          compacted_turns: 1,
        }}
      />,
    );

    const frame = panel.lastFrame() ?? "";
    expect(frame).toContain("Compact: failed");
    expect(frame).toContain("lịch sử gốc được giữ nguyên");
    expect(frame).not.toContain("/compact-show");
    panel.unmount();
  });

  it("combines cumulative usage with current context without helper notes", () => {
    const panel = render(
      <InformationPanel
        view="context"
        width={100}
        context={{
          event: "context",
          estimated: true,
          token_counter: "utf8_bytes_div_4",
          session: null,
          resident_prompt_tokens: 1400,
          output_reserve_tokens: 4096,
          model_context_tokens: 32768,
          available_dynamic_tokens: 27272,
          components: [],
        }}
        memory={null}
        usage={{
          event: "usage",
          turns: 1,
          llm_turns: 1,
          prompt_tokens: 682,
          completion_tokens: 128,
          mean_ttft_ms: 6354,
          mean_tokens_per_second: 20.1,
        }}
        stack={{}}
        knowledge={null}
        memoryCompaction={null}
      />,
    );

    const frame = panel.lastFrame() ?? "";
    expect(frame).toContain("Usage LLM tích lũy");
    expect(frame).toContain("1 lượt hội thoại · 1 lần gọi LLM");
    expect(frame).toContain("682 prompt · 128 completion · 810 total");
    expect(frame).toContain("TTFT 6.35 s · 20.1 tok/s");
    expect(frame).not.toContain("ước lượng UTF-8/4");
    expect(frame).not.toContain("bắt đầu nhập");
    expect(frame).not.toContain("Đây là usage tích lũy");
    panel.unmount();
  });
});
