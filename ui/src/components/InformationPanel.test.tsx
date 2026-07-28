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
        profiles={[]}
        knowledge={null}
        memoryCompaction={null}
      />,
    );

    expect(panel.lastFrame()).toContain("╭─");
    expect(panel.lastFrame()).toContain(title);
    expect(panel.lastFrame()).toContain("╰");
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
        profiles={[]}
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

  it("explains that an empty summary keeps the original history", () => {
    const panel = render(
      <InformationPanel
        view="compaction"
        width={78}
        context={null}
        memory={null}
        usage={null}
        profiles={[]}
        knowledge={null}
        memoryCompaction={{
          event: "memory_compaction",
          status: "failed",
          detail: "empty_summary_artifact",
          complete_turns: 5,
          minimum_complete_turns: 5,
        }}
      />,
    );

    expect(panel.lastFrame()).toContain("lịch sử gốc được giữ nguyên");
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
        profiles={[]}
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
