import React from "react";
import { render } from "ink-testing-library";
import { describe, expect, it } from "vitest";
import { TimelineLine } from "./Timeline.js";

describe("TimelineLine", () => {
  it("wraps SoCa responses in a subtle bordered surface", () => {
    const view = render(
      <TimelineLine entry={{ kind: "soca", text: "Câu trả lời có grounding." }} />,
    );

    const frame = view.lastFrame() ?? "";
    expect(frame).toContain("╭");
    expect(frame).toContain("(o> Câu trả lời có grounding.");
    expect(frame).toContain("╰");
    view.unmount();
  });

  it("keeps user messages in the existing unboxed style", () => {
    const view = render(
      <TimelineLine entry={{ kind: "user", text: "Câu hỏi của tôi" }} />,
    );

    const frame = view.lastFrame() ?? "";
    expect(frame).toContain("❯ Câu hỏi của tôi");
    expect(frame).not.toContain("╭");
    view.unmount();
  });

  it("renders structured sources once at the end of the answer", () => {
    const view = render(
      <TimelineLine
        entry={{
          kind: "soca",
          text: "Attention dùng query, key và value.",
          citations: [
            {
              label: "K1",
              path: "wiki/learning/attention.md",
              title: "Attention",
              line_start: 12,
              line_end: 18,
              source: "knowledge",
            },
          ],
        }}
      />,
    );

    const frame = view.lastFrame() ?? "";
    expect(frame).toContain("Attention dùng query, key và value.");
    expect(frame).toContain("── nguồn");
    expect(frame).toContain("K1  Attention · wiki/learning/attention.md:12-18");
    expect(frame.indexOf("── nguồn")).toBeGreaterThan(
      frame.indexOf("Attention dùng query, key và value."),
    );
    view.unmount();
  });
});
