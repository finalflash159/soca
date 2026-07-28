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
});
