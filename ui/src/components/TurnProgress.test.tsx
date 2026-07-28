import React from "react";
import { render } from "ink-testing-library";
import { describe, expect, it } from "vitest";
import { TurnProgress } from "./TurnProgress.js";

describe("TurnProgress", () => {
  it("shows the active real phase and completed phase trail", () => {
    const view = render(
      <TurnProgress
        width={88}
        progress={{
          event: "turn_progress",
          surface: "chat",
          phase: "retrieval",
          operation: "tool:knowledge.search",
          status: "active",
        }}
        completed={["preparing", "analyzing", "routing"]}
      />,
    );

    const frame = view.lastFrame() ?? "";
    expect(frame).toContain("SoCa đang xử lý");
    expect(frame).toContain("Tra cứu knowledge");
    expect(frame).toContain("đang chạy knowledge.search");
    expect(frame).toContain("✓ Chuẩn bị");
    expect(frame).toContain("✓ Phân tích yêu cầu");
    expect(frame).toContain("✓ Định tuyến");
    expect(frame).toContain("╭─");
    view.unmount();
  });

  it("describes voice recognition separately", () => {
    const view = render(
      <TurnProgress
        width={78}
        progress={{
          event: "turn_progress",
          surface: "voice",
          phase: "analyzing",
          operation: "speech_recognition",
          status: "active",
        }}
        completed={[]}
      />,
    );

    expect(view.lastFrame()).toContain("chuyển âm thanh thành văn bản");
    expect(view.lastFrame()).toContain("voice");
    view.unmount();
  });
});
