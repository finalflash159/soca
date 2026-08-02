import React from "react";
import { render } from "ink-testing-library";
import { describe, expect, it } from "vitest";
import { TurnProgress } from "./TurnProgress.js";

describe("TurnProgress", () => {
  it("shows one concise active operation without a frame or history", () => {
    const view = render(
      <TurnProgress
        progress={{
          event: "turn_progress",
          surface: "chat",
          phase: "retrieval",
          operation: "tool:knowledge.search",
          status: "active",
        }}
      />,
    );

    const frame = view.lastFrame() ?? "";
    expect(frame).toContain("Running knowledge.search…");
    expect(frame).not.toContain("✓");
    expect(frame).not.toContain("╭─");
    view.unmount();
  });

  it("uses a short voice recognition label", () => {
    const view = render(
      <TurnProgress
        progress={{
          event: "turn_progress",
          surface: "voice",
          phase: "analyzing",
          operation: "speech_recognition",
          status: "active",
        }}
      />,
    );

    expect(view.lastFrame()).toContain("Transcribing…");
    view.unmount();
  });

  it("uses a short listening label while recording", () => {
    const view = render(
      <TurnProgress
        progress={{
          event: "turn_progress",
          surface: "voice",
          phase: "preparing",
          operation: "listening",
          status: "active",
        }}
      />,
    );

    expect(view.lastFrame()).toContain("Listening…");
    view.unmount();
  });
});
