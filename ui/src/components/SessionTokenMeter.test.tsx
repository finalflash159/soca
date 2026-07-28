import React from "react";
import { render } from "ink-testing-library";
import { describe, expect, it } from "vitest";
import { SessionTokenMeter } from "./SessionTokenMeter.js";

describe("SessionTokenMeter", () => {
  it("renders current and hard-limit session tokens", () => {
    const view = render(
      <SessionTokenMeter
        stats={{
          current_tokens: 1500,
          rendered_tokens: 1500,
          hard_limit_tokens: 16384,
          high_watermark_tokens: 15000,
          target_tokens: 12000,
          summary_tokens: 200,
          recent_tokens: 1300,
          turn_count: 4,
          complete_turn_count: 4,
          summary_generation: 1,
          pending_compaction: false,
          worker_state: "idle",
        }}
      />,
    );

    expect(view.lastFrame()).toContain("~1.50k / 16.4k tok");
    view.unmount();
  });
});
