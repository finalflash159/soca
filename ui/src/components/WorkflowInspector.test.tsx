import React from "react";
import { render } from "ink-testing-library";
import { describe, expect, it } from "vitest";
import type { WorkflowEvent } from "../protocol.js";
import { WorkflowInspector } from "./WorkflowInspector.js";

const events: WorkflowEvent[] = [
  {
    event: "turn_started",
    protocol_version: 2,
    session_id: "session",
    run_id: "run-1",
    goal_id: "goal-1",
    sequence: 0,
    surface: "chat",
    timestamp: "2026-07-30T00:00:00Z",
    node: "admit",
    status: "started",
    payload: {},
  },
  {
    event: "turn_terminal",
    protocol_version: 2,
    session_id: "session",
    run_id: "run-1",
    goal_id: "goal-1",
    sequence: 1,
    surface: "chat",
    timestamp: "2026-07-30T00:00:01Z",
    node: "finalize",
    status: "completed",
    payload: { terminal_status: "achieved" },
  },
];

describe("WorkflowInspector", () => {
  it("shows a collapsed terminal summary and expands the event trace", () => {
    let expanded = false;
    const view = render(
      <WorkflowInspector
        events={events}
        pendingAnswer=""
        terminalStatus="achieved"
        expanded={expanded}
        width={72}
        onToggle={() => {
          expanded = !expanded;
        }}
      />,
    );

    expect(view.lastFrame()).toContain("workflow");
    expect(view.lastFrame()).toContain("2 events");
    expect(view.lastFrame()).not.toContain("turn started");
    view.unmount();
  });
});
