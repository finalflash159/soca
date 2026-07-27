import React from "react";
import { render } from "ink-testing-library";
import { describe, expect, it } from "vitest";
import { MemoryProposalInbox, type MemoryProposalView } from "./MemoryProposalInbox.js";

const proposals: MemoryProposalView[] = [
  {
    id: "proposal-1",
    kind: "preference",
    statement: "first proposal",
    confidence: 0.8,
    createdAt: "2026-07-27T00:00:00Z",
  },
  {
    id: "proposal-2",
    kind: "project",
    statement: "second proposal",
    confidence: 0.8,
    createdAt: "2026-07-27T00:00:00Z",
  },
  {
    id: "proposal-3",
    kind: "stable_fact",
    statement: "third proposal",
    confidence: 0.8,
    createdAt: "2026-07-27T00:00:00Z",
  },
];

describe("MemoryProposalInbox", () => {
  it("clamps the selected proposal when the list shrinks", async () => {
    const view = render(
      <MemoryProposalInbox
        proposals={proposals}
        error=""
        onApprove={() => undefined}
        onReject={() => undefined}
        onClose={() => undefined}
      />,
    );

    await new Promise((resolve) => setImmediate(resolve));
    view.stdin.write("\u001b[B");
    view.stdin.write("\u001b[B");
    await new Promise((resolve) => setImmediate(resolve));
    expect(view.lastFrame()).toContain("third proposal");

    view.rerender(
      <MemoryProposalInbox
        proposals={proposals.slice(0, 2)}
        error=""
        onApprove={() => undefined}
        onReject={() => undefined}
        onClose={() => undefined}
      />,
    );
    await new Promise((resolve) => setImmediate(resolve));

    expect(view.lastFrame()).toContain("❯ project · second proposal");
    view.stdin.write("\u001b[A");
    await new Promise((resolve) => setImmediate(resolve));
    expect(view.lastFrame()).toContain("❯ preference · first proposal");
    view.unmount();
  });
});
