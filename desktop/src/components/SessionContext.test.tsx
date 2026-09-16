// @vitest-environment jsdom

import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { SessionContext } from "./SessionContext";
import type { SessionState } from "@/engine/session";
import type { SessionHistoryState } from "@/engine/session-history";

const session: SessionState = {
  context: {
    ready: true,
    estimated: false,
    error: null,
    errorDetail: null,
    residentPromptTokens: 1_200,
    outputReserveTokens: 800,
    modelContextTokens: 8_192,
    inputBudgetTokens: 7_392,
    availableDynamicTokens: 6_192,
    observedPromptTokens: 1_200,
    providerPromptTokens: 1_200,
    components: [],
  },
  usage: {
    turns: 2,
    llmTurns: 2,
    promptTokens: 1_200,
    completionTokens: 320,
    meanTtftMs: 210,
    meanTokensPerSecond: 28,
  },
};

const saved = {
  sessionId: "11111111-1111-4111-8111-111111111111",
  title: "Kế hoạch tuần sau",
  preview: "",
  updatedAt: "2026-08-26T10:00:00Z",
  revision: 1,
  turnCount: 2,
  checkpointOnly: false,
};

const history: SessionHistoryState = {
  sessions: [saved],
  nextCursor: null,
  listState: "ready",
  listError: null,
  snapshotError: null,
  activeSessionId: saved.sessionId,
  active: saved,
  persistence: "local_resumable",
  autoOpenLast: true,
  busy: false,
  operation: null,
};

afterEach(cleanup);

describe("SessionContext", () => {
  it("keeps context and saved sessions on the active conversation surface", async () => {
    const user = userEvent.setup();
    const onRefresh = vi.fn();
    const onOpenSession = vi.fn();

    render(
      <SessionContext
        session={session}
        history={history}
        connected
        busy={false}
        onRefresh={onRefresh}
        onOpenSession={onOpenSession}
        onRenameSession={vi.fn()}
        onDeleteSession={vi.fn()}
        onLoadMoreSessions={vi.fn()}
        onOpenSessionSettings={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: /ngữ cảnh/i }));
    expect(onRefresh).toHaveBeenCalledOnce();
    expect(screen.getByText("Ngân sách ngữ cảnh")).toBeTruthy();
    expect(screen.getByText("Mức dùng")).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "Phiên đã lưu" }));
    const dialog = await screen.findByRole("dialog", { name: "Phiên đã lưu" });
    await user.click(
      within(dialog).getByRole("button", { name: /^kế hoạch tuần sau/i }),
    );
    expect(onOpenSession).toHaveBeenCalledWith(saved);
  });
});
