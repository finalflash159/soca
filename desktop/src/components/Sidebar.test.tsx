// @vitest-environment jsdom

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./SessionList", () => ({
  SessionList: () => <button type="button">Phiên đã lưu</button>,
}));

import { Sidebar } from "./Sidebar";
import type { SessionHistoryState } from "@/engine/session-history";

const sessions: SessionHistoryState = {
  sessions: [],
  nextCursor: null,
  listState: "ready",
  listError: null,
  snapshotError: null,
  activeSessionId: null,
  active: null,
  persistence: "ram_only",
  autoOpenLast: false,
  busy: false,
  operation: null,
};

beforeEach(() => {
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    value: () => ({ matches: true, addEventListener: vi.fn(), removeEventListener: vi.fn() }),
  });
});

afterEach(cleanup);

describe("compact Sidebar", () => {
  it("acts as a dialog: focuses inside, traps Tab, and closes on Escape", async () => {
    const user = userEvent.setup();
    const onCollapse = vi.fn();
    render(
      <Sidebar
        page="chat"
        onNavigate={vi.fn()}
        onNewConversation={vi.fn()}
        sessions={sessions}
        connected
        starting={false}
        voiceRunning={false}
        sessionBusy={false}
        newConversationDisabled={false}
        onRestartEngine={vi.fn()}
        onCollapse={onCollapse}
        onOpenSession={vi.fn()}
        onRenameSession={vi.fn()}
        onDeleteSession={vi.fn()}
        onLoadMoreSessions={vi.fn()}
        onOpenSessionSettings={vi.fn()}
      />,
    );

    const dialog = screen.getByRole("dialog", { name: "Thanh điều hướng" });
    const collapse = screen.getByRole("button", { name: "Thu gọn thanh bên" });
    await waitFor(() => expect(document.activeElement).toBe(collapse));

    const buttons = Array.from(dialog.querySelectorAll<HTMLButtonElement>("button:not([disabled])"));
    buttons[buttons.length - 1]?.focus();
    await user.keyboard("{Tab}");
    expect(document.activeElement).toBe(collapse);

    await user.keyboard("{Escape}");
    expect(onCollapse).toHaveBeenCalledTimes(1);
  });
});
