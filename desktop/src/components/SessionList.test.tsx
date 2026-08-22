// @vitest-environment jsdom

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import axe from "axe-core";
import type { ComponentProps } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { SessionList } from "./SessionList";
import type { SessionHistoryState } from "@/engine/session-history";

const first = {
  sessionId: "11111111-1111-4111-8111-111111111111",
  title: "Nghiên cứu bờ biển miền Trung",
  preview: "",
  updatedAt: "2026-08-22T10:00:00Z",
  revision: 3,
  turnCount: 4,
  checkpointOnly: false,
};

const second = {
  sessionId: "22222222-2222-4222-8222-222222222222",
  title: "Kế hoạch tuần sau",
  preview: "",
  updatedAt: "2026-08-21T10:00:00Z",
  revision: 1,
  turnCount: 0,
  checkpointOnly: true,
};

const history: SessionHistoryState = {
  sessions: [first, second],
  nextCursor: null,
  listState: "ready",
  listError: null,
  snapshotError: null,
  activeSessionId: first.sessionId,
  active: first,
  persistence: "local_resumable",
  autoOpenLast: false,
  busy: false,
  operation: null,
};

function renderList(overrides: Partial<ComponentProps<typeof SessionList>> = {}) {
  const props = {
    history,
    disabled: false,
    onOpen: vi.fn(),
    onRename: vi.fn(),
    onDelete: vi.fn(),
    onLoadMore: vi.fn(),
    onOpenSettings: vi.fn(),
    ...overrides,
  };
  const view = render(<SessionList {...props} />);
  return { props, ...view };
}

afterEach(cleanup);

describe("SessionList", () => {
  it("opens a non-active saved session through its named row", async () => {
    const user = userEvent.setup();
    const { props } = renderList();

    await user.click(screen.getByRole("button", { name: /^kế hoạch tuần sau/i }));

    expect(props.onOpen).toHaveBeenCalledWith(second);
    expect(screen.getByRole("button", { name: /^nghiên cứu bờ biển/i }).getAttribute("aria-current")).toBe("page");
  });

  it("supports keyboard rename and Escape cancellation without touching the chat composer", async () => {
    const user = userEvent.setup();
    const { props } = renderList();

    const actions = screen.getByRole("button", { name: /thao tác cho phiên kế hoạch/i });
    actions.focus();
    await user.keyboard("{ArrowDown}");
    await user.click(await screen.findByRole("menuitem", { name: /đổi tên/i }));

    const input = screen.getByRole("textbox", { name: /đổi tên phiên kế hoạch/i });
    expect(document.activeElement).toBe(input);
    await user.clear(input);
    await user.type(input, "Kế hoạch đã sửa");
    await user.keyboard("{Escape}");

    expect(props.onRename).not.toHaveBeenCalled();
    expect(screen.queryByRole("textbox", { name: /đổi tên phiên/i })).toBeNull();
  });

  it("requires a permanent-delete confirmation with the exact session and turn count", async () => {
    const user = userEvent.setup();
    const { props } = renderList();

    const actions = screen.getByRole("button", { name: /thao tác cho phiên nghiên cứu/i });
    actions.focus();
    await user.keyboard("{ArrowDown}");
    await user.click(await screen.findByRole("menuitem", { name: /xóa vĩnh viễn/i }));

    expect(screen.getByRole("dialog").textContent).toContain("Nghiên cứu bờ biển miền Trung");
    expect(screen.getByRole("dialog").textContent).toContain("4 lượt trò chuyện");
    await user.click(screen.getByRole("button", { name: "Xóa vĩnh viễn" }));

    expect(props.onDelete).toHaveBeenCalledWith(first);
  });

  it("has no automated WCAG A/AA violations in its ready state", async () => {
    renderList();
    document.documentElement.lang = "vi";
    document.title = "Sơn Ca";
    const results = await axe.run(document, {
      runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21aa"] },
      rules: { "color-contrast": { enabled: false } },
    });
    expect(results.violations).toEqual([]);
  });

  it("moves focus to the replacement active session after a confirmed delete", async () => {
    const user = userEvent.setup();
    const { props, rerender } = renderList();

    await user.click(screen.getByRole("button", { name: /thao tác cho phiên nghiên cứu/i }));
    await user.click(await screen.findByRole("menuitem", { name: /xóa vĩnh viễn/i }));
    await user.click(screen.getByRole("button", { name: "Xóa vĩnh viễn" }));

    rerender(
      <SessionList
        {...props}
        history={{
          ...history,
          sessions: [second],
          activeSessionId: second.sessionId,
          active: second,
          operation: {
            requestId: "delete-1",
            action: "delete",
            status: "completed",
            sessionId: second.sessionId,
            revision: second.revision,
            errorCode: null,
          },
        }}
      />,
    );

    await waitFor(() => {
      expect(document.activeElement).toBe(screen.getByRole("button", { name: /^kế hoạch tuần sau/i }));
    });
  });
});
