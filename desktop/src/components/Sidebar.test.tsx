// @vitest-environment jsdom

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Sidebar } from "./Sidebar";

beforeEach(() => {
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    value: () => ({
      matches: true,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    }),
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
        voiceRunning={false}
        newConversationDisabled={false}
        onCollapse={onCollapse}
      />,
    );

    const dialog = screen.getByRole("dialog", { name: "Thanh điều hướng" });
    const collapse = screen.getByRole("button", { name: "Thu gọn thanh bên" });
    await waitFor(() => expect(document.activeElement).toBe(collapse));

    const buttons = Array.from(
      dialog.querySelectorAll<HTMLButtonElement>("button:not([disabled])"),
    );
    buttons[buttons.length - 1]?.focus();
    await user.keyboard("{Tab}");
    expect(document.activeElement).toBe(collapse);

    await user.keyboard("{Escape}");
    expect(onCollapse).toHaveBeenCalledTimes(1);
  });
});
