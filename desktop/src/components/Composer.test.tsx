// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { Composer } from "./Composer";

afterEach(cleanup);

describe("Composer", () => {
  it("uses Escape only to dismiss a palette and keeps the drafted message", async () => {
    const user = userEvent.setup();
    render(
      <Composer
        connected
        documents={[]}
        model="model-test"
        onSend={vi.fn()}
        onCommand={vi.fn()}
        onEnterVoiceMode={vi.fn()}
        onOpenSettings={vi.fn()}
      />,
    );

    const input = screen.getByRole("textbox", { name: "Message" });
    await user.type(input, "Bản nháp cần giữ lại");
    await user.keyboard("{Escape}");

    expect((input as HTMLTextAreaElement).value).toBe("Bản nháp cần giữ lại");
    expect(screen.getByRole("button", { name: "Nhắc tài liệu phiên này đã thấy" })).not.toBeNull();
  });
});
