// @vitest-environment jsdom

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import axe from "axe-core";
import { afterEach, describe, expect, it, vi } from "vitest";

import { StartupView } from "./StartupView";

afterEach(cleanup);

describe("StartupView", () => {
  it("announces and focuses a startup failure so recovery begins in context", async () => {
    render(
      <StartupView
        starting={false}
        problem="Không tìm thấy bundled engine."
        onStart={vi.fn()}
      />,
    );

    const problem = screen.getByRole("alert");
    expect(problem.textContent).toContain("Không tìm thấy bundled engine.");
    await waitFor(() => expect(document.activeElement).toBe(problem));
  });

  it("exposes the explicit engine recovery path as an accessible disclosure", async () => {
    const onStart = vi.fn();
    render(<StartupView starting={false} problem={null} onStart={onStart} />);
    const user = userEvent.setup();
    const disclosure = screen.getByRole("button", { name: "Engine không chạy được?" });

    expect(disclosure.getAttribute("aria-expanded")).toBe("false");
    await user.click(disclosure);
    expect(disclosure.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByRole("region", { name: "Khôi phục engine" })).not.toBeNull();

    await user.type(screen.getByLabelText("Lệnh chạy engine"), "/tmp/soca-recovery");
    await user.click(screen.getByRole("button", { name: "Dùng engine này" }));

    expect(onStart).toHaveBeenCalledWith("/tmp/soca-recovery");
  });

  it("has no automated WCAG A/AA violations in its recovery-ready state", async () => {
    render(<StartupView starting={false} problem={null} onStart={vi.fn()} />);
    document.documentElement.lang = "vi";
    document.title = "SoCa";
    const results = await axe.run(document, {
      runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21aa"] },
      rules: { "color-contrast": { enabled: false } },
    });

    expect(results.violations).toEqual([]);
  });
});
