// @vitest-environment jsdom

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { CitationChip } from "./CitationChip";
import { citationKey, type CitationPreviewState } from "@/engine/citation-preview";
import type { Citation } from "@/engine/conversation";

const citation: Citation = {
  label: "K1",
  path: "wiki/plan.md",
  title: "Kế hoạch",
  line_start: 2,
  line_end: 3,
  source: "knowledge",
};

function preview(status: CitationPreviewState["status"]): CitationPreviewState {
  return {
    requestId: "preview-1",
    status,
    title: "Kế hoạch",
    lineStart: 2,
    lineEnd: 3,
    passage: status === "missing" ? null : "Mục tiêu\nBước một",
    errorCode: status === "missing" ? "source_missing" : null,
  };
}

afterEach(cleanup);

describe("CitationChip", () => {
  it("opens a keyboard-dismissible current evidence dialog", async () => {
    const user = userEvent.setup();
    const onRequestPreview = vi.fn().mockResolvedValue(true);
    render(
      <CitationChip
        citation={citation}
        previews={{ [citationKey(citation)]: preview("current") }}
        onRequestPreview={onRequestPreview}
      />,
    );

    const trigger = screen.getByRole("button", { name: /kiểm tra nguồn k1/i });
    await user.click(trigger);

    expect(onRequestPreview).toHaveBeenCalledWith(citation);
    expect(screen.getByRole("dialog").textContent).toContain("Nguồn hiện tại khớp");
    expect(screen.getByRole("dialog").textContent).toContain("Mục tiêu\nBước một");

    await user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(document.activeElement).toBe(trigger);
  });

  it("makes a missing source explicit instead of showing a fake preview", async () => {
    const user = userEvent.setup();
    render(
      <CitationChip
        citation={citation}
        previews={{ [citationKey(citation)]: preview("missing") }}
        onRequestPreview={vi.fn().mockResolvedValue(true)}
      />,
    );

    await user.click(screen.getByRole("button", { name: /kiểm tra nguồn k1/i }));

    expect(screen.getByRole("alert").textContent).toContain("Không còn tìm thấy tệp nguồn");
    expect(screen.queryByText("Mục tiêu\nBước một")).toBeNull();
  });
});
