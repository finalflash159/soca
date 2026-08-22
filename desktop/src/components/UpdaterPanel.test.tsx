// @vitest-environment jsdom

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

const updater = vi.hoisted(() => ({ check: vi.fn() }));
const process = vi.hoisted(() => ({ relaunch: vi.fn() }));

vi.mock("@tauri-apps/plugin-updater", () => updater);
vi.mock("@tauri-apps/plugin-process", () => process);

import { UpdaterPanel } from "./UpdaterPanel";

afterEach(() => {
  cleanup();
  updater.check.mockReset();
  process.relaunch.mockReset();
});

describe("UpdaterPanel", () => {
  it("states that the signed release is current", async () => {
    updater.check.mockResolvedValue(null);

    render(<UpdaterPanel />);
    await userEvent.setup().click(screen.getByRole("button", { name: "Kiểm tra" }));

    await waitFor(() => expect(screen.getByText("Bạn đang dùng bản mới nhất.")).not.toBeNull());
  });

  it("downloads an available update and relaunches only after installation", async () => {
    const update = {
      currentVersion: "0.1.0",
      version: "0.2.0",
      close: vi.fn(),
      downloadAndInstall: vi.fn(async (onEvent: (event: unknown) => void) => {
        onEvent({ event: "Started", data: { contentLength: 100 } });
        onEvent({ event: "Progress", data: { chunkLength: 100 } });
        onEvent({ event: "Finished" });
      }),
    };
    updater.check.mockResolvedValue(update);
    process.relaunch.mockResolvedValue(undefined);

    render(<UpdaterPanel />);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Kiểm tra" }));
    await screen.findByText("Sẵn sàng cài 0.2.0");
    await user.click(screen.getByRole("button", { name: "Cài và mở lại" }));

    await waitFor(() => expect(update.downloadAndInstall).toHaveBeenCalledOnce());
    await waitFor(() => expect(process.relaunch).toHaveBeenCalledOnce());
  });

  it("shows an unavailable updater configuration without inventing a successful check", async () => {
    updater.check.mockRejectedValue(new Error("Updater does not have any endpoints set."));

    render(<UpdaterPanel />);
    await userEvent.setup().click(screen.getByRole("button", { name: "Kiểm tra" }));

    await waitFor(() => {
      expect(screen.getByRole("alert").textContent).toContain("Updater does not have any endpoints set.");
    });
  });
});
