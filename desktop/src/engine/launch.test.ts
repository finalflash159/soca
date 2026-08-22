// @vitest-environment jsdom

import { describe, expect, it, vi } from "vitest";

async function loadWithRoot(root: string) {
  vi.stubGlobal("__SOCA_CHECKOUT_ROOT__", root);
  vi.resetModules();
  const module = await import("./launch");
  return module.launchOptions();
}

describe("launchOptions", () => {
  it("pins PYTHONPATH to the checkout the bundle was built from", async () => {
    // The regression this exists for: `soca` on PATH is an editable install
    // pinned to one checkout, so without this the engine ran another worktree's
    // Python and every fix on this branch looked like it had not been applied.
    const options = await loadWithRoot("/repo/worktrees/desktop-app");
    expect(options.env).toEqual({ PYTHONPATH: "/repo/worktrees/desktop-app" });
  });

  it("uses `soca` only for a checkout-backed development build", async () => {
    expect((await loadWithRoot("/repo")).program).toBe("soca");
    expect((await loadWithRoot("")).program).toBeUndefined();
  });

  it("passes the explicit privacy mode to the sidecar", async () => {
    const module = await import("./launch");
    expect(module.launchOptions("local_resumable").args).toEqual([
      "--session-persistence",
      "local_resumable",
    ]);
    expect(module.launchOptions("ram_only").args).toEqual(["--session-persistence", "ram_only"]);
  });

  it("sets no environment in a packaged build", async () => {
    // A shipped app has no checkout, and must never carry a developer's path.
    expect(await loadWithRoot("")).toEqual({
      args: ["--session-persistence", "ram_only"],
    });
  });

  it("reports an unavailable WebView store instead of claiming the launch choice was saved", async () => {
    const module = await import("./launch");
    const original = Object.getOwnPropertyDescriptor(window, "localStorage");
    try {
      Object.defineProperty(window, "localStorage", {
        configurable: true,
        get: () => {
          throw new DOMException("blocked", "SecurityError");
        },
      });

      expect(module.saveSessionPersistence("local_resumable")).toBe(false);
    } finally {
      if (original !== undefined) Object.defineProperty(window, "localStorage", original);
    }
  });
});
