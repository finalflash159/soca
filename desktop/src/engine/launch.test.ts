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

  it("always launches `soca`", async () => {
    expect((await loadWithRoot("/repo")).program).toBe("soca");
    expect((await loadWithRoot("")).program).toBe("soca");
  });

  it("sets no environment in a packaged build", async () => {
    // A shipped app has no checkout, and must never carry a developer's path.
    expect(await loadWithRoot("")).toEqual({ program: "soca" });
  });
});
