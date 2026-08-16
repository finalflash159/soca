import type { LaunchOptions } from "./useEngine";

/**
 * Absolute path of the checkout this bundle was built from, injected by
 * `vite.config.ts`. Empty in a packaged build, where there is no checkout.
 */
declare const __SOCA_CHECKOUT_ROOT__: string;

/**
 * How to launch `soca engine`.
 *
 * `soca` is resolved on PATH, which in this repo is an editable install whose
 * `.pth` pins **one** checkout. A dev running the app from a git worktree gets
 * that other checkout's Python source: the engine silently ignores every change
 * on the current branch, and a bug that was fixed here still reproduces.
 *
 * `PYTHONPATH` fixes it deterministically. It is searched before site-packages,
 * so it wins over the editable install without touching the shared venv — which
 * matters because other sessions are working in the checkout that `.pth` names.
 *
 * Relying on the child's cwd would also work, since Python puts cwd first, but
 * only by accident of where the app happens to be launched from. This is the
 * same rule stated once, in the open.
 */
export function launchOptions(): LaunchOptions {
  const root = typeof __SOCA_CHECKOUT_ROOT__ === "string" ? __SOCA_CHECKOUT_ROOT__ : "";
  if (root === "") {
    return { program: "soca" };
  }
  return { program: "soca", env: { PYTHONPATH: root } };
}
