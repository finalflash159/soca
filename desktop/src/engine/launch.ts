import type { LaunchOptions } from "./useEngine";

export type LaunchSessionPersistence = "ram_only" | "local_resumable";

export const SESSION_PERSISTENCE_STORAGE_KEY = "soca.session-persistence.v1";

/**
 * Absolute path of the checkout this bundle was built from, injected by
 * `vite.config.ts`. Empty in a packaged build, where there is no checkout.
 */
declare const __SOCA_CHECKOUT_ROOT__: string;

/** Pin development to this checkout; packaged builds resolve the bundled sidecar. */
function storage(): Storage | null {
  return typeof window === "undefined" ? null : window.localStorage;
}

export function savedSessionPersistence(): LaunchSessionPersistence {
  try {
    return storage()?.getItem(SESSION_PERSISTENCE_STORAGE_KEY) === "local_resumable"
      ? "local_resumable"
      : "ram_only";
  } catch {
    // A blocked WebView store must never make persistence opt-in by accident.
    return "ram_only";
  }
}

/** Stores only the explicit sidecar launch choice, never a transcript or session ID. */
export function saveSessionPersistence(persistence: LaunchSessionPersistence): boolean {
  try {
    const localStorage = storage();
    if (localStorage === null) {
      return false;
    }
    localStorage.setItem(SESSION_PERSISTENCE_STORAGE_KEY, persistence);
    return localStorage.getItem(SESSION_PERSISTENCE_STORAGE_KEY) === persistence;
  } catch {
    return false;
  }
}

export function launchOptions(
  persistence: LaunchSessionPersistence = savedSessionPersistence(),
): LaunchOptions {
  const root = typeof __SOCA_CHECKOUT_ROOT__ === "string" ? __SOCA_CHECKOUT_ROOT__ : "";
  const args = ["--session-persistence", persistence];
  if (root === "") {
    // Leaving `program` unset is intentional: Rust resolves only the Tauri
    // externalBin sidecar for a release build and never guesses from PATH.
    return { args };
  }
  return { program: "soca", args, env: { PYTHONPATH: root } };
}
