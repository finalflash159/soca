import path from "node:path";
import { defineConfig } from "vitest/config";

const here = import.meta.dirname;

/**
 * Scoped to `desktop/src` on purpose.
 *
 * Without an explicit root vitest walks up and collects the Ink TUI suite in
 * `ui/`, which has its own toolchain and its own install. The two surfaces
 * share the engine protocol, not a test runner.
 */
export default defineConfig({
  resolve: {
    alias: { "@": path.resolve(here, "./src") },
  },
  test: {
    root: here,
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
  },
});
