import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

// @ts-expect-error process is a nodejs global
const host = process.env.TAURI_DEV_HOST;
const devEngineProgram =
  process.platform === "win32"
    ? path.resolve(__dirname, "..", ".venv", "Scripts", "soca.exe")
    : path.resolve(__dirname, "..", ".venv", "bin", "soca");

// https://vite.dev/config/
export default defineConfig(async ({ command }) => ({
  plugins: [react(), tailwindcss()],

  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },

  define: {
    // The checkout this bundle was built from, so a dev run launches the engine
    // from *this* source tree rather than whichever one the editable `soca`
    // install happens to pin. See `src/engine/launch.ts`. A packaged build ships
    // its own sidecar and must not point at a developer's path.
    __SOCA_CHECKOUT_ROOT__: JSON.stringify(
      command === "serve" ? path.resolve(__dirname, "..") : "",
    ),
    // Development must invoke the virtual environment belonging to this
    // checkout. `soca` is not guaranteed to be on the GUI process PATH, and a
    // global editable install can point at a different worktree.
    __SOCA_DEV_ENGINE_PROGRAM__: JSON.stringify(command === "serve" ? devEngineProgram : ""),
  },

  build: {
    // The transcript renderer pulls Markdown, syntax and math support. Keep
    // those independently cacheable so opening voice or chat does not make the
    // app shell cross a single opaque bundle budget.
    chunkSizeWarningLimit: 500,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes("node_modules")) return undefined;
          if (id.includes("/katex/")) return "math-typesetting";
          if (
            id.includes("/rehype-highlight/") ||
            id.includes("/lowlight/") ||
            id.includes("/highlight.js/")
          ) {
            return "syntax-highlighting";
          }
          if (
            [
              "/react-markdown/",
              "/remark-gfm/",
              "/remark-math/",
              "/rehype-katex/",
              "/unified/",
              "/remark-parse/",
              "/remark-rehype/",
              "/mdast-",
              "/micromark",
              "/hast-",
              "/vfile/",
              "/property-information/",
            ].some((segment) => id.includes(segment))
          ) {
            return "markdown-rendering";
          }
          return undefined;
        },
      },
    },
  },

  // Vite options tailored for Tauri development and only applied in `tauri dev` or `tauri build`
  //
  // 1. prevent Vite from obscuring rust errors
  clearScreen: false,
  // 2. tauri expects a fixed port, fail if that port is not available
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? {
          protocol: "ws",
          host,
          port: 1421,
        }
      : undefined,
    watch: {
      // 3. tell Vite to ignore watching `src-tauri`
      ignored: ["**/src-tauri/**"],
    },
  },
}));
