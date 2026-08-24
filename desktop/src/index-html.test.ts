import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

describe("desktop document metadata", () => {
  it("declares Vietnamese as the page language and removes the Vite placeholder title", async () => {
    const html = await readFile(fileURLToPath(new URL("../index.html", import.meta.url)), "utf8");
    expect(html).toContain('<html lang="vi"');
    expect(html).toContain('<link rel="icon" type="image/png" href="/soca-parrot.png" />');
    expect(html).toContain("<title>SoCa</title>");
    expect(html).not.toContain("Tauri + React + Typescript");
    expect(html).not.toContain("vite.svg");
  });
});
