import { describe, expect, it } from "vitest";
import {
  canonicalCommand,
  filterSlashCommands,
  SLASH_COMMANDS,
} from "./keymap.js";

describe("slash command registry", () => {
  it("shows every canonical command for a bare slash", () => {
    expect(filterSlashCommands("/")).toHaveLength(SLASH_COMMANDS.length);
    expect(filterSlashCommands("/").map((command) => command.value)).toContain(
      "/context",
    );
    expect(filterSlashCommands("/").map((command) => command.value)).toContain(
      "/memory proposals",
    );
    expect(filterSlashCommands("/").map((command) => command.value)).not.toContain(
      "/inspect",
    );
  });

  it("filters nested commands as the user types", () => {
    expect(filterSlashCommands("/memory compact s").map((item) => item.value)).toEqual([
      "/memory compact status",
    ]);
    expect(filterSlashCommands("/cont").map((item) => item.value)).toEqual([
      "/context",
    ]);
  });

  it("keeps aliases canonical without duplicating palette rows", () => {
    expect(filterSlashCommands("/s").map((item) => item.value)).toEqual([
      "/settings",
    ]);
    expect(canonicalCommand("/EXIT")).toBe("/quit");
  });
});
