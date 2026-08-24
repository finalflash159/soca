// @vitest-environment jsdom

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { BrandMark } from "./BrandMark";

describe("BrandMark", () => {
  it("keeps the official parrot decorative beside the SoCa name", () => {
    render(<BrandMark />);

    expect(screen.getByText("SoCa").hidden).toBe(false);
    expect(screen.getByText("🦜").getAttribute("aria-hidden")).toBe("true");
  });
});
