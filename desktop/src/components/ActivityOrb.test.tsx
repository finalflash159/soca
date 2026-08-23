// @vitest-environment jsdom

import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ActivityOrb } from "./ActivityOrb";

describe("ActivityOrb", () => {
  it("renders a fixed cluster of separate decorative spheres for the real engine state", () => {
    const { container } = render(<ActivityOrb state="listening" size={96} />);
    const orb = container.querySelector(".activity-orb");

    expect(orb?.getAttribute("data-state")).toBe("listening");
    expect(orb?.getAttribute("data-visual")).toBe("static-sphere-cluster");
    expect(orb?.getAttribute("aria-hidden")).toBe("true");
    expect(orb?.querySelectorAll(".activity-orb__sphere")).toHaveLength(3);
    expect(orb?.querySelector(".activity-orb__satellite, .activity-orb__core")).toBeNull();
    expect(orb?.className).not.toContain("animate-");
  });
});
