// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { initialVoice } from "@/engine/voice";

import { perceptualVoiceLevel, VoiceOrb } from "./VoiceOrb";

afterEach(cleanup);

describe("perceptualVoiceLevel", () => {
  it("keeps recorder silence visually still", () => {
    expect(perceptualVoiceLevel(0)).toBe(0);
    expect(perceptualVoiceLevel(0.00035)).toBe(0);
  });

  it("makes an observed close-mic speech envelope visibly responsive", () => {
    // RMS measured from the real default MacBook microphone during the
    // packaged-sidecar probe was approximately 0.003.
    expect(perceptualVoiceLevel(0.003)).toBeGreaterThan(0.15);
    expect(perceptualVoiceLevel(0.003)).toBeLessThan(1);
  });

  it("is bounded and monotonic", () => {
    expect(perceptualVoiceLevel(0.01)).toBeGreaterThan(perceptualVoiceLevel(0.003));
    expect(perceptualVoiceLevel(1)).toBe(1);
  });
});

describe("VoiceOrb surface", () => {
  it("renders one clipped core image and a non-image halo", () => {
    render(
      <VoiceOrb
        voice={{ ...initialVoice, phase: "listening", levels: [0.003] }}
        ready
        presentation="immersive"
      />,
    );

    const orb = screen.getByTestId("voice-orb");
    expect(orb.querySelectorAll("img")).toHaveLength(1);
    expect(orb.querySelector(".voice-orb__aura")?.tagName).toBe("SPAN");
    expect(orb.style.getPropertyValue("--voice-orb-level")).not.toBe("0");
  });
});
